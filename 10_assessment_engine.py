"""Evidence-grounded exam generation and full-attempt grading for AS."""

import json
import re
import requests


GRADING_MODES = {
    "Strict": (
        "Require every material rubric point and source-specific key terminology. "
        "Penalize omissions and imprecise statements heavily."
    ),
    "Balanced": (
        "Require the correct meaning and central key terms. Accept reasonable "
        "paraphrasing and alternative valid explanations."
    ),
    "Flexible": (
        "Accept any semantically correct explanation supported by the evidence, "
        "even when the wording differs substantially from the reference answer."
    ),
}


def _ask(api_key, model_name, system, user, temperature=0):
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=150,
    )
    response.raise_for_status()
    choices = response.json().get("choices", [])
    if not choices:
        raise RuntimeError("The grading model returned no completion")
    return choices[0]["message"]["content"]


def _json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def _verified_evidence_ids(returned_ids, allowed_ids):
    """Reject invented evidence identifiers in grading output."""
    allowed = {str(item) for item in allowed_ids}
    returned = {str(item) for item in returned_ids}
    invalid = sorted(returned - allowed)
    if invalid:
        raise ValueError(f"Grading returned unverified evidence IDs: {invalid}")
    return sorted(returned or allowed)


def validate_exam(exam, requested_total=None):
    """Validate identity, types, marks, answers, and rubrics before an exam starts."""
    if not isinstance(exam, dict) or not isinstance(exam.get("questions"), list):
        raise ValueError("The generated exam has an invalid structure")
    questions = exam["questions"]
    if not questions:
        raise ValueError("The generated exam contains no questions")
    allowed_types = {"Essay", "MCQ", "True/False", "Explain Why"}
    ids = set()
    total = 0
    for index, question in enumerate(questions, start=1):
        required = {"id", "type", "question", "marks", "answer", "rubric", "evidence_ids"}
        if not required.issubset(question):
            raise ValueError(f"Question {index} is missing required fields")
        qid = str(question["id"]).strip()
        if not qid or qid in ids:
            raise ValueError("Question identifiers must be non-empty and unique")
        ids.add(qid)
        if question["type"] not in allowed_types:
            raise ValueError(f"Unsupported question type: {question['type']}")
        marks = int(question["marks"])
        if marks <= 0:
            raise ValueError("Every question must have positive marks")
        question["marks"] = marks
        total += marks
        if not str(question["question"]).strip() or not str(question["answer"]).strip():
            raise ValueError("Every question needs prompt text and a reference answer")
        if not isinstance(question["rubric"], list) or not question["rubric"]:
            raise ValueError("Every question needs at least one rubric point")
        if question["type"] == "MCQ":
            options = question.get("options", [])
            if not isinstance(options, list) or len(options) < 3:
                raise ValueError("Every MCQ needs at least three options")
            if str(question["answer"]).strip() not in [str(option).strip() for option in options]:
                raise ValueError("The MCQ reference answer must exactly match one option")
        if question["type"] == "True/False":
            normalized = str(question["answer"]).strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError("True/False answers must be True or False")
            question["answer"] = normalized.title()
    expected = int(requested_total if requested_total is not None else exam.get("total_marks", total))
    if total != expected:
        raise ValueError(f"Question marks sum to {total}, not the requested {expected}")
    exam["total_marks"] = total
    return exam


def generate_exam(context, topic, total_marks, counts, difficulty, api_key, model_name):
    count_instructions = {
        key: int(value) for key, value in counts.items() if int(value) > 0
    }
    if not count_instructions:
        raise ValueError("Choose at least one question")
    system = """You create rigorous exams using only supplied evidence. Return JSON only.
Schema: {"title":"...","instructions":"...","total_marks":50,"questions":[{"id":"Q1","type":"Essay|MCQ|True/False|Explain Why","question":"...","marks":10,"options":[],"answer":"...","rubric":["specific scoring point"],"evidence_ids":["System 1","Upload 2"],"topic_tag":"..."}]}.
Generate exactly the requested count for each question type. Marks must sum exactly to the requested total. Each question must be independently answerable from evidence. Do not expose answers in question wording. Do not invent facts or evidence IDs. MCQs need one correct answer whose text exactly matches one option. True/False reference answers must be exactly True or False. Rubrics must be concrete enough for consistent grading."""
    user = (
        f"Topic: {topic}\nDifficulty: {difficulty}\nTotal marks: {int(total_marks)}\n"
        f"Exact question counts: {json.dumps(count_instructions)}\n\nEvidence:\n{context}"
    )
    raw = _ask(api_key, model_name, system, user, 0.1)
    last_error = None
    for attempt in range(2):
        try:
            result = validate_exam(_json(raw), int(total_marks))
            actual_counts = {}
            for question in result["questions"]:
                actual_counts[question["type"]] = actual_counts.get(question["type"], 0) + 1
            if actual_counts != count_instructions:
                raise ValueError(
                    f"Generated counts {actual_counts} do not match {count_instructions}"
                )
            return result
        except Exception as error:
            last_error = error
            if attempt == 1:
                break
            repair_system = """Repair an exam JSON object without changing its topic or evidence boundary. Return JSON only. Make every required field valid, produce exactly the requested question counts, and make question marks sum exactly to the requested total. Do not add unsupported facts or evidence identifiers."""
            repair_user = (
                f"Validation error: {error}\nRequested total: {int(total_marks)}\n"
                f"Requested counts: {json.dumps(count_instructions)}\n"
                f"Invalid exam JSON:\n{raw}\n\nEvidence:\n{context}"
            )
            raw = _ask(api_key, model_name, repair_system, repair_user, 0)
    raise ValueError(f"The exam remained invalid after one repair attempt: {last_error}")


def deterministic_objective_grade(question, learner_answer):
    """Grade MCQ and True/False without an LLM when exact grading is possible."""
    answer = str(learner_answer or "").strip()
    reference = str(question["answer"]).strip()
    maximum = int(question["marks"])
    correct = answer.casefold() == reference.casefold()
    return {
        "question_id": question["id"],
        "awarded_marks": maximum if correct else 0,
        "max_marks": maximum,
        "percentage": 100.0 if correct else 0.0,
        "verdict": "Correct" if correct else "Incorrect",
        "matched_points": question["rubric"] if correct else [],
        "missing_points": [] if correct else question["rubric"],
        "feedback": (
            "Correct answer." if correct
            else "The selected answer does not match the evidence-grounded answer."
        ),
        "reference_answer": reference,
        "evidence_ids": question.get("evidence_ids", []),
        "topic_tag": question.get("topic_tag", "General"),
    }


def grade_written_answer(question, learner_answer, mode, context, api_key, model_name):
    if mode not in GRADING_MODES:
        raise ValueError("Unknown grading mode")
    maximum = int(question["marks"])
    if not str(learner_answer or "").strip():
        return {
            "question_id": question["id"],
            "awarded_marks": 0,
            "max_marks": maximum,
            "percentage": 0.0,
            "verdict": "Not answered",
            "matched_points": [],
            "missing_points": question["rubric"],
            "feedback": "No answer was submitted.",
            "reference_answer": question["answer"],
            "evidence_ids": question.get("evidence_ids", []),
            "topic_tag": question.get("topic_tag", "General"),
        }
    system = """You are an evidence-bounded examiner. Grade only against the supplied question, reference answer, rubric and evidence. Return JSON only: {"awarded_marks":0,"verdict":"...","matched_points":[],"missing_points":[],"feedback":"...","evidence_ids":[]}. Award partial credit proportionally to satisfied rubric points. Never reward unsupported claims. Never exceed maximum marks. Do not penalize language style unless the rubric requires terminology."""
    user = (
        f"Grading mode: {mode}. {GRADING_MODES[mode]}\n"
        f"Maximum marks: {maximum}\n"
        f"Question package: {json.dumps(question, ensure_ascii=False)}\n"
        f"Learner answer: {learner_answer}\n\nEvidence:\n{context}"
    )
    result = _json(_ask(api_key, model_name, system, user, 0))
    awarded = max(0.0, min(float(result.get("awarded_marks", 0)), float(maximum)))
    return {
        "question_id": question["id"],
        "awarded_marks": round(awarded, 2),
        "max_marks": maximum,
        "percentage": round(100 * awarded / maximum, 1) if maximum else 0.0,
        "verdict": str(result.get("verdict", "Graded")),
        "matched_points": list(result.get("matched_points", [])),
        "missing_points": list(result.get("missing_points", [])),
        "feedback": str(result.get("feedback", "")),
        "reference_answer": question["answer"],
        "evidence_ids": _verified_evidence_ids(
            result.get("evidence_ids", []), question.get("evidence_ids", [])
        ),
        "topic_tag": question.get("topic_tag", "General"),
    }


def grade_answer(question, learner_answer, mode, context, api_key, model_name):
    """Backward-compatible single-question grading entry point."""
    if question["type"] in {"MCQ", "True/False"}:
        return deterministic_objective_grade(question, learner_answer)
    return grade_written_answer(
        question, learner_answer, mode, context, api_key, model_name
    )


def grade_written_batch(packages, mode, context, api_key, model_name):
    """Grade all answered essay-style questions in one controlled API request."""
    if not packages:
        return []
    if mode not in GRADING_MODES:
        raise ValueError("Unknown grading mode")
    system = """You are an evidence-bounded examiner grading several written answers. Use only each question's reference answer, rubric and supplied evidence. Return JSON only: {"results":[{"question_id":"Q1","awarded_marks":0,"verdict":"...","matched_points":[],"missing_points":[],"feedback":"...","evidence_ids":[]}]}. Return exactly one result for every supplied question_id. Award partial credit proportionally. Never exceed that question's maximum marks. Never reward unsupported claims. Do not penalize writing style unless terminology is explicitly required by the selected grading mode or rubric."""
    user = (
        f"Grading mode: {mode}. {GRADING_MODES[mode]}\n"
        f"Written answer packages: {json.dumps(packages, ensure_ascii=False)}\n\n"
        f"Evidence:\n{context}"
    )
    raw = _json(_ask(api_key, model_name, system, user, 0))
    returned = raw.get("results", [])
    if not isinstance(returned, list):
        raise ValueError("Batch grading did not return a results list")
    by_id = {str(item.get("question_id")): item for item in returned}
    if set(by_id) != {str(item["question"]["id"]) for item in packages}:
        raise ValueError("Batch grading returned missing or unexpected question IDs")
    normalized = []
    for package in packages:
        question = package["question"]
        result = by_id[str(question["id"])]
        maximum = int(question["marks"])
        awarded = max(0.0, min(float(result.get("awarded_marks", 0)), float(maximum)))
        normalized.append({
            "question_id": question["id"],
            "awarded_marks": round(awarded, 2),
            "max_marks": maximum,
            "percentage": round(100 * awarded / maximum, 1) if maximum else 0.0,
            "verdict": str(result.get("verdict", "Graded")),
            "matched_points": list(result.get("matched_points", [])),
            "missing_points": list(result.get("missing_points", [])),
            "feedback": str(result.get("feedback", "")),
            "reference_answer": question["answer"],
            "evidence_ids": _verified_evidence_ids(
                result.get("evidence_ids", []), question.get("evidence_ids", [])
            ),
            "topic_tag": question.get("topic_tag", "General"),
        })
    return normalized


def grade_full_exam(exam, answers, mode, context, api_key, model_name):
    """Grade a complete locked attempt and return verified totals and weak topics."""
    validate_exam(exam)
    if mode not in GRADING_MODES:
        raise ValueError("Unknown grading mode")
    results_by_id = {}
    written_packages = []
    for question in exam["questions"]:
        learner_answer = answers.get(question["id"], "")
        if question["type"] in {"MCQ", "True/False"}:
            results_by_id[question["id"]] = deterministic_objective_grade(
                question, learner_answer
            )
        elif not str(learner_answer or "").strip():
            results_by_id[question["id"]] = grade_written_answer(
                question, "", mode, context, api_key, model_name
            )
        else:
            written_packages.append({
                "question": question,
                "learner_answer": learner_answer,
            })
    for item in grade_written_batch(
        written_packages, mode, context, api_key, model_name
    ):
        results_by_id[item["question_id"]] = item
    results = [results_by_id[question["id"]] for question in exam["questions"]]
    awarded = round(sum(float(item["awarded_marks"]) for item in results), 2)
    maximum = int(exam["total_marks"])
    percentage = round(100 * awarded / maximum, 1) if maximum else 0.0
    weak_topics = sorted({
        item["topic_tag"] for item in results if float(item["percentage"]) < 60
    })
    return {
        "exam_title": exam.get("title", "AS Assessment"),
        "awarded_marks": awarded,
        "max_marks": maximum,
        "percentage": percentage,
        "status": "Passed" if percentage >= 60 else "Needs Review",
        "grading_mode": mode,
        "question_results": results,
        "weak_topics": weak_topics,
    }


def result_to_json_bytes(result):
    return json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
