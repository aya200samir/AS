"""Deterministic study-plan state, progress, export, and RAG request builders."""

from datetime import date, timedelta
from io import BytesIO, StringIO
import csv
import json
import re

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


FOUNDATION_PLAN = [
    ("AI Foundations", "What AI means, intelligent agents, and common AI tasks"),
    ("Machine Learning Foundations", "Data, features, labels, models, training, and testing"),
    ("Learning Paradigms", "Supervised, unsupervised, and reinforcement learning"),
    ("Model Evaluation", "Generalization, overfitting, validation, and evaluation metrics"),
    ("Core ML Models", "Linear models, decision trees, nearest neighbors, and ensembles"),
    ("Neural Network Foundations", "Neurons, layers, activation functions, loss, and gradient descent"),
    ("Deep Learning", "Backpropagation, representation learning, and deep architectures"),
    ("Applied Learning Project", "Problem framing, error analysis, iteration, and responsible deployment"),
]


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "topic"


def _clean_line(line):
    line = re.sub(r"^\s*(?:[-*•]+|\d+[.)]|week\s+\d+\s*[:.-]?)\s*", "", line, flags=re.I)
    return re.sub(r"\s+", " ", line).strip()


def build_plan(topics, title="My AI Learning Plan", start_date=None, days_per_unit=7):
    """Create serializable units while keeping user topic order unchanged."""
    if days_per_unit < 1:
        raise ValueError("days_per_unit must be at least one")
    start = start_date or date.today()
    units = []
    seen = set()
    for index, item in enumerate(topics, start=1):
        if isinstance(item, str):
            topic, objective = item, f"Understand and explain {item}"
        else:
            topic, objective = item
        topic = re.sub(r"\s+", " ", str(topic)).strip()
        objective = re.sub(r"\s+", " ", str(objective)).strip()
        if not topic:
            continue
        base_id = _slug(topic)
        unit_id = base_id
        suffix = 2
        while unit_id in seen:
            unit_id = f"{base_id}-{suffix}"
            suffix += 1
        seen.add(unit_id)
        due = start + timedelta(days=index * days_per_unit - 1)
        units.append({
            "unit_id": unit_id,
            "sequence": index,
            "topic": topic,
            "objective": objective,
            "status": "Not Started",
            "completed_at": "",
            "due_date": due.isoformat(),
        })
    if not units:
        raise ValueError("The study plan does not contain readable topics")
    if len(units) > 100:
        raise ValueError("A study plan can contain at most 100 units")
    return {
        "schema_version": 1,
        "title": title.strip() or "My AI Learning Plan",
        "created_at": start.isoformat(),
        "days_per_unit": days_per_unit,
        "units": units,
    }


def parse_pasted_plan(text, title="My AI Learning Plan", start_date=None, days_per_unit=7):
    """Parse a simple weekly, numbered, or bulleted plan without treating it as evidence."""
    lines = [_clean_line(line) for line in text.splitlines()]
    topics = [line for line in lines if line and len(line) >= 2]
    return build_plan(topics, title, start_date, days_per_unit)


def build_foundation_plan(start_date=None, days_per_unit=7):
    return build_plan(
        FOUNDATION_PLAN,
        "AI Foundations: From Zero to Applied Learning",
        start_date,
        days_per_unit,
    )


def mark_completed(plan, unit_id, completed_on=None):
    """Return a copied plan with one unit marked as completed."""
    updated = json.loads(json.dumps(plan))
    matched = False
    for unit in updated["units"]:
        if unit["unit_id"] == unit_id:
            unit["status"] = "Completed"
            unit["completed_at"] = (completed_on or date.today()).isoformat()
            matched = True
            break
    if not matched:
        raise KeyError(f"Unknown unit_id: {unit_id}")
    return updated


def progress_summary(plan):
    total = len(plan.get("units", []))
    completed = sum(unit.get("status") == "Completed" for unit in plan.get("units", []))
    percentage = round(100 * completed / total) if total else 0
    next_unit = next(
        (unit for unit in plan.get("units", []) if unit.get("status") != "Completed"),
        None,
    )
    return {"total": total, "completed": completed, "percentage": percentage, "next_unit": next_unit}


def reschedule_remaining(plan, delay_days):
    """Shift only unfinished due dates, preserving topic order and completed history."""
    if delay_days < 0:
        raise ValueError("delay_days cannot be negative")
    updated = json.loads(json.dumps(plan))
    for unit in updated["units"]:
        if unit.get("status") != "Completed":
            due = date.fromisoformat(unit["due_date"])
            unit["due_date"] = (due + timedelta(days=delay_days)).isoformat()
    return updated


def diagnostic_recommendation(correct_answers, total_questions=5):
    if not 0 <= correct_answers <= total_questions:
        raise ValueError("Diagnostic score is outside the valid range")
    ratio = correct_answers / total_questions
    if ratio <= 0.4:
        return "Complete Beginner", 0
    if ratio <= 0.8:
        return "Basic Knowledge", 1
    return "Intermediate", 2


def validate_plan(plan):
    """Validate imported progress before it reaches widgets or HTML rendering."""
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("Unsupported study-plan schema")
    units = plan.get("units")
    if not isinstance(units, list) or not 1 <= len(units) <= 100:
        raise ValueError("The saved plan must contain between 1 and 100 units")
    required = {"unit_id", "sequence", "topic", "objective", "status", "due_date", "completed_at"}
    unit_ids = set()
    for unit in units:
        if not isinstance(unit, dict) or not required.issubset(unit):
            raise ValueError("A saved unit is missing required fields")
        if unit["status"] not in {"Not Started", "Completed"}:
            raise ValueError("A saved unit has an unsupported status")
        date.fromisoformat(unit["due_date"])
        if unit["completed_at"]:
            date.fromisoformat(unit["completed_at"])
        if not str(unit["topic"]).strip() or len(str(unit["topic"])) > 300:
            raise ValueError("A saved topic is empty or too long")
        if unit["unit_id"] in unit_ids:
            raise ValueError("A saved plan contains duplicate unit identifiers")
        unit_ids.add(unit["unit_id"])
    return plan


def lesson_request(unit):
    return (
        f"Teach the study-plan topic '{unit['topic']}' from the beginning. "
        f"Learning objective: {unit['objective']}. Define prerequisites, explain step by step, "
        "include a simple example, common mistakes, a short recap, and cite every factual claim."
    )


def reading_request(unit, learner_question):
    return (
        f"The current study-plan topic is '{unit['topic']}'. "
        f"Answer this question about the assigned book evidence: {learner_question}"
    )


def quiz_request(unit, question_count=5):
    return (
        f"Create a {question_count}-question formative quiz about '{unit['topic']}' using only "
        "the retrieved approved-book evidence. Mix conceptual and applied questions. Put the "
        "answer key after the questions, explain each answer briefly, and cite the supporting sources."
    )


def plan_to_csv_bytes(plan):
    text_buffer = StringIO(newline="")
    writer = csv.DictWriter(text_buffer, fieldnames=[
        "sequence", "topic", "objective", "status", "due_date", "completed_at"
    ])
    writer.writeheader()
    writer.writerows({key: unit.get(key, "") for key in writer.fieldnames} for unit in plan["units"])
    return text_buffer.getvalue().encode("utf-8-sig")


def plan_to_json_bytes(plan):
    return json.dumps(plan, ensure_ascii=False, indent=2).encode("utf-8")


def plan_to_pdf_bytes(plan):
    """Export the English plan as a clean A4 progress document."""
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    story = [Paragraph(plan["title"], styles["Title"]), Spacer(1, 12)]
    summary = progress_summary(plan)
    story.append(Paragraph(
        f"Progress: {summary['completed']} of {summary['total']} units ({summary['percentage']}%)",
        styles["Heading2"],
    ))
    for unit in plan["units"]:
        story.extend([
            Spacer(1, 10),
            Paragraph(f"Unit {unit['sequence']}: {unit['topic']}", styles["Heading3"]),
            Paragraph(f"Objective: {unit['objective']}", styles["BodyText"]),
            Paragraph(f"Status: {unit['status']} | Due: {unit['due_date']}", styles["BodyText"]),
        ])
    SimpleDocTemplate(buffer, pagesize=A4, title=plan["title"]).build(story)
    return buffer.getvalue()
