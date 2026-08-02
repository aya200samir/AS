"""Strict document discovery and page-level extraction for the approved corpus."""

from pathlib import Path

import fitz


DOCUMENT_CATALOG = {
    "practical-machine-learning-with-python.pdf": {
        "title": "Practical Machine Learning with Python",
        "authors": "Dipanjan Sarkar, Raghav Bali, Tushar Sharma",
        "edition": "First Edition",
        "publication_year": 2018,
        "subject": "Applied Machine Learning with Python",
    },
    "Machine_Learning_Yearning.pdf": {
        "title": "Machine Learning Yearning",
        "authors": "Andrew Ng",
        "edition": "Draft Edition",
        "publication_year": 2018,
        "subject": "Machine Learning Strategy",
    },
    "Applied_Artificial_Intelligence.pdf": {
        "title": "Applied Artificial Intelligence: A Handbook for Business Leaders",
        "authors": "Mariya Yao, Adelyn Zhou, Marlene Jia",
        "edition": "First Edition",
        "publication_year": 2018,
        "subject": "Applied Artificial Intelligence",
    },
    "Applied Deep Learning with TensorFlow 2.pdf": {
        "title": "Applied Deep Learning with TensorFlow 2",
        "authors": "Umberto Michelucci",
        "edition": "Second Edition",
        "publication_year": 2022,
        "subject": "Deep Learning with TensorFlow",
    },
}

APPROVED_FILENAMES = tuple(DOCUMENT_CATALOG)


def discover_approved_pdf_files(documents_dir):
    """Return only the four approved PDFs and reject an incomplete corpus."""
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    available = {path.name: path for path in documents_dir.glob("*.pdf")}
    missing = [name for name in APPROVED_FILENAMES if name not in available]
    if missing:
        missing_list = "\n- ".join(missing)
        raise FileNotFoundError(
            "The closed knowledge base is incomplete. Missing approved files:\n- "
            + missing_list
        )

    return [available[name] for name in APPROVED_FILENAMES]


def find_unapproved_pdf_files(documents_dir):
    """List PDFs that exist in the folder but are excluded from indexing."""
    approved = set(APPROVED_FILENAMES)
    return sorted(
        path for path in Path(documents_dir).glob("*.pdf") if path.name not in approved
    )


def inspect_pdf_files(documents_dir):
    """Inspect readability and basic extraction quality for every approved PDF."""
    rows = []
    for document_id, pdf_file in enumerate(discover_approved_pdf_files(documents_dir)):
        with fitz.open(pdf_file) as pdf:
            page_count = pdf.page_count
            sample_count = min(5, page_count)
            sample_text = " ".join(
                pdf.load_page(index).get_text("text") or ""
                for index in range(sample_count)
            )
        rows.append(
            {
                "document_id": document_id,
                "source": pdf_file.name,
                "title": DOCUMENT_CATALOG[pdf_file.name]["title"],
                "pages": page_count,
                "sample_characters_per_page": round(
                    len(sample_text) / max(sample_count, 1), 1
                ),
                "possible_scan": len(sample_text) / max(sample_count, 1) < 80,
            }
        )
    return rows


def load_pdf_pages(documents_dir):
    """Extract page records with complete citation metadata."""
    records = []
    for document_id, pdf_file in enumerate(discover_approved_pdf_files(documents_dir)):
        metadata = DOCUMENT_CATALOG[pdf_file.name]
        with fitz.open(pdf_file) as pdf:
            for page_index, page in enumerate(pdf):
                records.append(
                    {
                        "document_id": document_id,
                        "source": pdf_file.name,
                        "title": metadata["title"],
                        "authors": metadata["authors"],
                        "edition": metadata["edition"],
                        "publication_year": metadata["publication_year"],
                        "subject": metadata["subject"],
                        "page_number": page_index + 1,
                        "location_label": "Page",
                        "raw_text": page.get_text("text") or "",
                    }
                )
    return records
