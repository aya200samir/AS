"""Paragraph- and sentence-aware chunking for technical books."""

import re

import pandas as pd


TARGET_WORDS = 220
OVERLAP_WORDS = 45
MINIMUM_CHUNK_WORDS = 20

LOW_INFORMATION_TEXTS = {
    "thank you!",
    "thank you",
    "references",
    "index",
}


def split_into_units(text):
    """Split text into paragraphs and sentences while preserving readable units."""
    paragraphs = [part.strip() for part in re.split(r"\n\n+", text) if part.strip()]
    units = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        units.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return units


def split_oversized_unit(unit, target_words, overlap_words):
    """Prevent a single long sentence or extracted code block from breaking the limit."""
    words = unit.split()
    if len(words) <= target_words:
        return [unit]

    parts = []
    step = target_words - overlap_words
    for start in range(0, len(words), step):
        part = words[start : start + target_words]
        if part:
            parts.append(" ".join(part))
        if start + target_words >= len(words):
            break
    return parts


def structure_aware_chunk_text(
    text,
    target_words=TARGET_WORDS,
    overlap_words=OVERLAP_WORDS,
):
    """Create bounded chunks with overlap and a fallback for oversized units."""
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be between 0 and target_words - 1")

    expanded_units = []
    for unit in split_into_units(text):
        expanded_units.extend(
            split_oversized_unit(unit, target_words, overlap_words)
        )

    chunks, current = [], []
    for unit in expanded_units:
        unit_words = unit.split()
        if current and len(current) + len(unit_words) > target_words:
            chunks.append(" ".join(current).strip())
            current = current[-overlap_words:] if overlap_words else []
        current.extend(unit_words)

    if current:
        chunks.append(" ".join(current).strip())

    # A long extracted sentence can combine with the overlap and exceed the target.
    # Apply one final bounded window so the size contract is always true.
    bounded_chunks = []
    step = target_words - overlap_words
    for chunk in chunks:
        words = chunk.split()
        if len(words) <= target_words:
            bounded_chunks.append(chunk)
            continue
        for start in range(0, len(words), step):
            bounded_chunks.append(" ".join(words[start : start + target_words]))
            if start + target_words >= len(words):
                break
    return bounded_chunks


def is_useful_chunk(chunk_text, minimum_words=MINIMUM_CHUNK_WORDS):
    """Remove empty, boilerplate, and extremely short chunks."""
    normalized = re.sub(r"\s+", " ", chunk_text).strip().lower()
    return len(normalized.split()) >= minimum_words and normalized not in LOW_INFORMATION_TEXTS


def build_chunks(documents_df):
    """Create the searchable chunk table from the closed book corpus."""
    rows = []
    for page in documents_df.itertuples():
        if page.is_empty:
            continue

        page_chunks = structure_aware_chunk_text(page.clean_text)
        for chunk_index, chunk_text in enumerate(page_chunks):
            if not is_useful_chunk(chunk_text):
                continue

            rows.append(
                {
                    "chunk_id": (
                        f"doc{page.document_id}_p{page.page_number}_c{chunk_index}"
                    ),
                    "document_id": int(page.document_id),
                    "source": page.source,
                    "title": page.title,
                    "authors": page.authors,
                    "edition": page.edition,
                    "publication_year": int(page.publication_year),
                    "subject": page.subject,
                    "page_number": int(page.page_number),
                    "location_label": page.location_label,
                    "chunk_index": int(chunk_index),
                    "chunk_text": chunk_text,
                    "embedding_text": (
                        f"{page.title} | {page.subject} | {chunk_text}"
                    ),
                    "word_count": len(chunk_text.split()),
                }
            )
    return pd.DataFrame(rows)
