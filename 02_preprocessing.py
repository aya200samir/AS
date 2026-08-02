"""Meaning-preserving preprocessing and extraction quality flags."""

import re

import pandas as pd


NOISE_PATTERNS = (
    r"https?://\S+",
    r"\bPage\s+\d+\s+of\s+\d+\b",
    r"\bwww\.allitebooks\.com\b",
)


def clean_educational_text(text):
    """Remove common PDF noise without removing technical content."""
    text = text.replace("\xa0", " ").replace("\u200b", "")
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Keep paragraph boundaries because the chunker uses them as soft structure.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def preprocess_documents(document_pages):
    """Return a DataFrame containing both raw and cleaned page content."""
    rows = []
    for record in document_pages:
        cleaned = clean_educational_text(record["raw_text"])
        word_count = len(cleaned.split())
        rows.append(
            {
                **record,
                "clean_text": cleaned,
                "raw_character_count": len(record["raw_text"]),
                "clean_word_count": word_count,
                "is_empty": word_count == 0,
                "is_very_short": 0 < word_count < 20,
            }
        )
    return pd.DataFrame(rows)
