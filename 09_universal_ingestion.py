"""Session-scoped multimodal ingestion for AS Intelligence Studio."""

from io import BytesIO
import base64
import hashlib
import re

import fitz
import numpy as np
import pandas as pd
from docx import Document
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED_EXTENSIONS = {"pdf", "docx", "xlsx", "xls", "csv", "png", "jpg", "jpeg", "webp"}
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILES = 12
MAX_VISUALS_PER_FILE = 12


def _clean(text):
    text = str(text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunks(text, source_name, location, kind="text", target_words=210, overlap=35):
    words = _clean(text).split()
    if not words:
        return []
    rows, start, part = [], 0, 1
    while start < len(words):
        end = min(start + target_words, len(words))
        payload = " ".join(words[start:end])
        identity = f"{source_name}|{location}|{part}|{payload}".encode("utf-8")
        rows.append({
            "chunk_id": hashlib.sha256(identity).hexdigest()[:24],
            "source_name": source_name,
            "location": location,
            "content_kind": kind,
            "text": payload,
            "word_count": len(payload.split()),
        })
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
        part += 1
    return rows


def _image_record(image_bytes, source_name, location, mime_type):
    digest = hashlib.sha256(image_bytes).hexdigest()[:24]
    return {
        "visual_id": digest,
        "source_name": source_name,
        "location": location,
        "mime_type": mime_type,
        "bytes": image_bytes,
    }


def _extract_pdf(data, name):
    rows, visuals = [], []
    document = fitz.open(stream=data, filetype="pdf")
    for page_index, page in enumerate(document, start=1):
        rows.extend(_chunks(page.get_text("text"), name, f"Page {page_index}"))
        for image_index, image in enumerate(page.get_images(full=True)[:MAX_VISUALS_PER_FILE], start=1):
            try:
                payload = document.extract_image(image[0])
                visuals.append(_image_record(
                    payload["image"], name, f"Page {page_index}, image {image_index}",
                    f"image/{payload.get('ext', 'png')}"
                ))
            except Exception:
                continue
    return rows, visuals


def _extract_docx(data, name):
    rows, visuals = [], []
    document = Document(BytesIO(data))
    for index, paragraph in enumerate(document.paragraphs, start=1):
        rows.extend(_chunks(paragraph.text, name, f"Paragraph {index}"))
    for table_index, table in enumerate(document.tables, start=1):
        table_text = "\n".join(" | ".join(cell.text for cell in row.cells) for row in table.rows)
        rows.extend(_chunks(table_text, name, f"Table {table_index}", "table"))
    for index, relation in enumerate(document.part.rels.values(), start=1):
        if "image" in relation.reltype and len(visuals) < MAX_VISUALS_PER_FILE:
            blob = relation.target_part.blob
            visuals.append(_image_record(blob, name, f"Embedded image {index}", "image/png"))
    return rows, visuals


def _extract_spreadsheet(data, name, extension):
    rows = []
    if extension == "csv":
        sheets = {"CSV": pd.read_csv(BytesIO(data))}
    else:
        sheets = pd.read_excel(BytesIO(data), sheet_name=None)
    for sheet_name, frame in sheets.items():
        frame = frame.dropna(how="all").dropna(axis=1, how="all")
        if frame.empty:
            continue
        text = frame.astype(str).to_csv(index=False)
        rows.extend(_chunks(text, name, f"Sheet: {sheet_name}", "table"))
        summary = (
            f"Sheet {sheet_name}. Rows: {len(frame)}. Columns: {len(frame.columns)}. "
            f"Column names: {', '.join(map(str, frame.columns))}."
        )
        rows.extend(_chunks(summary, name, f"Sheet: {sheet_name} summary", "table_summary"))
    return rows, []


def ingest_uploaded_files(uploaded_files):
    if len(uploaded_files) > MAX_FILES:
        raise ValueError(f"Upload at most {MAX_FILES} files per workspace.")
    rows, visuals, report = [], [], []
    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        name = uploaded.name
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if extension not in SUPPORTED_EXTENSIONS:
            report.append({"file": name, "status": "Rejected", "detail": "Unsupported type"})
            continue
        if len(data) > MAX_FILE_BYTES:
            report.append({"file": name, "status": "Rejected", "detail": "File exceeds 25 MB"})
            continue
        try:
            if extension == "pdf":
                file_rows, file_visuals = _extract_pdf(data, name)
            elif extension == "docx":
                file_rows, file_visuals = _extract_docx(data, name)
            elif extension in {"xlsx", "xls", "csv"}:
                file_rows, file_visuals = _extract_spreadsheet(data, name, extension)
            else:
                Image.open(BytesIO(data)).verify()
                file_rows = []
                file_visuals = [_image_record(data, name, "Standalone image", uploaded.type or f"image/{extension}")]
            rows.extend(file_rows)
            visuals.extend(file_visuals)
            report.append({"file": name, "status": "Ready", "detail": f"{len(file_rows)} text chunks, {len(file_visuals)} visuals"})
        except Exception as error:
            report.append({"file": name, "status": "Failed", "detail": str(error)[:180]})
    return pd.DataFrame(rows), visuals, pd.DataFrame(report)


def embed_workspace(chunks_df, embedding_model):
    if chunks_df.empty:
        return np.empty((0, 384), dtype=np.float32)
    return embedding_model.encode(
        chunks_df["text"].tolist(), normalize_embeddings=True,
        show_progress_bar=False, batch_size=32,
    )


def retrieve_workspace(question, chunks_df, embeddings, embedding_model, top_k=7):
    if chunks_df.empty or not len(embeddings):
        return chunks_df.head(0).copy()
    query = embedding_model.encode([question], normalize_embeddings=True, show_progress_bar=False)
    scores = cosine_similarity(query, embeddings)[0]
    take = np.argsort(scores)[::-1][:min(top_k, len(scores))]
    result = chunks_df.iloc[take].copy()
    result["score"] = scores[take]
    return result.reset_index(drop=True)


def build_evidence_context(results):
    blocks = []
    for index, row in results.reset_index(drop=True).iterrows():
        blocks.append(
            f"[Evidence {index + 1}]\nFile: {row['source_name']}\n"
            f"Location: {row['location']}\nType: {row['content_kind']}\nText: {row['text']}"
        )
    return "\n\n".join(blocks)


def image_data_uri(visual):
    encoded = base64.b64encode(visual["bytes"]).decode("ascii")
    return f"data:{visual['mime_type']};base64,{encoded}"
