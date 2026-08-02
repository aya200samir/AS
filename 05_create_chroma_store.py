"""Versioned persistent Chroma storage that rebuilds when corpus content changes."""

import hashlib

import chromadb


COLLECTION_NAME = "ai_knowledge_tutor_closed_corpus"


def compute_index_fingerprint(chunks_df, embedding_model_name):
    """Hash the exact chunks and embedding model used to build the index."""
    digest = hashlib.sha256(embedding_model_name.encode("utf-8"))
    for row in chunks_df.itertuples():
        digest.update(str(row.chunk_id).encode("utf-8"))
        digest.update(row.embedding_text.encode("utf-8"))
    return digest.hexdigest()


def _metadata_rows(chunks_df):
    return [
        {
            "document_id": int(row.document_id),
            "source": str(row.source),
            "title": str(row.title),
            "authors": str(row.authors),
            "edition": str(row.edition),
            "publication_year": int(row.publication_year),
            "subject": str(row.subject),
            "page_number": int(row.page_number),
            "location_label": str(row.location_label),
            "chunk_index": int(row.chunk_index),
            "word_count": int(row.word_count),
        }
        for row in chunks_df.itertuples()
    ]


def get_or_create_chroma_collection(
    chroma_path,
    chunks_df,
    embeddings,
    embedding_model_name,
):
    """Reuse a matching index or atomically rebuild a stale collection."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    fingerprint = compute_index_fingerprint(chunks_df, embedding_model_name)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        stored_fingerprint = (collection.metadata or {}).get("fingerprint")
        if stored_fingerprint != fingerprint or collection.count() != len(chunks_df):
            client.delete_collection(COLLECTION_NAME)
            collection = None
    except Exception:
        collection = None

    if collection is None:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "fingerprint": fingerprint},
        )
        metadata = _metadata_rows(chunks_df)
        batch_size = 500
        for start in range(0, len(chunks_df), batch_size):
            end = start + batch_size
            collection.add(
                ids=chunks_df["chunk_id"].iloc[start:end].tolist(),
                documents=chunks_df["chunk_text"].iloc[start:end].tolist(),
                metadatas=metadata[start:end],
                embeddings=embeddings[start:end].tolist(),
            )

    return collection, fingerprint
