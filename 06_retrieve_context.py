"""One production retrieval implementation shared by evaluation and Streamlit."""

import math
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


def lexical_tokenize(text):
    """Preserve English technical symbols and Arabic words for BM25."""
    return re.findall(
        r"[A-Za-z0-9_+.-]+|[\u0600-\u06FF]+",
        text.lower(),
    )


def min_max_normalize(values):
    """Normalize ranking signals only; never use this value as an answer threshold."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    minimum, maximum = values.min(), values.max()
    if math.isclose(minimum, maximum):
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def build_bm25(chunks_df):
    """Build the lexical index from the same chunks stored in Chroma."""
    tokenized = [
        lexical_tokenize(text) for text in chunks_df["embedding_text"].tolist()
    ]
    return BM25Okapi(tokenized)


def _title_filter(selected_titles):
    if not selected_titles:
        return None
    return {"title": {"$in": list(selected_titles)}}


def _eligible_indices(chunks_df, selected_titles):
    if not selected_titles:
        return np.arange(len(chunks_df))
    return np.flatnonzero(chunks_df["title"].isin(selected_titles).to_numpy())


def retrieve_hybrid(
    search_queries,
    model,
    collection,
    chunks_df,
    bm25_index,
    selected_titles=None,
    k=12,
    candidate_pool=40,
    alpha=0.65,
):
    """Retrieve the union of Chroma semantic and BM25 lexical candidates."""
    if isinstance(search_queries, str):
        search_queries = [search_queries]
    search_queries = [query.strip() for query in search_queries if query.strip()]
    if not search_queries:
        raise ValueError("At least one search query is required")

    eligible_indices = _eligible_indices(chunks_df, selected_titles)
    if len(eligible_indices) == 0:
        return pd.DataFrame()

    where = _title_filter(selected_titles)
    semantic_by_id = {}
    for query in search_queries:
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_arguments = dict(
            query_embeddings=query_embedding.tolist(),
            n_results=min(candidate_pool, len(eligible_indices)),
        )
        if where is not None:
            query_arguments["where"] = where
        semantic = collection.query(**query_arguments)
        for index, chunk_id in enumerate(semantic["ids"][0]):
            score = 1 - float(semantic["distances"][0][index])
            semantic_by_id[chunk_id] = max(
                score,
                semantic_by_id.get(chunk_id, float("-inf")),
            )

    lexical_scores = np.zeros(len(chunks_df), dtype=float)
    for query in search_queries:
        lexical_scores = np.maximum(
            lexical_scores,
            np.asarray(bm25_index.get_scores(lexical_tokenize(query))),
        )

    eligible_scores = lexical_scores[eligible_indices]
    lexical_order = eligible_indices[
        np.argsort(eligible_scores)[::-1][: min(candidate_pool, len(eligible_indices))]
    ]
    lexical_ids = chunks_df.iloc[lexical_order]["chunk_id"].tolist()

    candidate_ids = set(semantic_by_id) | set(lexical_ids)
    candidates = chunks_df[chunks_df["chunk_id"].isin(candidate_ids)].copy()
    candidates["semantic_score"] = candidates["chunk_id"].map(
        semantic_by_id
    ).fillna(0.0)
    candidates["bm25_score"] = lexical_scores[candidates.index]

    candidates["semantic_normalized"] = min_max_normalize(
        candidates["semantic_score"]
    )
    candidates["bm25_normalized"] = min_max_normalize(candidates["bm25_score"])
    candidates["hybrid_score"] = (
        alpha * candidates["semantic_normalized"]
        + (1 - alpha) * candidates["bm25_normalized"]
    )
    return candidates.sort_values(
        ["hybrid_score", "semantic_score"],
        ascending=False,
    ).head(k).reset_index(drop=True)


def jaccard_similarity(text_a, text_b):
    set_a = set(lexical_tokenize(text_a))
    set_b = set(lexical_tokenize(text_b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def build_context_package(
    results,
    max_words=1200,
    max_chunks=7,
    duplicate_threshold=0.82,
):
    """Deduplicate candidate passages and package traceable evidence."""
    selected, used_words = [], 0
    for row in results.itertuples():
        if any(
            jaccard_similarity(row.chunk_text, item["chunk_text"])
            >= duplicate_threshold
            for item in selected
        ):
            continue

        chunk_words = len(row.chunk_text.split())
        if selected and used_words + chunk_words > max_words:
            continue

        selected.append(row._asdict())
        used_words += chunk_words
        if len(selected) >= max_chunks:
            break

    blocks = []
    for index, item in enumerate(selected, start=1):
        blocks.append(
            f'<source id="Source {index}">\n'
            f'title: {item["title"]}\n'
            f'authors: {item["authors"]}\n'
            f'edition: {item["edition"]}\n'
            f'page: {item["page_number"]}\n'
            f'content:\n{item["chunk_text"]}\n'
            f'</source>'
        )

    return {
        "context_text": "\n\n".join(blocks),
        "selected_evidence": pd.DataFrame(selected),
        "used_words": used_words,
    }
