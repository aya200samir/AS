"""Multilingual dense vector representation for English books and Arabic queries."""

from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Load the same normalized embedding model for chunks and queries."""
    return SentenceTransformer(model_name)


def generate_embeddings(chunks_df, model):
    """Encode every searchable chunk into one normalized dense vector."""
    return model.encode(
        chunks_df["embedding_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
