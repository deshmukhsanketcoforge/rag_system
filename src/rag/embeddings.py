"""Embedding model wrapper. The model is loaded once per process and batches are encoded."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from rag.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches; returns one 384-dim vector per text."""
    vectors = get_model().encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False)
    return vectors.tolist()
