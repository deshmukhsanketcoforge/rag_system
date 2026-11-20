"""Cross-encoder reranking (RAG-7).

Vector and keyword search score the question and each chunk separately. A cross-encoder reads the
question and one chunk *together* and outputs a single relevance score, which is more accurate but
too slow to run over everything. So the pipeline does what is cheap first (hybrid search, a short
candidate list) and lets the cross-encoder choose the final top-n from those candidates.
"""

from collections.abc import Callable
from functools import lru_cache

from rag.config import RERANKER_MODEL

Scorer = Callable[[list[tuple[str, str]]], list[float]]


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def load_scorer() -> Scorer:
    """Load the cross-encoder now and return the scoring function.

    Called before retrieval so a missing or undownloadable model fails fast with a clear error,
    instead of after the search has run and with a raw traceback.
    """
    try:
        _model()
    except OSError as exc:
        raise RuntimeError(
            f"could not load the cross-encoder '{RERANKER_MODEL}' (not cached, no network?): {exc}"
        ) from exc
    return score_pairs


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Relevance score for each (question, chunk text) pair; higher is more relevant."""
    return [float(s) for s in _model().predict(pairs, show_progress_bar=False)]


def rerank(query: str, chunks: list[dict], k: int, scorer: Scorer | None = None) -> list[dict]:
    """The `k` chunks the cross-encoder rates highest, best first.

    Returns copies with `rerank_score` added; every other field (id, text, metadata, distance,
    the hybrid `score`) is kept. Ties keep the input order. `scorer` is for tests.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not chunks:
        return []
    scores = (scorer or score_pairs)([(query, c["text"]) for c in chunks])
    if len(scores) != len(chunks):
        raise ValueError("the scorer must return one score per chunk")
    scored = [{**c, "rerank_score": float(s)} for c, s in zip(chunks, scores, strict=True)]
    scored.sort(key=lambda c: -c["rerank_score"])  # stable, so ties keep their input order
    return scored[:k]
