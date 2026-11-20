"""Question in, grounded answer out (RAG-5), with switchable retrieval.

`mode` picks the retrieval: "vector" (RAG-5), "hybrid" = vector + keyword merged with RRF (RAG-6),
"rerank" = hybrid candidates re-scored by a cross-encoder (RAG-7). Keeping every mode means the
first pipeline can still be reproduced, and the eval can compare them.
"""

from collections.abc import Callable

from chromadb.errors import NotFoundError
from dotenv import load_dotenv

from rag import generate as gen
from rag import rerank as rr
from rag import retrieval, store
from rag.config import RERANK_CANDIDATES, ROOT, TOP_K
from rag.llm import require_settings

MODES = ("vector", "hybrid", "rerank")

load_dotenv(ROOT / ".env")  # once, at import; real environment variables still win


def retrieve(
    question: str,
    collection,
    embedder: Callable,
    k: int = TOP_K,
    mode: str = "vector",
    reranker: Callable | None = None,
) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}', expected one of {MODES}")
    if mode == "rerank":
        candidates = retrieval.hybrid_search(
            question, collection, embedder, k=max(RERANK_CANDIDATES, k)
        )
        return rr.rerank(question, candidates, k=k, scorer=reranker)
    if mode == "hybrid":
        return retrieval.hybrid_search(question, collection, embedder, k=k)
    return store.query(collection, embedder([question])[0], k=k)


def sources_of(chunks: list[dict]) -> list[dict]:
    """Distinct (document, section) pairs in rank order."""
    seen, out = set(), []
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["doc_name"], meta["section"])
        if key not in seen:
            seen.add(key)
            out.append({"doc_name": key[0], "section": key[1]})
    return out


def _require_ingested(collection=None):
    """The collection to search; a missing or empty store is a setup error, not an empty answer."""
    if collection is None:
        try:
            collection = store.get_collection()
        except NotFoundError:
            collection = None
    if collection is None or collection.count() == 0:
        raise RuntimeError("the vector store is empty: run scripts/02_ingest.py first")
    return collection


def resolve_defaults(
    collection=None,
    embedder: Callable | None = None,
    reranker: Callable | None = None,
    rerank: bool = False,
) -> tuple:
    """The (collection, embedder, reranker) to search with; anything left as None gets the real one.

    The store is checked first and the cross-encoder loads before the embedding model, so a setup
    problem fails fast, before any slow model loads. The reranker is only loaded when `rerank`.
    """
    collection = _require_ingested(collection)
    if rerank and reranker is None:
        reranker = rr.load_scorer()
    if embedder is None:
        from rag.embeddings import embed as embedder
    return collection, embedder, reranker


def answer_from_chunks(question: str, chunks: list[dict], llm=None) -> dict:
    """Generate from chunks that were already retrieved: {answer, chunks, sources}.

    A fallback answer ("not enough information") has no sources, since nothing supports it.
    """
    text = gen.generate(question, chunks, client=llm)
    return {
        "answer": text,
        "chunks": chunks,
        "sources": [] if text == gen.NOT_FOUND_ANSWER else sources_of(chunks),
    }


def answer(
    question: str,
    k: int = TOP_K,
    mode: str = "vector",
    collection=None,
    embedder: Callable | None = None,
    llm=None,
    reranker: Callable | None = None,
) -> dict:
    """Retrieve, generate, and return {answer, chunks, sources}.

    collection, embedder and llm default to the real ones; tests pass fakes. Everything that can
    fail cheaply (question, k, mode, API key, empty store) is checked before the model loads.
    """
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}', expected one of {MODES}")
    if reranker is not None and mode != "rerank":
        raise ValueError(f"a reranker only applies to mode 'rerank', not '{mode}'")
    if llm is None:
        require_settings()
    collection, embedder, reranker = resolve_defaults(
        collection, embedder, reranker, rerank=mode == "rerank"
    )
    chunks = retrieve(question, collection, embedder, k=k, mode=mode, reranker=reranker)
    return answer_from_chunks(question, chunks, llm=llm)
