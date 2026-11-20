"""Question in, grounded answer out (RAG-5). Retrieval is vector-only for now.

`mode` is the switch later tickets extend: "vector" today, then hybrid and hybrid + rerank, so
the first pipeline can still be reproduced at the end.
"""

import os
from collections.abc import Callable

from chromadb.errors import NotFoundError
from dotenv import load_dotenv

from rag import generate as gen
from rag import store
from rag.config import ROOT, TOP_K

MODES = ("vector",)

load_dotenv(ROOT / ".env")  # once, at import; real environment variables still win


def retrieve(
    question: str, collection, embedder: Callable, k: int = TOP_K, mode: str = "vector"
) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}', expected one of {MODES}")
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


def answer(
    question: str,
    k: int = TOP_K,
    mode: str = "vector",
    collection=None,
    embedder: Callable | None = None,
    llm=None,
) -> dict:
    """Retrieve, generate, and return {answer, chunks, sources}.

    collection, embedder and llm default to the real ones; tests pass fakes. Everything that can
    fail cheaply (question, k, mode, API key, empty store) is checked before the model loads.
    A fallback answer ("not enough information") has no sources, since nothing supports it.
    """
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}', expected one of {MODES}")
    if llm is None and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set: add it to .env or the environment")
    collection = _require_ingested(collection)
    if embedder is None:
        from rag.embeddings import embed as embedder
    chunks = retrieve(question, collection, embedder, k=k, mode=mode)
    text = gen.generate(question, chunks, client=llm)
    return {
        "answer": text,
        "chunks": chunks,
        "sources": [] if text == gen.NOT_FOUND_ANSWER else sources_of(chunks),
    }
