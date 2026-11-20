"""Shared test setup: skip `llm` tests without Foundry settings, and the real-corpus fixtures.

Two things can make a test skip instead of run, and they are handled differently:
- no Foundry settings: `llm` tests skip (forks and fresh checkouts have no secrets);
- a model that cannot be downloaded: skips locally, but FAILS when RAG_REQUIRE_MODELS=1, which
  CI sets. Otherwise a failed download would leave CI green with the recall and eval tests
  silently not run.
"""

import os

import chromadb
import pytest
from dotenv import load_dotenv

from rag import llm, store
from rag import rerank as rerank_module
from rag.config import ROOT

load_dotenv(ROOT / ".env")


def models_required() -> bool:
    """True when a model that cannot be loaded must fail the run (RAG_REQUIRE_MODELS=1)."""
    return os.getenv("RAG_REQUIRE_MODELS") == "1"


def skip_or_fail(reason: str) -> None:
    if models_required():
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def pytest_collection_modifyitems(config, items):
    if not llm.missing_settings():
        return
    skip = pytest.mark.skip(reason="Foundry settings are not set (see .env.example)")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def real_collection(tmp_path_factory):
    """The real corpus, chunked and embedded into a temp store (50 chunks, a few seconds)."""
    from rag.chunking import chunk_corpus
    from rag.config import INGEST_FINGERPRINT, RAW_DIR
    from rag.embeddings import embed

    client = chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("real_store")))
    try:
        collection, _ = store.sync(chunk_corpus(RAW_DIR), embed, INGEST_FINGERPRINT, client=client)
    except OSError as exc:  # the embedding model is not cached and there is no network
        skip_or_fail(f"embedding model unavailable: {exc}")
    return collection, embed


@pytest.fixture(scope="session")
def cross_encoder():
    try:
        rerank_module.score_pairs([("q", "t")])  # loads the model once
    except OSError as exc:  # not cached and no network
        skip_or_fail(f"cross-encoder model unavailable: {exc}")
