"""Skip tests marked `llm` unless an API key is available (from the environment or .env)."""

import chromadb
import pytest
from dotenv import load_dotenv

from rag import llm, store
from rag.config import ROOT

load_dotenv(ROOT / ".env")


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
        pytest.skip(f"embedding model unavailable: {exc}")
    return collection, embed
