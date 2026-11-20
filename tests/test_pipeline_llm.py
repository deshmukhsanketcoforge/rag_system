"""End-to-end tests against the real embedding model and the real LLM.

Skipped without the Foundry settings (see conftest.py). They build their own temporary vector
store from data/raw, so they do not depend on or touch chroma_db.
"""

import chromadb
import pytest

from rag import generate, pipeline, store
from rag.chunking import chunk_corpus
from rag.config import INGEST_FINGERPRINT, RAW_DIR

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    from rag.embeddings import embed

    client = chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("chroma")))
    col, _ = store.sync(chunk_corpus(RAW_DIR), embed, INGEST_FINGERPRINT, client=client)
    return col


def test_grounded_answer_with_the_right_source(collection):
    out = pipeline.answer("How much is the home office equipment stipend?", collection=collection)
    assert "$750" in out["answer"]
    assert out["sources"][0]["doc_name"] == "remote_work_policy"


def test_unanswerable_question_gets_the_fallback(collection):
    out = pipeline.answer("What is the CEO's favourite colour?", collection=collection)
    # The prompt asks for this exact sentence; tolerate only a curly apostrophe.
    assert out["answer"].replace("\u2019", "'").strip() == generate.NOT_FOUND_ANSWER
    assert out["sources"] == []
