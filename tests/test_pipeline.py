"""Pipeline tests with a fake embedder, a temp vector store and a fake LLM."""

import chromadb
import pytest

from rag import generate, pipeline, store
from rag.chunking import Chunk

VECTORS = {
    "text 0": [1, 0, 0],
    "text 1": [0, 1, 0],
    "text 2": [0, 0, 1],
    "text 3": [0, 0, 1],
}


def fake_embed(texts):
    return [VECTORS.get(t, [0, 1, 0]) for t in texts]


class RecordingLLM:
    def __init__(self, reply="ANSWER"):
        self.reply = reply
        self.prompts: list[str] = []
        self.messages = self

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        block = type("B", (), {"type": "text", "text": self.reply})()
        return type("R", (), {"content": [block], "stop_reason": "end_turn"})()


@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    chunks = [
        Chunk("a::0", "text 0", "doc_a", "1. Alpha", 0),
        Chunk("a::1", "text 1", "doc_a", "2. Beta", 1),
        Chunk("b::0", "text 2", "doc_b", "1. Gamma", 0),
        Chunk("b::1", "text 3", "doc_b", "2. Delta", 1),
    ]
    col, _ = store.sync(chunks, fake_embed, "fp", client=client)
    return col


def _answer(question, collection, **kw):
    return pipeline.answer(question, collection=collection, embedder=fake_embed, **kw)


def test_answer_returns_answer_chunks_and_sources(collection):
    llm = RecordingLLM()
    out = _answer("text 1", collection, k=2, llm=llm)
    assert set(out) == {"answer", "chunks", "sources"}
    assert out["answer"] == "ANSWER"
    assert out["chunks"][0]["id"] == "a::1"  # nearest chunk first
    assert len(out["chunks"]) == 2


def test_the_retrieved_chunks_are_what_the_llm_sees(collection):
    llm = RecordingLLM()
    _answer("text 0", collection, k=1, llm=llm)
    assert "text 0" in llm.prompts[0]
    assert "text 1" not in llm.prompts[0]


def test_k_limits_how_many_chunks_are_retrieved(collection):
    assert len(_answer("text 0", collection, k=1, llm=RecordingLLM())["chunks"]) == 1
    assert len(_answer("text 0", collection, k=3, llm=RecordingLLM())["chunks"]) == 3


def test_default_k_comes_from_config(collection):
    from rag.config import TOP_K

    assert len(_answer("text 0", collection, llm=RecordingLLM())["chunks"]) == min(TOP_K, 4)


def test_sources_list_each_document_section_once_in_rank_order(collection):
    out = _answer("text 2", collection, k=4, llm=RecordingLLM())
    pairs = [(s["doc_name"], s["section"]) for s in out["sources"]]
    assert pairs[0][0] == "doc_b"  # text 2 and text 3 tie as the nearest; both are in doc_b
    assert pairs[1][0] == "doc_b"
    assert len(pairs) == len(set(pairs)) == 4


def test_sources_are_deduplicated_when_a_section_appears_twice(collection):
    hits = [
        {"id": "x", "text": "t", "metadata": {"doc_name": "d", "section": "S"}, "distance": 0.1},
        {"id": "y", "text": "t", "metadata": {"doc_name": "d", "section": "S"}, "distance": 0.2},
    ]
    assert pipeline.sources_of(hits) == [{"doc_name": "d", "section": "S"}]


def test_unknown_mode_is_rejected(collection):
    with pytest.raises(ValueError, match="mode"):
        _answer("q", collection, mode="bm25-only", llm=RecordingLLM())


def test_blank_question_is_rejected_before_any_work(collection):
    llm = RecordingLLM()
    with pytest.raises(ValueError, match="question"):
        _answer("   ", collection, llm=llm)
    assert llm.prompts == []


def test_fallback_answer_has_no_sources(collection):
    out = _answer("text 0", collection, llm=RecordingLLM(reply=generate.NOT_FOUND_ANSWER))
    assert out["answer"] == generate.NOT_FOUND_ANSWER
    assert out["sources"] == []
    assert len(out["chunks"]) == 3  # still shown, so the retrieval can be inspected


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(collection, k):
    llm = RecordingLLM()
    with pytest.raises(ValueError, match="k must"):
        _answer("q", collection, k=k, llm=llm)
    assert llm.prompts == []


def test_empty_collection_is_a_clear_error(tmp_path):
    empty = chromadb.PersistentClient(path=str(tmp_path)).create_collection("policies")
    with pytest.raises(RuntimeError, match="02_ingest"):
        _answer("q", empty, llm=RecordingLLM())


def test_missing_collection_is_a_clear_error(monkeypatch):
    from chromadb.errors import NotFoundError

    def boom():
        raise NotFoundError("no collection")

    monkeypatch.setattr(store, "get_collection", boom)
    with pytest.raises(RuntimeError, match="02_ingest"):
        pipeline.answer("q", embedder=fake_embed, llm=RecordingLLM())


def test_missing_api_key_fails_before_any_retrieval(collection, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls = []

    def embedder(texts):
        calls.append(texts)
        return fake_embed(texts)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        pipeline.answer("q", collection=collection, embedder=embedder)
    assert calls == []


def test_question_is_stripped_before_embedding_and_prompting(collection):
    seen, llm = [], RecordingLLM()

    def embedder(texts):
        seen.extend(texts)
        return fake_embed(texts)

    pipeline.answer("  text 0\n", collection=collection, embedder=embedder, llm=llm)
    assert seen == ["text 0"]
    assert "<question>text 0</question>" in llm.prompts[0]
