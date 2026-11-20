"""Pipeline tests with a fake embedder, a temp vector store and a fake LLM."""

import chromadb
import pytest

from rag import attribution, generate, pipeline, store
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
    assert set(out) == {"answer", "chunks", "sources", "sources_basis", "invalid_citations"}
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
    assert attribution.group_sources(hits, [1, 2]) == [
        {"doc_name": "d", "section": "S", "ranks": [1, 2]}
    ]


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


def test_missing_llm_settings_fail_before_any_retrieval(collection, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
    calls = []

    def embedder(texts):
        calls.append(texts)
        return fake_embed(texts)

    with pytest.raises(RuntimeError, match="ANTHROPIC_FOUNDRY_API_KEY"):
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


# --- hybrid mode (RAG-6) ---


def test_hybrid_is_a_known_mode():
    assert "hybrid" in pipeline.MODES


def test_hybrid_mode_adds_the_keyword_hit_vector_search_missed(collection):
    # The query "3" embeds to [0, 1, 0], so vector search puts "text 1" first and never reaches
    # "text 3". The keyword search matches the digit and hybrid brings that chunk in.
    vector = _answer("3", collection, k=1, mode="vector", llm=RecordingLLM())
    assert vector["chunks"][0]["id"] == "a::1"
    out = _answer("3", collection, k=2, mode="hybrid", llm=RecordingLLM())
    assert "b::1" in [c["id"] for c in out["chunks"]]
    assert all("score" in c for c in out["chunks"])


# --- rerank mode (RAG-7) ---


class FakeReranker:
    """Prefers 'text 2' over everything, then 'text 3'; records the candidates it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, pairs):
        self.calls.append([text for _, text in pairs])
        return [{"text 2": 2.0, "text 3": 1.0}.get(text, 0.0) for _, text in pairs]


def test_rerank_is_a_known_mode():
    assert "rerank" in pipeline.MODES


def test_rerank_mode_reorders_the_candidates_and_the_llm_sees_the_new_order(collection):
    llm = RecordingLLM()
    reranker = FakeReranker()
    out = _answer("text 0", collection, k=2, mode="rerank", llm=llm, reranker=reranker)
    assert [c["id"] for c in out["chunks"]] == ["b::0", "b::1"]  # text 2, text 3
    assert all("rerank_score" in c for c in out["chunks"])
    assert llm.prompts[0].index("text 2") < llm.prompts[0].index("text 3")


def test_rerank_scores_more_candidates_than_it_returns(collection):
    reranker = FakeReranker()
    out = _answer("text 0", collection, k=1, mode="rerank", llm=RecordingLLM(), reranker=reranker)
    assert len(out["chunks"]) == 1
    assert len(reranker.calls[0]) == 4  # every chunk in the store, below RERANK_CANDIDATES


def test_a_reranker_with_another_mode_is_rejected_not_ignored(collection):
    reranker = FakeReranker()
    for mode in ("vector", "hybrid"):
        with pytest.raises(ValueError, match="reranker"):
            _answer("text 0", collection, mode=mode, llm=RecordingLLM(), reranker=reranker)
    assert reranker.calls == []


def test_rerank_mode_loads_the_cross_encoder_before_any_retrieval(collection, monkeypatch):
    embedded = []

    def spy_embed(texts):
        embedded.append(texts)
        return fake_embed(texts)

    def cannot_load():
        raise RuntimeError("the cross-encoder could not be loaded")

    monkeypatch.setattr(pipeline.rr, "load_scorer", cannot_load)
    with pytest.raises(RuntimeError, match="cross-encoder"):
        pipeline.answer(
            "q", mode="rerank", collection=collection, embedder=spy_embed, llm=RecordingLLM()
        )
    assert embedded == []  # failed before the question was embedded


def test_rerank_mode_uses_the_loaded_scorer_when_none_is_passed(collection, monkeypatch):
    reranker = FakeReranker()
    monkeypatch.setattr(pipeline.rr, "load_scorer", lambda: reranker)
    out = pipeline.answer(
        "text 0", k=1, mode="rerank", collection=collection, embedder=fake_embed, llm=RecordingLLM()
    )
    assert out["chunks"][0]["id"] == "b::0" and len(reranker.calls) == 1


# --- resolve_defaults and answer_from_chunks (shared with scripts/run_eval.py)


def test_resolve_defaults_hands_back_what_it_was_given(collection):
    got = pipeline.resolve_defaults(collection, fake_embed)
    assert got == (collection, fake_embed, None)


def test_resolve_defaults_rejects_an_empty_store(tmp_path):
    empty = chromadb.PersistentClient(path=str(tmp_path)).create_collection("policies")
    with pytest.raises(RuntimeError, match="02_ingest"):
        pipeline.resolve_defaults(empty, fake_embed)


def test_resolve_defaults_loads_the_cross_encoder_only_when_asked(collection, monkeypatch):
    loaded = []
    monkeypatch.setattr(pipeline.rr, "load_scorer", lambda: loaded.append(1) or "scorer")
    assert pipeline.resolve_defaults(collection, fake_embed)[2] is None
    assert pipeline.resolve_defaults(collection, fake_embed, rerank=True)[2] == "scorer"
    assert loaded == [1]


def test_resolve_defaults_keeps_a_given_reranker(collection, monkeypatch):
    def not_called():
        raise AssertionError("must not load a second scorer")

    monkeypatch.setattr(pipeline.rr, "load_scorer", not_called)
    mine = object()
    assert pipeline.resolve_defaults(collection, fake_embed, mine, rerank=True)[2] is mine


def test_answer_from_chunks_generates_from_exactly_the_chunks_it_is_given():
    llm = RecordingLLM("ANSWER")
    chunks = [
        {"id": "a::0", "text": "text 0", "metadata": {"doc_name": "doc_a", "section": "1. Alpha"}}
    ]
    out = pipeline.answer_from_chunks("q", chunks, llm=llm)
    assert out["answer"] == "ANSWER" and out["chunks"] == chunks
    assert out["sources"] == [{"doc_name": "doc_a", "section": "1. Alpha", "ranks": [1]}]
    assert out["sources_basis"] == "retrieved"  # the stub answer cites nothing
    assert "text 0" in llm.prompts[0]


def test_answer_from_chunks_with_nothing_retrieved_gives_the_fallback_without_an_llm_call():
    llm = RecordingLLM()
    out = pipeline.answer_from_chunks("q", [], llm=llm)
    assert out["answer"] == generate.NOT_FOUND_ANSWER and out["sources"] == []
    assert llm.prompts == []


# --- source attribution (RAG-10) ---


def test_the_sources_are_the_passages_the_answer_cites(collection):
    out = _answer("text 0", collection, k=3, llm=RecordingLLM(reply="It is 20 days [2]."))
    assert out["sources_basis"] == "cited"
    assert [s["ranks"] for s in out["sources"]] == [[2]]
    cited = out["chunks"][1]["metadata"]
    assert out["sources"][0]["doc_name"] == cited["doc_name"]
    assert out["sources"][0]["section"] == cited["section"]
    assert out["invalid_citations"] == []


def test_a_citation_to_a_passage_that_was_not_retrieved_is_reported_not_trusted(collection):
    out = _answer("text 0", collection, k=3, llm=RecordingLLM(reply="Fact [1]. Other [9]."))
    assert out["invalid_citations"] == [9]
    assert [s["ranks"] for s in out["sources"]] == [[1]]


def test_an_answer_that_cites_nothing_falls_back_to_the_retrieved_chunks_and_says_so(collection):
    out = _answer("text 0", collection, k=3, llm=RecordingLLM(reply="No numbers here."))
    assert out["sources_basis"] == "retrieved"
    assert len(out["sources"]) == 3


def test_the_fallback_answer_has_basis_none(collection):
    out = _answer("text 0", collection, llm=RecordingLLM(reply=generate.NOT_FOUND_ANSWER))
    assert out["sources_basis"] == "none" and out["sources"] == []


@pytest.mark.parametrize("mode", pipeline.MODES)
def test_every_mode_keeps_the_document_and_section_on_every_chunk(mode, collection):
    chunks = pipeline.retrieve(
        "text 0", collection, fake_embed, k=4, mode=mode, reranker=FakeReranker()
    )
    assert chunks
    for chunk in chunks:
        assert chunk["metadata"]["doc_name"] in {"doc_a", "doc_b"}
        assert chunk["metadata"]["section"]
        assert "chunk_index" in chunk["metadata"]
