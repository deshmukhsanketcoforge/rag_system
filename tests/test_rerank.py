"""Cross-encoder reranking (RAG-7). A fake scorer stands in for the model, so nothing downloads."""

import pytest

from rag import rerank


def ch(id_, text="t"):
    return {"id": id_, "text": text, "metadata": {"doc_name": "d", "section": "S"}, "distance": 0.5}


def ids(hits):
    return [h["id"] for h in hits]


class Scorer:
    """Scores a (query, text) pair by a lookup on the text; records every call."""

    def __init__(self, by_text):
        self.by_text = by_text
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(self, pairs):
        self.calls.append(list(pairs))
        return [self.by_text[text] for _, text in pairs]


def test_chunks_come_back_best_score_first():
    chunks = [ch("a", "low"), ch("b", "high"), ch("c", "mid")]
    scorer = Scorer({"low": 0.1, "high": 0.9, "mid": 0.5})
    assert ids(rerank.rerank("q", chunks, k=3, scorer=scorer)) == ["b", "c", "a"]


def test_only_the_top_k_are_returned():
    chunks = [ch("a", "low"), ch("b", "high"), ch("c", "mid")]
    scorer = Scorer({"low": 0.1, "high": 0.9, "mid": 0.5})
    assert ids(rerank.rerank("q", chunks, k=2, scorer=scorer)) == ["b", "c"]


def test_k_above_the_candidate_count_returns_them_all():
    chunks = [ch("a", "x"), ch("b", "y")]
    assert len(rerank.rerank("q", chunks, k=10, scorer=Scorer({"x": 1, "y": 2}))) == 2


def test_the_scorer_gets_every_query_chunk_pair_once_in_input_order():
    chunks = [ch("a", "one"), ch("b", "two")]
    scorer = Scorer({"one": 1, "two": 2})
    rerank.rerank("my question", chunks, k=1, scorer=scorer)
    assert scorer.calls == [[("my question", "one"), ("my question", "two")]]


def test_each_chunk_gets_a_rerank_score_and_keeps_its_other_fields():
    out = rerank.rerank("q", [ch("a", "x")], k=1, scorer=Scorer({"x": 0.75}))[0]
    assert out["rerank_score"] == 0.75
    assert out["id"] == "a" and out["distance"] == 0.5
    assert out["metadata"]["doc_name"] == "d"


def test_ties_keep_the_input_order():
    chunks = [ch("a", "x"), ch("b", "y"), ch("c", "z")]
    scorer = Scorer({"x": 1.0, "y": 1.0, "z": 1.0})
    assert ids(rerank.rerank("q", chunks, k=3, scorer=scorer)) == ["a", "b", "c"]


def test_the_input_chunks_are_not_modified():
    chunk = ch("a", "x")
    rerank.rerank("q", [chunk], k=1, scorer=Scorer({"x": 1}))
    assert "rerank_score" not in chunk


def test_no_candidates_means_no_scorer_call():
    scorer = Scorer({})
    assert rerank.rerank("q", [], k=3, scorer=scorer) == []
    assert scorer.calls == []


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError, match="k"):
        rerank.rerank("q", [ch("a")], k=0, scorer=Scorer({"t": 1}))


def test_a_scorer_that_returns_the_wrong_number_of_scores_is_rejected():
    with pytest.raises(ValueError, match="one score per"):
        rerank.rerank("q", [ch("a"), ch("b")], k=1, scorer=lambda pairs: [1.0])


def test_scores_are_plain_floats_even_when_the_model_returns_numpy_values():
    import numpy as np

    out = rerank.rerank("q", [ch("a", "x")], k=1, scorer=lambda pairs: np.array([0.25], "float32"))
    assert type(out[0]["rerank_score"]) is float


def test_a_model_that_cannot_be_loaded_gives_a_clear_error(monkeypatch):
    def offline():
        raise OSError("couldn't connect to huggingface.co")

    monkeypatch.setattr(rerank, "_model", offline)
    with pytest.raises(RuntimeError, match="cross-encoder") as err:
        rerank.load_scorer()
    assert "huggingface" in str(err.value)  # the underlying reason is kept


def test_load_scorer_returns_the_scoring_function(monkeypatch):
    monkeypatch.setattr(rerank, "_model", lambda: object())
    assert rerank.load_scorer() is rerank.score_pairs


# The `real_collection` fixture lives in tests/conftest.py (shared with test_retrieval.py).
# --- hybrid vs hybrid + rerank on the real corpus and the real cross-encoder (screenshot #6) ---


@pytest.fixture(scope="module")
def cross_encoder():
    try:
        rerank.score_pairs([("q", "t")])  # loads the model once
    except OSError as exc:  # not cached and no network
        pytest.skip(f"cross-encoder model unavailable: {exc}")


def _rank_of(chunks, needle):
    return next((i for i, c in enumerate(chunks, start=1) if needle in c["text"]), None)


def _label(chunk):
    meta = chunk["metadata"]
    return f"{meta['doc_name']} > {meta['section']}"


def _candidates(collection, embed, question):
    from rag import retrieval
    from rag.config import RERANK_CANDIDATES

    return retrieval.hybrid_search(question, collection, embed, k=RERANK_CANDIDATES)


def test_rerank_beats_hybrid(real_collection, cross_encoder):
    """'What is the mileage rate for my own car?' has no word in common with 'per mile'.

    Hybrid ranks the chunk that holds the answer far down; the cross-encoder reads question and
    chunk together and puts it first. Run with -s to see both orderings.
    """
    collection, embed = real_collection
    cases = [
        ("What is the mileage rate for my own car?", "$0.67 per mile"),
        ("When do I get paid back after claiming?", "paid by direct deposit"),
    ]
    rows = []
    for question, needle in cases:
        hybrid = _candidates(collection, embed, question)
        reranked = rerank.rerank(question, hybrid, k=len(hybrid))
        rows.append((question, needle, hybrid, reranked))
        print(f"\n\nQuestion: {question}")
        for title, chunks in (("before (hybrid)", hybrid), ("after (rerank)", reranked)):
            print(f"  {title}:")
            for i, chunk in enumerate(chunks[:5], start=1):
                mark = "  <- answer" if needle in chunk["text"] else ""
                print(f"    {i}. {_label(chunk)}{mark}")

    for question, needle, hybrid, reranked in rows:
        before, after = _rank_of(hybrid, needle), _rank_of(reranked, needle)
        assert after == 1, f"rerank should put the answer chunk first for {question!r}"
        assert after < before, f"order did not improve for {question!r}"
    # the headline case: hybrid at the app's k=3 misses the chunk, rerank returns it
    question, needle, hybrid, reranked = rows[0]
    assert _rank_of(hybrid, needle) > 3


def test_rerank_at_top_k_returns_the_answer_chunk_hybrid_misses(real_collection, cross_encoder):
    from rag import retrieval

    collection, embed = real_collection
    question = "What is the mileage rate for my own car?"
    top3_hybrid = retrieval.hybrid_search(question, collection, embed, k=3)
    top3_rerank = rerank.rerank(question, _candidates(collection, embed, question), k=3)
    assert not any("$0.67 per mile" in c["text"] for c in top3_hybrid)
    assert any("$0.67 per mile" in c["text"] for c in top3_rerank)
