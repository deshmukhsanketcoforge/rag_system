"""Keyword search, Reciprocal Rank Fusion and hybrid search (RAG-6). Fake embedder, temp store."""

import chromadb
import pytest

from rag import evaluation as ev
from rag import pipeline, retrieval, store
from rag.chunking import Chunk
from rag.config import RRF_K, TOP_K


def ch(id_, text, doc="doc", section="1. Section"):
    return {"id": id_, "text": text, "metadata": {"doc_name": doc, "section": section}}


def ids(hits):
    return [h["id"] for h in hits]


# --- tokenizing ---


def test_tokenize_lowercases_and_drops_stopwords():
    assert retrieval.tokenize("How many PTO days do I get?") == ["many", "pto", "days", "get"]


def test_tokenize_keeps_form_codes_and_section_numbers_whole():
    assert retrieval.tokenize("Form EX-14, see Section 7.3.") == [
        "form",
        "ex-14",
        "see",
        "section",
        "7.3",
    ]


def test_a_query_of_only_stopwords_has_no_tokens():
    assert retrieval.tokenize("what is the of a") == []


# --- keyword search ---


def test_keyword_search_ignores_case():
    hits = retrieval.keyword_search(
        "STIPEND", [ch("a", "The stipend is $750."), ch("b", "Nothing")], k=3
    )
    assert ids(hits) == ["a"]


def test_chunks_with_no_matching_word_are_left_out():
    assert retrieval.keyword_search("stipend", [ch("a", "Hotels are capped.")], k=3) == []


def test_more_matching_words_rank_higher():
    chunks = [ch("one", "stipend only"), ch("two", "stipend for the desk and chair")]
    assert ids(retrieval.keyword_search("desk chair stipend", chunks, k=2)) == ["two", "one"]


def test_a_word_matches_by_its_start_not_from_the_middle():
    chunks = [ch("mid", "Your laptop is covered."), ch("start", "Submit the form.")]
    assert ids(retrieval.keyword_search("pto top", chunks, k=3)) == []
    assert ids(retrieval.keyword_search("submitted", chunks, k=3)) == []
    assert ids(retrieval.keyword_search("submit", chunks, k=3)) == [
        "start"
    ]  # prefix of a longer word


def test_a_form_code_outweighs_several_plain_words():
    chunks = [
        ch("words", "submit the claim within thirty days of the purchase"),
        ch("code", "Form EX-14 is the claim form"),
    ]
    assert ids(retrieval.keyword_search("submit claim days EX-14", chunks, k=2))[0] == "code"


def test_a_code_does_not_match_a_longer_code():
    chunks = [ch("long", "Form EX-145 applies"), ch("exact", "Form EX-14 applies")]
    assert ids(retrieval.keyword_search("EX-14", chunks, k=3)) == ["exact"]


def test_a_section_number_matches_exactly():
    chunks = [ch("a", "see Section 17.3"), ch("b", "see Section 7.3."), ch("c", "see Section 7.35")]
    assert ids(retrieval.keyword_search("7.3", chunks, k=3)) == ["b"]


def test_an_exact_phrase_outweighs_the_same_words_scattered():
    chunks = [
        ch("scattered", "Meal costs are low. The per-diem is paid daily."),
        ch("phrase", "Receipts: not required for any meal per-diem."),
    ]
    assert ids(retrieval.keyword_search("meal per-diem", chunks, k=2))[0] == "phrase"


def test_keyword_search_returns_at_most_k_and_keeps_input_order_on_ties():
    chunks = [ch("a", "desk"), ch("b", "desk"), ch("c", "desk")]
    assert ids(retrieval.keyword_search("desk", chunks, k=2)) == ["a", "b"]


def test_a_stopword_only_query_matches_nothing():
    assert retrieval.keyword_search("what is the", [ch("a", "what is the")], k=3) == []


def test_keyword_search_does_not_change_its_input():
    chunk = ch("a", "desk")
    hits = retrieval.keyword_search("desk", [chunk], k=1)
    assert "score" not in chunk
    assert hits[0]["score"] > 0


# --- Reciprocal Rank Fusion ---


def test_rrf_scores_are_the_sum_of_one_over_k_plus_rank():
    fused = retrieval.rrf([[ch("a", "x"), ch("b", "x")], [ch("b", "x")]], k=3)
    by_id = {h["id"]: h["score"] for h in fused}
    assert by_id["a"] == pytest.approx(1 / (RRF_K + 1))
    assert by_id["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))


def test_a_chunk_found_by_both_lists_beats_one_found_by_a_single_list():
    fused = retrieval.rrf([[ch("only", "x"), ch("both", "x")], [ch("both", "x")]], k=2)
    assert ids(fused) == ["both", "only"]


def test_rrf_returns_each_chunk_once_and_at_most_k():
    fused = retrieval.rrf([[ch("a", "x"), ch("b", "x")], [ch("b", "x"), ch("c", "x")]], k=2)
    assert len(fused) == 2
    assert len(set(ids(fused))) == 2


def test_rrf_ties_go_to_the_first_list():
    fused = retrieval.rrf([[ch("v", "x")], [ch("k", "x")]], k=2)
    assert ids(fused) == ["v", "k"]


def test_rrf_keeps_the_vector_distance_when_a_chunk_has_one():
    vector_hit = {**ch("a", "x"), "distance": 0.25}
    fused = retrieval.rrf([[vector_hit], [ch("a", "x")]], k=1)
    assert fused[0]["distance"] == 0.25


# --- hybrid search ---

VECTORS = {
    "alpha": [1, 0, 0],
    "beta": [0.5, 0.5, 0],  # distinct distances from the default query vector: no ties
    "gamma": [0.2, 0, 1],
    "Form EX-14 rules": [0, 0, 1],
}


def fake_embed(texts):
    # a query with no entry points at "alpha", so vector search always prefers it
    return [VECTORS.get(t, [1, 0, 0]) for t in texts]


@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    chunks = [
        Chunk("d::0", "alpha", "doc", "1. A", 0),
        Chunk("d::1", "beta", "doc", "2. B", 1),
        Chunk("d::2", "gamma", "doc", "3. C", 2),
        Chunk("d::3", "Form EX-14 rules", "doc", "4. D", 3),
    ]
    col, _ = store.sync(chunks, fake_embed, "fp", client=client)
    return col


def test_all_chunks_returns_every_stored_chunk(collection):
    chunks = retrieval.all_chunks(collection)
    assert sorted(ids(chunks)) == ["d::0", "d::1", "d::2", "d::3"]
    assert {c["text"] for c in chunks} >= {"alpha", "Form EX-14 rules"}
    assert chunks[0]["metadata"]["doc_name"] == "doc"


def test_hybrid_surfaces_a_chunk_that_only_the_keyword_search_finds(collection):
    # Vector search ranks alpha first for this query; only keywords know about EX-14.
    vector_only = store.query(collection, fake_embed(["what is EX-14"])[0], k=1)
    assert ids(vector_only) == ["d::0"]
    hybrid = retrieval.hybrid_search("what is EX-14", collection, fake_embed, k=2)
    assert "d::3" in ids(hybrid)


def test_hybrid_results_carry_distance_for_vector_hits_and_none_otherwise(collection):
    hits = retrieval.hybrid_search("what is EX-14", collection, fake_embed, k=2, pool=1)
    by_id = {h["id"]: h for h in hits}
    assert by_id["d::0"]["distance"] == pytest.approx(0.0, abs=1e-6)  # the vector hit
    assert by_id["d::3"]["distance"] is None  # keyword-only
    assert all(h["score"] > 0 for h in hits)


def test_hybrid_returns_at_most_k_distinct_chunks(collection):
    hits = retrieval.hybrid_search("alpha beta gamma", collection, fake_embed, k=3)
    assert len(hits) == 3
    assert len(set(ids(hits))) == 3


def test_k_above_the_pool_still_returns_k_chunks(collection):
    # vector mode would return 4 here, so hybrid must not be capped at the pool of 1
    hits = retrieval.hybrid_search("alpha", collection, fake_embed, k=4, pool=1)
    assert len(hits) == 4


# --- vector vs hybrid on the real corpus and the real embedding model (screenshot #5) ---
# The `real_collection` fixture lives in tests/conftest.py (shared with test_rerank.py).


def _rank_of(hits, needle):
    return next((i for i, h in enumerate(hits, start=1) if needle in h["text"]), None)


def _show(rows):
    print("\n\nVector-only vs hybrid: rank of the chunk that holds the answer (lower is better)")
    print(f"  {'question':<28}{'answer chunk':<44}{'vector':>7}{'hybrid':>8}")
    for question, _needle, where, v, h in rows:
        print(f"  {question:<28}{where:<44}{v:>7}{h:>8}")


def test_hybrid_beats_vector(real_collection):
    """'Who signs EX-15?' hinges on a form code. Vector search buries the chunk; hybrid finds it."""
    collection, embed = real_collection
    cases = [
        ("Who signs EX-15?", "Form EX-15", "expense_reimbursement > 3. General Reqs"),
        ("What is Form PTO-1?", "Form PTO-1 at least 10", "pto_policy > 5. Requesting PTO"),
    ]
    rows = []
    for question, needle, where in cases:
        vector = store.query(collection, embed([question])[0], k=50)
        hybrid = retrieval.hybrid_search(question, collection, embed, k=50, pool=50)
        rows.append((question, needle, where, _rank_of(vector, needle), _rank_of(hybrid, needle)))
    _show(rows)

    for question, _, _, vector_rank, hybrid_rank in rows:
        assert hybrid_rank <= 3, f"hybrid did not surface the answer chunk for {question!r}"
        assert vector_rank > 3, f"vector-only already finds it for {question!r}; pick a harder case"
        assert hybrid_rank < vector_rank


def test_hybrid_at_top_k_returns_the_answer_chunk_vector_misses(real_collection):
    collection, embed = real_collection
    question = "Who signs EX-15?"
    vector = store.query(collection, embed([question])[0], k=3)
    hybrid = retrieval.hybrid_search(question, collection, embed, k=3)
    assert not any("Form EX-15" in h["text"] and "must sign" in h["text"] for h in vector)
    assert any("Form EX-15" in h["text"] and "must sign" in h["text"] for h in hybrid)


# --- gaps found by mutation checks ---


def test_repeating_a_query_word_does_not_boost_a_chunk():
    chunks = [ch("b", "chair"), ch("a", "desk")]
    assert ids(retrieval.keyword_search("desk desk chair", chunks, k=2)) == ["b", "a"]


def test_hybrid_ties_go_to_the_vector_hit(collection):
    hits = retrieval.hybrid_search("what is EX-14", collection, fake_embed, k=1, pool=1)
    assert ids(hits) == ["d::0"]  # one candidate each (d::0 by vector, d::3 by keyword), same score


def test_all_chunks_come_back_in_document_order_whatever_the_insert_order(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    reversed_chunks = [Chunk(f"d::{i}", f"t{i}", "doc", f"{i}. S", i) for i in (2, 0, 1)]
    col, _ = store.sync(
        reversed_chunks, lambda ts: [[1.0, 0.0, 0.0]] * len(ts), "fp", client=client
    )
    assert ids(retrieval.all_chunks(col)) == ["d::0", "d::1", "d::2"]


def test_the_keyword_list_is_not_cut_to_k_before_fusing(tmp_path):
    # Vector order: y, p, x. Keyword order: x, y. Fused over the full lists, y (in both near the
    # top) wins; if the keyword list were cut to k=1 first, x would win instead.
    texts = {"y": "gamma", "p": "plain", "x": "gamma delta"}
    vectors = {"y": [1, 0.1, 0], "p": [0.8, 0.6, 0], "x": [0, 0, 1]}
    by_text = {texts[n]: vectors[n] for n in texts}
    embed = lambda ts: [by_text.get(t, [1, 0, 0]) for t in ts]  # noqa: E731
    chunks = [Chunk(f"d::{i}", texts[n], "doc", f"{i}. S", i) for i, n in enumerate("ypx")]
    col, _ = store.sync(chunks, embed, "fp", client=chromadb.PersistentClient(path=str(tmp_path)))
    hits = retrieval.hybrid_search("delta gamma", col, embed, k=1)
    assert hits[0]["text"] == "gamma"


# --- review fixes ---


def test_a_stopword_in_the_question_breaks_a_phrase():
    # "reimbursement for travel" never had "reimbursement" next to "travel"
    chunks = [ch("b", "reimbursement and travel"), ch("a", "reimbursement travel")]
    assert ids(retrieval.keyword_search("reimbursement for travel", chunks, k=2)) == ["b", "a"]


def test_a_repeated_word_does_not_create_a_phrase_with_its_neighbour():
    chunks = [ch("b", "desk and chair"), ch("a", "desk chair")]
    assert ids(retrieval.keyword_search("desk desk chair", chunks, k=2)) == ["a", "b"]


def test_a_phrase_matches_inflected_words_like_single_words_do():
    chunks = [ch("scattered", "meal and reimburse"), ch("phrase", "reimbursed meals here")]
    assert ids(retrieval.keyword_search("reimburse meal", chunks, k=2)) == ["phrase", "scattered"]


def test_a_phrase_still_needs_its_words_in_order_and_adjacent():
    chunks = [ch("far", "meal costs, then per-diem"), ch("near", "meal per-diem")]
    assert ids(retrieval.keyword_search("meal per-diem", chunks, k=2)) == ["near", "far"]


def test_a_section_number_does_not_match_its_subsections():
    chunks = [
        ch("sub", "see Section 7.3.1"),
        ch("sub10", "see Section 7.3.10"),
        ch("end", "see Section 7.3."),
        ch("comma", "see Section 7.3, below"),
    ]
    assert ids(retrieval.keyword_search("7.3", chunks, k=4)) == ["end", "comma"]


def test_a_phrase_repeated_in_the_question_counts_once():
    chunks = [ch("y", "chair desk"), ch("x", "desk chair")]
    assert ids(retrieval.keyword_search("desk chair desk chair", chunks, k=2)) == ["y", "x"]


def test_a_stopword_ending_the_question_does_not_make_a_phrase():
    chunks = [ch("b", "travel and receipts"), ch("a", "travel for receipts")]
    assert ids(retrieval.keyword_search("receipts travel for", chunks, k=2)) == ["b", "a"]


def test_the_vector_list_is_not_cut_to_k_before_fusing(tmp_path):
    # Vector order: p, y. Keyword hit: y only. Over the full lists y wins (in both); with the
    # vector list cut to k=1, p and y would tie and p would win.
    vectors = {"plain": [1, 0, 0], "gamma": [0.8, 0.6, 0]}
    embed = lambda ts: [vectors.get(t, [1, 0, 0]) for t in ts]  # noqa: E731
    chunks = [Chunk(f"d::{i}", t, "doc", f"{i}. S", i) for i, t in enumerate(vectors)]
    col, _ = store.sync(chunks, embed, "fp", client=chromadb.PersistentClient(path=str(tmp_path)))
    assert retrieval.hybrid_search("gamma now", col, embed, k=1)[0]["text"] == "gamma"


def test_a_phrase_does_not_match_from_the_middle_of_a_word():
    chunks = [ch("a", "form and pto"), ch("b", "laptop form")]
    assert ids(retrieval.keyword_search("pto form", chunks, k=2)) == ["a", "b"]


# --- retrieval recall on the eval set (RAG-8): no LLM ---

RECALL_THRESHOLD = 0.8  # measured: vector 92%, hybrid 92%, rerank 100% (eval/results.md)


@pytest.mark.parametrize("mode", pipeline.MODES)
def test_recall_at_k_on_the_eval_set(mode, real_collection, cross_encoder):
    """The expected (document, section) is among the top-k chunks for most eval questions."""
    collection, embed = real_collection
    cases = ev.load_cases()
    results = ev.evaluate_retrieval(
        cases,
        lambda q: pipeline.retrieve(q, collection, embed, k=TOP_K, mode=mode),
        k=TOP_K,
    )
    print(f"\n{ev.format_report(mode, results, None, TOP_K)}")
    assert ev.recall(results) >= RECALL_THRESHOLD, [r.case.id for r in results if not r.hit]


def test_reranking_does_not_lower_recall_compared_with_vector_only(real_collection, cross_encoder):
    collection, embed = real_collection
    cases = ev.load_cases()

    def recall_of(mode):
        search = lambda q: pipeline.retrieve(q, collection, embed, k=TOP_K, mode=mode)  # noqa: E731
        return ev.recall(ev.evaluate_retrieval(cases, search, k=TOP_K))

    assert recall_of("rerank") >= recall_of("vector")
