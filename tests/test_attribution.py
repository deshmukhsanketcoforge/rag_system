"""Source attribution (RAG-10): which retrieved passages informed an answer, validated.

The model is asked to cite passages by number ([1], [1, 2]). Those numbers are checked against
the passages it was actually given, so a made-up citation can never become a source.
"""

import pytest

from rag import attribution as at
from rag import evaluation as ev
from rag import pipeline
from rag.chunking import chunk_corpus
from rag.config import RAW_DIR, TOP_K
from rag.generate import NOT_FOUND_ANSWER


def chunk(doc, section, text="t", **extra):
    return {
        "id": f"{doc}::{section}",
        "text": text,
        "metadata": {"doc_name": doc, "section": section},
        **extra,
    }


CHUNKS = [
    chunk("pto_policy", "3. Annual Allowance"),  # passage 1
    chunk("pto_policy_v1_2022", "3. Annual Allowance"),  # passage 2
    chunk("pto_policy", "4. Carryover"),  # passage 3
]


# --- reading the citations out of an answer ---


def test_a_single_citation_is_read():
    assert at.parse_citations("You get 20 days [1].") == [1]


def test_several_numbers_in_one_bracket_are_read():
    assert at.parse_citations("Up to 5 days [1, 3].") == [1, 3]
    assert at.parse_citations("Up to 5 days [1,3].") == [1, 3]


def test_adjacent_brackets_are_read_in_order_of_appearance_without_repeats():
    assert at.parse_citations("A [3][1]. B [1] [2].") == [3, 1, 2]


def test_brackets_that_are_not_citations_are_ignored():
    text = "See [Section 3.1] and [] and [1a] and [a1] and (2) and [1.5]."
    assert at.parse_citations(text) == []


def test_an_answer_with_no_citation_has_none():
    assert at.parse_citations("You get 20 days.") == []


def test_a_citation_works_inside_markdown_bold_and_lists():
    assert at.parse_citations("- **20 days** [2]\n- sick leave: 8 days [1]") == [2, 1]


# --- turning citations into sources ---


def test_sources_are_only_the_cited_passages():
    out = at.attribute("You get 20 days [1].", CHUNKS)
    assert out["basis"] == "cited"
    assert out["sources"] == [
        {"doc_name": "pto_policy", "section": "3. Annual Allowance", "ranks": [1]}
    ]
    assert out["invalid"] == []


def test_sources_follow_passage_order_not_citation_order():
    out = at.attribute("A [3]. B [1].", CHUNKS)
    assert [s["ranks"] for s in out["sources"]] == [[1], [3]]


def test_two_cited_passages_from_the_same_section_are_one_source_with_both_ranks():
    chunks = [chunk("d", "S", "a"), chunk("x", "Y"), chunk("d", "S", "b")]
    out = at.attribute("Fact [1][3].", chunks)
    assert out["sources"] == [{"doc_name": "d", "section": "S", "ranks": [1, 3]}]


def test_a_citation_to_a_passage_that_was_not_given_is_reported_and_not_a_source():
    out = at.attribute("Fact [1]. Other [7].", CHUNKS)
    assert out["invalid"] == [7]
    assert [s["ranks"] for s in out["sources"]] == [[1]]
    assert all(s["doc_name"] in {c["metadata"]["doc_name"] for c in CHUNKS} for s in out["sources"])


def test_passage_zero_and_negative_numbers_are_not_passages():
    assert at.attribute("Fact [0].", CHUNKS)["invalid"] == [0]


def test_when_nothing_valid_is_cited_the_sources_are_the_retrieved_chunks_and_say_so():
    out = at.attribute("You get 20 days.", CHUNKS)
    assert out["basis"] == "retrieved"
    assert [s["ranks"] for s in out["sources"]] == [[1], [2], [3]]
    only_invalid = at.attribute("Fact [9].", CHUNKS)
    assert only_invalid["basis"] == "retrieved" and only_invalid["invalid"] == [9]


def test_the_retrieved_fallback_groups_a_repeated_section():
    chunks = [chunk("d", "S", "a"), chunk("d", "S", "b")]
    out = at.attribute("Fact.", chunks)
    assert out["sources"] == [{"doc_name": "d", "section": "S", "ranks": [1, 2]}]


def test_the_not_found_answer_has_no_sources_even_if_it_somehow_cites():
    out = at.attribute(NOT_FOUND_ANSWER, CHUNKS)
    assert out == {"sources": [], "basis": "none", "invalid": []}


def test_no_chunks_means_no_sources():
    assert at.attribute("Anything [1].", [])["sources"] == []
    assert at.attribute("Anything [1].", [])["basis"] == "none"


def test_the_fallback_sentence_is_recognised_with_a_curly_apostrophe_and_trailing_space():
    curly = NOT_FOUND_ANSWER.replace("'", "’") + " "
    assert at.attribute(curly, CHUNKS)["basis"] == "none"


def test_attribute_does_not_modify_the_chunks():
    chunks = [chunk("d", "S")]
    before = repr(chunks)
    at.attribute("Fact [1].", chunks)
    assert repr(chunks) == before


@pytest.mark.parametrize("ranks", [[1], [2], [1, 2, 3]])
def test_every_source_names_a_document_and_section_from_the_chunks(ranks):
    text = "Fact " + "".join(f"[{r}]" for r in ranks)
    for s in at.attribute(text, CHUNKS)["sources"]:
        assert (s["doc_name"], s["section"]) in {
            (c["metadata"]["doc_name"], c["metadata"]["section"]) for c in CHUNKS
        }


# --- on the real corpus and models ---

CORPUS_PAIRS = {(c.doc_name, c.section) for c in chunk_corpus(RAW_DIR)}
CORPUS_DOCS = {p.stem for p in RAW_DIR.glob("*.md")}


@pytest.mark.parametrize("mode", pipeline.MODES)
def test_every_mode_returns_chunks_with_a_real_document_and_section(mode, real_collection, request):
    """doc_name and section survive vector search, the hybrid merge and reranking."""
    if mode == "rerank":
        request.getfixturevalue("cross_encoder")
    collection, embed = real_collection
    for case in ev.load_cases():
        chunks = pipeline.retrieve(case.question, collection, embed, k=TOP_K, mode=mode)
        assert chunks
        for c in chunks:
            meta = c["metadata"]
            assert meta["doc_name"] in CORPUS_DOCS
            assert (meta["doc_name"], meta["section"]) in CORPUS_PAIRS


# the live model must follow the citation instruction on real questions
LIVE_CASES = ["expense-01", "expense-02", "remote-01", "security-01", "pto-02"]


@pytest.mark.llm
@pytest.mark.parametrize("case_id", LIVE_CASES)
def test_a_real_answer_cites_the_section_that_holds_the_fact(
    case_id, real_collection, cross_encoder
):
    collection, embed = real_collection
    case = next(c for c in ev.load_cases() if c.id == case_id)
    out = pipeline.answer(
        case.question, k=TOP_K, mode="rerank", collection=collection, embedder=embed
    )
    assert out["sources_basis"] == "cited", out["answer"]
    assert out["sources"], out["answer"]
    assert out["invalid_citations"] == []
    pairs = {(s["doc_name"], s["section"]) for s in out["sources"]}
    assert pairs <= CORPUS_PAIRS
    assert {s["doc_name"] for s in out["sources"]} <= CORPUS_DOCS
    assert (case.expected_doc, case.expected_section) in pairs, (pairs, out["answer"])
    for s in out["sources"]:
        assert all(1 <= r <= len(out["chunks"]) for r in s["ranks"])


# --- gaps found by mutation checks ---


def test_passage_zero_is_never_a_source_and_does_not_count_as_a_valid_citation():
    out = at.attribute("Fact [0].", CHUNKS)
    assert out["invalid"] == [0]
    assert out["basis"] == "retrieved"
    assert all(0 not in s["ranks"] for s in out["sources"])


def test_the_number_just_past_the_last_passage_is_invalid():
    out = at.attribute("Fact [3] and [4].", CHUNKS)  # three passages were given
    assert out["invalid"] == [4]
    assert [s["ranks"] for s in out["sources"]] == [[3]]
    assert at.attribute("Fact [4].", CHUNKS)["basis"] == "retrieved"
