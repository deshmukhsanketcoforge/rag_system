"""Logic of the evaluation harness (RAG-8): matching, scoring, loading. No models, no LLM."""

import json

import pytest

from rag import evaluation as ev


def case(**kw):
    base = {
        "id": "c1",
        "category": "simple fact",
        "question": "How much?",
        "expected_doc": "doc_a",
        "expected_section": "4. Equipment",
        "expected_keywords": ["$750"],
    }
    return ev.Case.from_dict({**base, **kw})


def chunk(doc, section, text="t"):
    return {"id": f"{doc}::0", "text": text, "metadata": {"doc_name": doc, "section": section}}


# --- retrieval matching ---


def test_rank_is_the_one_based_position_of_the_first_matching_chunk():
    chunks = [
        chunk("doc_b", "4. Equipment"),
        chunk("doc_a", "2. Other"),
        chunk("doc_a", "4. Equipment"),
    ]
    assert ev.retrieval_rank(chunks, case()) == 3


def test_a_chunk_needs_both_the_right_document_and_the_right_section():
    assert ev.retrieval_rank([chunk("doc_b", "4. Equipment")], case()) is None
    assert ev.retrieval_rank([chunk("doc_a", "2. Other")], case()) is None


def test_the_document_name_must_match_exactly_not_as_a_prefix():
    # the stale PTO copy must not count as the current policy
    assert ev.retrieval_rank([chunk("doc_a_v1_2022", "4. Equipment")], case()) is None


def test_the_section_must_match_exactly_not_as_a_substring():
    # a longer or renumbered title is a different section, so it must not count as a hit
    assert ev.retrieval_rank([chunk("doc_a", "4. Equipment and Reimbursements")], case()) is None
    assert ev.retrieval_rank([chunk("doc_a", "14. Equipment")], case()) is None
    assert ev.retrieval_rank([chunk("doc_a", "4. Equipment")], case()) == 1


def test_no_chunks_means_no_rank():
    assert ev.retrieval_rank([], case()) is None


def test_a_case_without_an_expected_document_cannot_be_ranked():
    with pytest.raises(ValueError, match="expected_doc"):
        ev.retrieval_rank([chunk("doc_a", "x")], case(expected_doc=None, expected_section=None))


# --- keyword matching ---


def test_all_keywords_present_means_nothing_missing():
    c = case(expected_keywords=["30 days", "EX-14"])
    assert ev.missing_keywords("Submit Form EX-14 within 30 days.", c) == []


def test_missing_keywords_are_listed():
    c = case(expected_keywords=["30 days", "EX-14"])
    assert ev.missing_keywords("Submit within 30 days.", c) == ["EX-14"]


def test_keyword_matching_ignores_case_and_markdown_bold():
    c = case(expected_keywords=["30 days", "$750"])
    assert ev.missing_keywords("Within **30 DAYS**; the stipend is **$750**.", c) == []


def test_keyword_matching_treats_non_breaking_spaces_and_line_breaks_as_spaces():
    c = case(expected_keywords=["30 days"])
    assert ev.missing_keywords("within 30 days", c) == []
    assert ev.missing_keywords("within 30\n days", c) == []


def test_a_keyword_is_not_found_inside_a_longer_number():
    c = case(expected_keywords=["5 days"])
    assert ev.missing_keywords("You may carry over 15 days.", c) == ["5 days"]


# --- running the cases ---


def test_evaluate_retrieval_reports_each_rank_and_skips_unanswerable_cases():
    cases = [
        case(id="hit"),
        case(id="miss", expected_doc="doc_z"),
        case(id="none", expected_doc=None, expected_section=None),
    ]
    results = ev.evaluate_retrieval(cases, lambda q: [chunk("doc_a", "4. Equipment")], k=3)
    assert [(r.case.id, r.rank) for r in results] == [("hit", 1), ("miss", None)]
    assert [r.hit for r in results] == [True, False]


def test_only_the_top_k_chunks_count_towards_recall():
    chunks = [chunk("x", "1")] * 3 + [chunk("doc_a", "4. Equipment")]  # the answer is at rank 4
    assert ev.evaluate_retrieval([case()], lambda q: chunks, k=3)[0].rank is None
    assert ev.evaluate_retrieval([case()], lambda q: chunks, k=4)[0].rank == 4


def test_evaluate_retrieval_asks_the_search_for_each_question():
    asked = []
    ev.evaluate_retrieval(
        [case(question="Q1"), case(id="c2", question="Q2")], lambda q: asked.append(q) or [], k=3
    )
    assert asked == ["Q1", "Q2"]


def test_evaluate_answers_marks_pass_and_fail():
    cases = [case(id="ok"), case(id="bad", expected_keywords=["$999"])]
    results = ev.evaluate_answers(cases, lambda q: "It is $750.")
    assert [(r.case.id, r.passed) for r in results] == [("ok", True), ("bad", False)]
    assert results[1].missing == ["$999"]
    assert results[0].answer == "It is $750."


def test_recall_and_accuracy_are_fractions_of_the_cases():
    retrieval = ev.evaluate_retrieval(
        [case(id="a"), case(id="b", expected_doc="x"), case(id="c"), case(id="d")],
        lambda q: [chunk("doc_a", "4. Equipment")],
        k=3,
    )
    assert ev.recall(retrieval) == 0.75
    answers = ev.evaluate_answers(
        [case(id="a"), case(id="b", expected_keywords=["zz"])], lambda q: "$750"
    )
    assert ev.accuracy(answers) == 0.5


def test_a_score_over_no_cases_is_an_error_not_a_perfect_score():
    with pytest.raises(ValueError, match="no cases"):
        ev.recall([])
    with pytest.raises(ValueError, match="no cases"):
        ev.accuracy([])


# --- loading ---


def write(tmp_path, data):
    path = tmp_path / "set.json"
    path.write_text(json.dumps(data))
    return path


GOOD = {
    "id": "x-1",
    "category": "simple fact",
    "question": "Q?",
    "expected_doc": "doc_a",
    "expected_section": "1. S",
    "expected_keywords": ["k"],
}


def test_load_cases_reads_a_valid_file(tmp_path):
    cases = ev.load_cases(write(tmp_path, [GOOD]))
    assert cases[0].id == "x-1" and cases[0].expected_keywords == ("k",)
    assert cases[0].note == ""


@pytest.mark.parametrize("missing", ["id", "question", "expected_keywords", "category"])
def test_a_missing_field_is_rejected(tmp_path, missing):
    bad = {k: v for k, v in GOOD.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        ev.load_cases(write(tmp_path, [bad]))


def test_an_unknown_field_is_rejected_so_typos_do_not_pass_silently(tmp_path):
    with pytest.raises(ValueError, match="expected_keyword"):
        ev.load_cases(write(tmp_path, [{**GOOD, "expected_keyword": ["x"]}]))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        ev.load_cases(write(tmp_path, [GOOD, GOOD]))


def test_empty_keywords_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected_keywords"):
        ev.load_cases(write(tmp_path, [{**GOOD, "expected_keywords": []}]))


def test_doc_and_section_must_be_given_together(tmp_path):
    with pytest.raises(ValueError, match="expected_section"):
        ev.load_cases(write(tmp_path, [{**GOOD, "expected_section": None}]))


# --- report ---


def test_the_report_lists_every_case_with_pass_or_fail_and_the_totals():
    cases = [case(id="a"), case(id="b", expected_doc="x")]
    retrieval = ev.evaluate_retrieval(cases, lambda q: [chunk("doc_a", "4. Equipment")], k=3)
    answers = ev.evaluate_answers(cases, lambda q: "$750")
    out = ev.format_report("hybrid", retrieval, answers, k=3)
    assert "hybrid" in out
    assert "recall@3" in out and "50%" in out  # 1 of 2
    assert "accuracy" in out and "100%" in out
    assert "| a |" in out and "| b |" in out
    assert "PASS" in out and "FAIL" in out


def test_the_report_works_without_answers():
    retrieval = ev.evaluate_retrieval([case()], lambda q: [chunk("doc_a", "4. Equipment")], k=3)
    out = ev.format_report("vector", retrieval, None, k=3)
    assert "recall@3" in out and "accuracy" not in out


# --- gaps found by mutation checks ---


def test_a_keyword_is_not_found_at_the_start_of_a_longer_number():
    assert ev.missing_keywords(
        "The rate is $0.675 per mile.", case(expected_keywords=["$0.67"])
    ) == ["$0.67"]
    assert (
        ev.missing_keywords("The rate is $0.67 per mile.", case(expected_keywords=["$0.67"])) == []
    )


def test_markdown_bold_inside_a_keyword_does_not_hide_it():
    assert ev.missing_keywords("within **30** days", case(expected_keywords=["30 days"])) == []


def test_the_report_marks_a_failed_answer_on_its_own_row():
    cases = [case(id="good"), case(id="bad", expected_keywords=["$999"])]
    retrieval = ev.evaluate_retrieval(cases, lambda q: [chunk("doc_a", "4. Equipment")], k=3)
    answers = ev.evaluate_answers(cases, lambda q: "$750")
    rows = {
        line.split("|")[1].strip(): line
        for line in ev.format_report("m", retrieval, answers, 3).splitlines()
        if line.startswith("| ") and "---" not in line
    }
    assert "FAIL" not in rows["good"] and rows["good"].count("PASS") == 2
    assert "FAIL" in rows["bad"] and "$999" in rows["bad"]
