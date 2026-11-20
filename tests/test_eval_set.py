"""The real eval/test_set.json must be sound: enough cases, and every expectation exists in the
corpus. Offline (chunking only, no models), so a typo in the set fails fast."""

import pytest

from rag import evaluation as ev
from rag.chunking import chunk_corpus
from rag.config import RAW_DIR

CASES = ev.load_cases()
CHUNKS = chunk_corpus(RAW_DIR)
ANSWERABLE = [c for c in CASES if c.expected_doc]


def test_the_set_has_at_least_eight_cases():
    assert len(CASES) >= 8


def test_the_set_mixes_the_kinds_of_question_the_plan_asks_for():
    kinds = {c.category for c in CASES}
    assert {"simple fact", "cross-section", "exact code", "planted issue", "unanswerable"} <= kinds
    assert sum(c.category == "planted issue" for c in CASES) >= 2


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda c: c.id)
def test_the_expected_section_exists_in_the_expected_document(case):
    found = any(
        c.doc_name == case.expected_doc and case.expected_section == c.section for c in CHUNKS
    )
    assert found, f"{case.id}: {case.expected_doc} has no section {case.expected_section!r}"


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda c: c.id)
def test_the_expected_keywords_appear_in_the_expected_section(case):
    text = " ".join(
        c.text
        for c in CHUNKS
        if c.doc_name == case.expected_doc and case.expected_section == c.section
    )
    # keywords describe facts the section states, so the answer can be checked against its source
    for keyword in case.expected_keywords:
        assert ev.contains(text, keyword), f"{case.id}: {keyword!r} is not in the expected section"


def test_the_planted_issue_cases_point_at_the_current_policy_not_the_stale_copy():
    for c in CASES:
        if c.category == "planted issue":
            assert c.expected_doc == "pto_policy"
            assert c.note  # says which answer is expected and why


def test_the_unanswerable_case_expects_the_fixed_fallback():
    from rag.generate import NOT_FOUND_ANSWER

    (none,) = [c for c in CASES if c.category == "unanswerable"]
    assert none.expected_doc is None
    assert ev.contains(NOT_FOUND_ANSWER, none.expected_keywords[0])
