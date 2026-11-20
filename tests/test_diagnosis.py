"""RAG-9 diagnosis: the planted data issue, what a flawed answer looks like, and the evidence that
the cause is the source data and not retrieval. Offline unless a test says otherwise."""

import chromadb
import pytest

from rag import diagnosis as dg
from rag import generate as gen
from rag import pipeline, store
from rag.chunking import Chunk, chunk_corpus
from rag.config import INGEST_FINGERPRINT, RAW_DIR, TOP_K

CHUNKS = chunk_corpus(RAW_DIR)


def q(id_="annual"):
    return next(x for x in dg.QUESTIONS if x.id == id_)


# --- classifying an answer ---


def test_an_answer_with_only_the_current_value_is_correct():
    assert dg.classify("Full-time employees get **20 days** of PTO per year.", q()) == "correct"


def test_an_answer_that_also_mentions_the_stale_value_mixes_both():
    answer = "You get 20 days. An older document says 15 days."
    assert dg.classify(answer, q()) == "mixes both"


def test_naming_the_stale_document_counts_as_mentioning_it():
    answer = "You get 20 days, but pto_policy_v1_2022 differs."
    assert dg.classify(answer, q()) == "mixes both"


def test_an_answer_with_only_the_stale_value_is_outdated():
    assert dg.classify("Employees receive 15 days of PTO.", q()) == "outdated"


def test_an_answer_with_neither_value_gives_no_answer():
    assert dg.classify("I don't have enough information.", q()) == "no answer"


def test_a_hedge_between_the_two_values_is_not_correct():
    answer = "The policies conflict: 5 days in one, 10 days in the other. Please confirm with HR."
    assert dg.classify(answer, q("carryover")) == "mixes both"


def test_the_stale_value_is_matched_as_a_whole_token():
    # "15 days" must not be found inside "115 days", and "5 days" not inside "15 days"
    assert dg.classify("You get 20 days; carryover is 5 days.", q()) == "correct"
    assert dg.classify("Carry over up to 5 days.", q("carryover")) == "correct"
    assert dg.classify("You get 15 days.", q("carryover")) == "no answer"


# --- the source data contains the contradiction (Q2) ---


def test_every_planted_conflict_is_really_in_the_source_documents():
    assert dg.verify_conflicts(CHUNKS) == []


def test_verify_conflicts_reports_a_fact_that_is_not_in_the_source():
    chunks = [c for c in CHUNKS if c.doc_name != dg.STALE_DOC]
    problems = dg.verify_conflicts(chunks)
    assert problems and all(dg.STALE_DOC in p for p in problems)


def test_there_are_six_conflicts_between_the_two_policies():
    assert len(dg.CONFLICTS) == 6
    assert {c.current.doc for c in dg.CONFLICTS} == {dg.CURRENT_DOC}
    assert {c.stale.doc for c in dg.CONFLICTS} == {dg.STALE_DOC}


def test_the_primary_and_carryover_questions_are_two_of_the_conflicts():
    values = {(c.current.snippet, c.stale.snippet) for c in dg.CONFLICTS}
    for question in dg.QUESTIONS:
        assert (question.current_snippet, question.stale_snippet) in values


# --- nothing marks the old document as outdated ---


def test_status_markers_are_found_whatever_the_case():
    assert dg.status_markers("This policy was SUPERSEDED in 2024.") == ["superseded"]
    assert dg.status_markers("Replaced by v2; no longer in force") == ["replaced", "no longer"]


def test_a_plain_policy_text_has_no_status_markers():
    assert dg.status_markers("Employees receive 15 days of PTO per year.") == []


def test_the_stale_document_has_no_marker_saying_it_is_outdated():
    text = " ".join(c.text for c in CHUNKS if c.doc_name == dg.STALE_DOC)
    assert dg.status_markers(text) == []


def test_chunk_metadata_carries_no_version_or_date_the_retriever_could_use():
    stale = [c for c in CHUNKS if c.doc_name == dg.STALE_DOC]
    current = [c for c in CHUNKS if c.doc_name == dg.CURRENT_DOC]
    assert stale and current
    for chunk in stale + current:
        assert set(chunk.metadata) == {"doc_name", "section", "chunk_index"}


def test_both_policies_share_the_same_title_and_section_layout():
    def titles(doc):
        return {c.section for c in CHUNKS if c.doc_name == doc}

    shared = titles(dg.CURRENT_DOC) & titles(dg.STALE_DOC)
    assert {"3. Annual Allowance", "4. Carryover", "5. Requesting PTO"} <= shared
    title = lambda doc: next(c.text for c in CHUNKS if c.doc_name == doc).splitlines()[0]  # noqa: E731
    assert title(dg.CURRENT_DOC) == title(dg.STALE_DOC)


@pytest.mark.parametrize("question", dg.QUESTIONS, ids=lambda x: x.id)
def test_each_question_names_a_section_that_exists_in_both_documents(question):
    for doc in (dg.CURRENT_DOC, dg.STALE_DOC):
        assert any(c.doc_name == doc and question.section in c.section for c in CHUNKS)


# --- the evidence on the real corpus and real models (Q1: retrieval; control test) ---


def _rank(chunks, doc, section):
    return next(
        (
            i
            for i, c in enumerate(chunks, start=1)
            if c["metadata"]["doc_name"] == doc and c["metadata"]["section"] == section
        ),
        None,
    )


def _retrieve(collection, embed, question, mode, request):
    if mode == "rerank":
        request.getfixturevalue("cross_encoder")
    return pipeline.retrieve(question.question, collection, embed, k=TOP_K, mode=mode)


@pytest.fixture(scope="module")
def control_collection(real_collection, tmp_path_factory):
    """The real corpus with pto_policy_v1_2022 removed through the normal incremental sync."""
    _, embed = real_collection
    client = chromadb.PersistentClient(path=str(tmp_path_factory.mktemp("control_store")))
    store.sync(CHUNKS, embed, INGEST_FINGERPRINT, client=client)  # everything first
    kept = [c for c in CHUNKS if c.doc_name != dg.STALE_DOC]
    collection, result = store.sync(kept, embed, INGEST_FINGERPRINT, client=client)
    return collection, embed, result


@pytest.mark.parametrize("mode", pipeline.MODES)
@pytest.mark.parametrize("question", dg.QUESTIONS, ids=lambda x: x.id)
def test_retrieval_returns_the_matching_section_of_both_policies(
    mode, question, real_collection, request
):
    """Q1: the retriever did its job. Both versions are relevant, and both come back."""
    collection, embed = real_collection
    chunks = _retrieve(collection, embed, question, mode, request)
    assert _rank(chunks, dg.CURRENT_DOC, question.section) is not None, "current policy missing"
    assert _rank(chunks, dg.STALE_DOC, question.section) is not None, "stale policy missing"


def test_removing_the_stale_document_deletes_all_and_only_its_chunks(control_collection):
    collection, _, result = control_collection
    stale = sum(c.doc_name == dg.STALE_DOC for c in CHUNKS)
    assert result.deleted == stale == 11
    assert collection.count() == len(CHUNKS) - stale
    assert collection.get(where={"doc_name": dg.STALE_DOC})["ids"] == []


@pytest.mark.parametrize("mode", pipeline.MODES)
@pytest.mark.parametrize("question", dg.QUESTIONS, ids=lambda x: x.id)
def test_without_the_stale_document_retrieval_gives_only_the_current_policy(
    mode, question, control_collection, request
):
    collection, embed, _ = control_collection
    chunks = _retrieve(collection, embed, question, mode, request)
    assert _rank(chunks, dg.CURRENT_DOC, question.section) is not None
    assert all(c["metadata"]["doc_name"] != dg.STALE_DOC for c in chunks)


@pytest.mark.llm
@pytest.mark.parametrize("mode", pipeline.MODES)
@pytest.mark.parametrize("question", dg.QUESTIONS, ids=lambda x: x.id)
def test_control_the_answer_is_correct_once_the_stale_document_is_gone(
    mode, question, control_collection, request
):
    """The same question through the same code: only the data changed."""
    collection, embed, _ = control_collection
    if mode == "rerank":
        request.getfixturevalue("cross_encoder")
    answer = pipeline.answer(
        question.question, k=TOP_K, mode=mode, collection=collection, embedder=embed
    )["answer"]
    assert dg.classify(answer, question) == "correct", answer


@pytest.mark.llm
@pytest.mark.parametrize("question", dg.QUESTIONS, ids=lambda x: x.id)
def test_with_both_policies_at_least_one_mode_gives_a_flawed_answer(
    question, real_collection, request
):
    """The symptom. Wording varies run to run, so this asks for one flawed answer in three modes."""
    collection, embed = real_collection
    request.getfixturevalue("cross_encoder")
    verdicts = {}
    for mode in pipeline.MODES:
        answer = pipeline.answer(
            question.question, k=TOP_K, mode=mode, collection=collection, embedder=embed
        )["answer"]
        verdicts[mode] = dg.classify(answer, question)
    assert any(v != "correct" for v in verdicts.values()), verdicts


# --- gaps found by mutation checks ---


def test_a_fact_that_sits_in_the_wrong_document_is_reported():
    moved = [
        Chunk(c.id, c.text, dg.CURRENT_DOC, c.section, c.chunk_index)
        if c.doc_name == dg.STALE_DOC
        else c
        for c in CHUNKS
    ]
    problems = dg.verify_conflicts(moved)
    assert any(
        dg.STALE_DOC in p for p in problems
    )  # the stale facts are no longer in the stale doc


def test_a_fact_that_sits_in_the_wrong_section_is_reported():
    shifted = [
        Chunk(c.id, c.text, c.doc_name, "9. Elsewhere", c.chunk_index)
        if c.doc_name == dg.STALE_DOC
        else c
        for c in CHUNKS
    ]
    problems = dg.verify_conflicts(shifted)
    assert len(problems) == len(dg.CONFLICTS)  # every stale fact lost its section


# --- review fixes: a sturdier judge, a guard on its numbers, and the generation question ---


def test_number_words_a_hyphen_and_a_pto_filler_count_as_the_value():
    for text in ("up to five days", "a 5-day cap", "carry over 5 PTO days", "5 unused PTO days"):
        assert dg.classify(text, q("carryover")) == "correct", text
    for text in ("twenty days of PTO", "a 20-day allowance", "20 PTO days a year"):
        assert dg.classify(text, q()) == "correct", text


def test_the_stale_value_is_recognised_in_the_same_phrasings():
    assert dg.classify("15 PTO days per year.", q()) == "outdated"
    assert dg.classify("fifteen days a year", q()) == "outdated"
    assert dg.classify("up to ten days may be carried over", q("carryover")) == "outdated"


def test_a_hedge_in_other_phrasings_still_mixes_both():
    assert (
        dg.classify("It is five days in one policy and ten days in the other.", q("carryover"))
        == "mixes both"
    )


def test_a_value_inside_a_longer_number_or_a_compound_number_is_not_the_value():
    assert dg.classify("You get 120 days.", q()) == "no answer"  # not 20
    assert dg.classify("You get twenty-five days.", q("carryover")) == "no answer"  # not five
    assert dg.classify("You get 25 days.", q("carryover")) == "no answer"  # not 5


def test_ten_business_days_of_notice_is_not_the_ten_day_carryover():
    answer = "You can carry over up to 5 days. Request leave at least 10 business days ahead."
    assert dg.classify(answer, q("carryover")) == "correct"


def test_naming_the_2022_policy_counts_as_mentioning_the_stale_document():
    assert dg.classify("You get 20 days; the 2022 version differs.", q()) == "mixes both"
    assert dg.classify('You get 20 days; the "v1_2022" file differs.', q()) == "mixes both"


def test_the_stale_numbers_appear_nowhere_in_the_current_policy():
    # so an answer that says 15 or 10 days can only be quoting the stale document
    current = " ".join(c.text for c in CHUNKS if c.doc_name == dg.CURRENT_DOC)
    for question in dg.QUESTIONS:
        assert not dg.mentions(current, question.stale_number), question.id


def test_the_current_number_is_stated_in_the_current_policys_matching_section():
    for question in dg.QUESTIONS:
        text = " ".join(
            c.text for c in CHUNKS if c.doc_name == dg.CURRENT_DOC and c.section == question.section
        )
        assert dg.mentions(text, question.current_number), question.id


@pytest.mark.parametrize("question", dg.QUESTIONS, ids=lambda x: x.id)
def test_the_questions_values_match_their_conflict_snippets(question):
    assert str(question.current_number) in question.current_snippet
    assert str(question.stale_number) in question.stale_snippet


def test_versioning_keys_finds_version_status_and_date_fields_only():
    assert dg.versioning_keys(["doc_name", "section", "chunk_index", "content_hash"]) == []
    found = dg.versioning_keys(["doc_name", "effective_date", "Version", "status", "superseded_by"])
    assert found == ["effective_date", "Version", "status", "superseded_by"]


def test_the_generation_prompt_says_nothing_about_conflicts_or_versions():
    """Q: was it generation? The prompt neither asks the model to compare versions nor hides it."""
    prompt = gen.SYSTEM_PROMPT.lower()
    for word in (
        "conflict",
        "disagree",
        "version",
        "outdated",
        "older",
        "newer",
        "latest",
        "current",
    ):
        assert word not in prompt, word


@pytest.mark.parametrize("question", dg.QUESTIONS, ids=lambda x: x.id)
def test_the_llm_is_given_both_contradictory_passages_so_the_input_is_what_is_wrong(
    question, real_collection
):
    collection, embed = real_collection
    chunks = pipeline.retrieve(question.question, collection, embed, k=TOP_K, mode="vector")
    prompt = gen.build_user_prompt(question.question, chunks)
    assert question.current_snippet in prompt and question.stale_snippet in prompt


# --- gaps found by mutation checks ---


def test_the_value_must_be_followed_by_the_word_days_not_a_word_that_starts_with_it():
    assert dg.classify("You get 20 daylight hours.", q()) == "no answer"
    assert dg.classify("You get 20 dayshifts.", q()) == "no answer"


def test_naming_the_v1_policy_counts_as_mentioning_the_stale_document():
    assert dg.classify("You get 20 days; the v1 policy differs.", q()) == "mixes both"
