"""scripts/diagnose.py: the two-question diagnosis and the control test, on a tiny fake corpus."""

# ruff: noqa: E501  (the fake corpus below reads best one chunk per line)

import importlib.util
from pathlib import Path

import chromadb
import pytest

from rag import diagnosis as dg
from rag.chunking import Chunk

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose.py"
spec = importlib.util.spec_from_file_location("diagnose", SCRIPT)
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)

CURRENT, STALE = dg.CURRENT_DOC, dg.STALE_DOC

CHUNKS = [
    Chunk(f"{CURRENT}::0", "Full-time employees receive 20 days of PTO per leave year.", CURRENT, "3. Annual Allowance", 0),
    Chunk(f"{CURRENT}::1", "Unused PTO may be carried over: a maximum of 5 days.", CURRENT, "4. Carryover", 1),
    Chunk(f"{STALE}::0", "Full-time employees receive 15 days of PTO per year.", STALE, "3. Annual Allowance", 0),
    Chunk(f"{STALE}::1", "Unused PTO may be carried over: a maximum of 10 days.", STALE, "4. Carryover", 1),
    Chunk("other::0", "The cafeteria serves lunch at noon.", "other", "1. Cafe", 0),
]  # fmt: skip

# distinct distances from the question vector, so the ranking is deterministic
VECTORS = {
    CHUNKS[0].text: [1, 0.05, 0],
    CHUNKS[2].text: [1, 0.3, 0],
    CHUNKS[1].text: [1, 0.1, 0],
    CHUNKS[3].text: [1, 0.4, 0],
    CHUNKS[4].text: [0, 0, 1],
}


def fake_embed(texts):
    return [VECTORS.get(t, [1, 0, 0]) for t in texts]


class FakeLLM:
    """Quotes the old value whenever the old policy is in the prompt, like the real model does."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][0]["content"]
        text = "20 days; carryover is 5 days."
        if STALE in prompt:
            text += f" But {STALE} says 15 days and 10 days."
        block = type("B", (), {"type": "text", "text": text})()
        return type("R", (), {"content": [block], "stop_reason": "end_turn"})()


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path))


def run(client, **kw):
    args = dict(
        chunks=CHUNKS, questions=dg.QUESTIONS, modes=["vector"], k=4, answers=True,
        embedder=fake_embed, llm=FakeLLM(), client=client,
    )  # fmt: skip
    return diagnose.run(**{**args, **kw})


def test_the_full_corpus_phase_retrieves_the_right_section_from_both_policies(client):
    d = run(client)
    assert len(d.full) == len(dg.QUESTIONS)
    for r in d.full:
        assert r.current_rank is not None and r.stale_rank is not None


def test_the_full_corpus_answers_mix_both_values(client):
    assert {r.verdict for r in run(client).full} == {"mixes both"}


def test_the_control_phase_has_no_chunk_from_the_stale_document(client):
    d = run(client)
    for r in d.control:
        assert r.stale_rank is None
        assert all(c["metadata"]["doc_name"] != STALE for c in r.chunks)
        assert r.current_rank is not None


def test_the_control_answers_are_correct_with_the_same_question_and_code(client):
    d = run(client)
    assert {r.verdict for r in d.control} == {"correct"}
    assert [r.question.id for r in d.control] == [r.question.id for r in d.full]


def test_removing_the_stale_document_deletes_exactly_its_chunks(client):
    d = run(client)
    assert d.removed == sum(c.doc_name == STALE for c in CHUNKS)


def test_without_answers_no_llm_is_called_and_there_are_no_verdicts(client):
    llm = FakeLLM()
    d = run(client, answers=False, llm=llm)
    assert llm.calls == 0
    assert all(r.answer is None and r.verdict is None for r in d.full + d.control)
    assert all(r.current_rank is not None for r in d.full)  # retrieval evidence still there


def test_every_question_is_asked_in_every_mode_in_both_phases(client):
    d = run(client, modes=["vector", "hybrid"])
    assert len(d.full) == len(d.control) == len(dg.QUESTIONS) * 2
    assert {r.mode for r in d.full} == {"vector", "hybrid"}


def test_an_unknown_mode_is_rejected(client):
    with pytest.raises(ValueError, match="mode"):
        run(client, modes=["bm25"])


def test_missing_llm_settings_fail_before_any_work_when_answers_are_wanted(client, monkeypatch):
    def not_configured():
        raise RuntimeError("Foundry is not configured")

    embedded = []
    monkeypatch.setattr(diagnose, "require_settings", not_configured)
    with pytest.raises(RuntimeError, match="Foundry"):
        run(client, llm=None, embedder=lambda t: embedded.append(t) or fake_embed(t))
    assert embedded == []


def test_a_corpus_without_the_stale_document_is_rejected(client):
    with pytest.raises(ValueError, match=STALE):
        run(client, chunks=[c for c in CHUNKS if c.doc_name != STALE])


def test_the_report_has_the_evidence_for_each_step(client):
    out = diagnose.format_report(run(client))
    for part in (
        "Flawed answers",
        "Was it retrieval?",
        "Was it the source data?",
        "Control test",
        dg.QUESTIONS[0].question,
        "mixes both",
        "correct",
    ):
        assert part in out
    assert out.index("Flawed answers") < out.index("Was it retrieval?")
    assert out.index("Was it retrieval?") < out.index("Was it the source data?")
    assert out.index("Was it the source data?") < out.index("Control test")


def test_the_report_lists_both_retrieval_ranks_and_the_removed_chunk_count(client):
    d = run(client)
    out = diagnose.format_report(d)
    assert f"{d.removed} chunks" in out
    assert "rank" in out.lower()


def test_main_prints_an_error_line_when_setup_is_missing(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("the vector store is empty")

    monkeypatch.setattr(diagnose, "run", boom)
    assert diagnose.main([]) == 1
    assert "error: the vector store is empty" in capsys.readouterr().err


# --- gaps found by mutation checks ---


def test_the_rank_is_that_of_the_matching_section_not_just_the_document(client):
    d = run(client)
    carry = next(r for r in d.full if r.question.id == "carryover")
    assert (
        carry.current_rank == 2
    )  # the current policy's annual chunk ranks first and must not count


def test_an_unknown_mode_is_rejected_before_anything_is_embedded(client):
    embedded = []
    with pytest.raises(ValueError, match="mode"):
        run(client, modes=["bm25"], embedder=lambda t: embedded.append(t) or fake_embed(t))
    assert embedded == []


def test_the_report_concludes_that_retrieval_did_its_job_when_both_policies_come_back(client):
    out = diagnose.format_report(run(client))
    assert "retrieval returned what the corpus contains" in out
    assert "NOT retrieved" not in out


def test_the_report_points_at_retrieval_when_a_policy_was_not_retrieved(client):
    out = diagnose.format_report(run(client, k=1))
    assert "look at retrieval first" in out
    assert "retrieval returned what the corpus contains" not in out


# --- review fixes ---

import dataclasses  # noqa: E402


def test_the_report_does_not_say_nothing_marks_it_outdated_when_a_marker_was_found(client):
    d = dataclasses.replace(run(client), markers=["superseded"])
    out = diagnose.format_report(d)
    assert "superseded" in out
    assert "nothing says it is outdated" not in out


def test_the_report_says_nothing_marks_it_outdated_when_no_marker_was_found(client):
    out = diagnose.format_report(run(client))
    assert "nothing says it is outdated" in out


def test_the_report_lists_the_metadata_keys_it_actually_found(client):
    d = run(client)
    assert d.metadata_keys == ["chunk_index", "doc_name", "section"]
    out = diagnose.format_report(d)
    assert "chunk_index, doc_name, section" in out
    assert "no version, status or date" in out


def test_the_report_does_not_claim_there_is_no_version_when_a_version_key_exists(client):
    d = dataclasses.replace(run(client), metadata_keys=["doc_name", "effective_date", "section"])
    out = diagnose.format_report(d)
    assert "effective_date" in out
    assert "no version, status or date" not in out


def test_the_retrieval_conclusion_names_the_modes_and_k_that_were_run(client):
    out = diagnose.format_report(run(client, modes=["vector", "hybrid"], k=4))
    assert "top 4" in out
    assert "4 of 4 runs" in out  # 2 questions x 2 modes
    assert "vector, hybrid" in out


def test_the_retrieval_conclusion_counts_the_runs_that_found_both_policies(client):
    out = diagnose.format_report(run(client, k=1))
    assert "0 of 2 runs" in out


def test_a_control_that_removes_nothing_is_rejected_not_reported(client, monkeypatch):
    real_sync = diagnose.store.sync
    calls = []

    def sync_that_deletes_nothing(chunks, embedder, fingerprint, client=None, rebuild=False):
        calls.append(1)
        collection, result = real_sync(chunks, embedder, fingerprint, client=client)
        if len(calls) == 2:
            result = dataclasses.replace(result, deleted=0)
        return collection, result

    monkeypatch.setattr(diagnose.store, "sync", sync_that_deletes_nothing)
    with pytest.raises(RuntimeError, match="control"):
        run(client)


def test_a_control_that_changed_other_chunks_is_rejected(client, monkeypatch):
    real_sync = diagnose.store.sync
    calls = []

    def sync_that_updates_others(chunks, embedder, fingerprint, client=None, rebuild=False):
        calls.append(1)
        collection, result = real_sync(chunks, embedder, fingerprint, client=client)
        if len(calls) == 2:
            result = dataclasses.replace(result, updated=1)
        return collection, result

    monkeypatch.setattr(diagnose.store, "sync", sync_that_updates_others)
    with pytest.raises(RuntimeError, match="control"):
        run(client)


def test_the_report_says_how_many_other_chunks_were_left_unchanged(client):
    d = run(client)
    assert d.unchanged == sum(c.doc_name != STALE for c in CHUNKS)
    out = diagnose.format_report(d)
    assert f"deleted {d.removed} chunks" in out
    assert f"the other {d.unchanged} chunks were left as they were" in out


def test_one_llm_client_is_built_for_the_whole_run(client, monkeypatch):
    built = []
    shared = FakeLLM()
    monkeypatch.setattr(diagnose, "require_settings", lambda: None)
    monkeypatch.setattr(diagnose, "llm_client", lambda: built.append(1) or shared)
    run(client, llm=None)
    assert built == [1]
    assert shared.calls == len(dg.QUESTIONS) * 2  # one mode, both phases


def test_a_partial_retrieval_result_is_reported_as_such_not_as_success(client):
    # k=3 on the fake corpus: the annual question finds both policies, carryover does not
    out = diagnose.format_report(run(client, k=3))
    assert "1 of 2 runs" in out
    assert "look at retrieval first" in out
    assert "so it did its job" not in out
