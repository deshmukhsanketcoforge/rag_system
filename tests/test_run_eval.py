"""scripts/run_eval.py: the runner and its output, with a fake embedder, LLM and reranker."""

import importlib.util
from pathlib import Path

import chromadb
import pytest

from rag import evaluation as ev
from rag import store
from rag.chunking import Chunk

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
spec = importlib.util.spec_from_file_location("run_eval", SCRIPT)
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)

VECTORS = {
    "alpha text": [1, 0, 0],
    "beta text": [0, 1, 0],
    "Q alpha": [1, 0, 0],
    "Q beta": [0, 1, 0],
}


def fake_embed(texts):
    return [VECTORS.get(t, [0, 0, 1]) for t in texts]


class FakeLLM:
    """Answers '$750' to everything, and records how many answers it was asked for."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        block = type("B", (), {"type": "text", "text": "It is $750."})()
        return type("R", (), {"content": [block], "stop_reason": "end_turn"})()


@pytest.fixture
def collection(tmp_path):
    chunks = [
        Chunk("a::0", "alpha text", "doc_a", "1. Alpha", 0),
        Chunk("b::0", "beta text", "doc_b", "1. Beta", 0),
    ]
    client = chromadb.PersistentClient(path=str(tmp_path))
    return store.sync(chunks, fake_embed, "fp", client=client)[0]


CASES = [
    ev.Case.from_dict(
        {
            "id": "a",
            "category": "simple fact",
            "question": "Q alpha",
            "expected_doc": "doc_a",
            "expected_section": "1. Alpha",
            "expected_keywords": ["$750"],
        }
    ),
    ev.Case.from_dict(
        {
            "id": "b",
            "category": "simple fact",
            "question": "Q beta",
            "expected_doc": "doc_b",
            "expected_section": "1. Beta",
            "expected_keywords": ["$999"],
        }
    ),
]


def test_retrieval_only_run_scores_each_mode_and_never_calls_the_llm(collection):
    llm = FakeLLM()
    runs = run_eval.run(
        CASES, ["vector", "hybrid"], k=1, answers=False, collection=collection,
        embedder=fake_embed, llm=llm,
    )  # fmt: skip
    assert [r.mode for r in runs] == ["vector", "hybrid"]
    assert all(r.answers is None for r in runs)
    assert ev.recall(runs[0].retrieval) == 1.0
    assert llm.calls == 0


def test_an_answers_run_asks_the_llm_once_per_case_and_mode(collection):
    llm = FakeLLM()
    runs = run_eval.run(
        CASES, ["vector"], k=1, answers=True, collection=collection, embedder=fake_embed, llm=llm
    )
    assert llm.calls == 2
    assert (
        ev.accuracy(runs[0].answers) == 0.5
    )  # "$750" is right for case a, "$999" is missing for b


def test_the_rerank_mode_uses_the_given_reranker(collection):
    scored = []

    def reranker(pairs):
        scored.extend(pairs)
        return [0.0] * len(pairs)

    run_eval.run(
        CASES, ["rerank"], k=1, answers=False, collection=collection, embedder=fake_embed,
        reranker=reranker,
    )  # fmt: skip
    assert scored  # the cross-encoder was consulted


def test_an_unknown_mode_is_rejected(collection):
    with pytest.raises(ValueError, match="mode"):
        run_eval.run(
            CASES, ["bm25"], k=1, answers=False, collection=collection, embedder=fake_embed
        )


def test_the_summary_compares_the_modes_side_by_side(collection):
    runs = run_eval.run(
        CASES, ["vector", "hybrid"], k=1, answers=True, collection=collection,
        embedder=fake_embed, llm=FakeLLM(),
    )  # fmt: skip
    out = run_eval.format_summary(runs, k=1)
    assert "| vector |" in out and "| hybrid |" in out
    assert "recall@1" in out and "accuracy" in out
    assert "100%" in out and "50%" in out


def test_the_summary_without_answers_has_no_accuracy_column(collection):
    runs = run_eval.run(
        CASES, ["vector"], k=1, answers=False, collection=collection, embedder=fake_embed
    )
    assert "accuracy" not in run_eval.format_summary(runs, k=1)


def test_the_full_output_has_the_summary_then_one_section_per_mode(collection):
    runs = run_eval.run(
        CASES, ["vector", "hybrid"], k=1, answers=False, collection=collection, embedder=fake_embed
    )
    out = run_eval.format_output(runs, k=1)
    assert out.index("Summary") < out.index("## vector") < out.index("## hybrid")


def test_saving_appends_a_dated_run_and_keeps_the_earlier_ones(tmp_path):
    path = tmp_path / "results.md"
    run_eval.save(path, "first run output", when="2026-01-01 10:00")
    run_eval.save(path, "second run output", when="2026-01-02 11:00")
    text = path.read_text()
    assert text.index("first run output") < text.index("second run output")
    assert "2026-01-01 10:00" in text and "2026-01-02 11:00" in text
    assert text.count("# Eval run") == 2


def test_the_default_modes_are_all_three_pipeline_modes():
    from rag import pipeline

    assert tuple(run_eval.DEFAULT_MODES) == pipeline.MODES


def test_an_unknown_mode_is_rejected_before_the_llm_settings_are_checked(collection, monkeypatch):
    def not_configured():
        raise RuntimeError("Foundry is not configured")

    monkeypatch.setattr(run_eval, "require_settings", not_configured)
    with pytest.raises(ValueError, match="mode"):
        run_eval.run(CASES, ["bm25"], k=1, answers=True, collection=collection, embedder=fake_embed)


def test_missing_llm_settings_fail_before_any_retrieval_when_answers_are_wanted(
    collection, monkeypatch
):
    def not_configured():
        raise RuntimeError("Foundry is not configured")

    embedded = []
    monkeypatch.setattr(run_eval, "require_settings", not_configured)
    with pytest.raises(RuntimeError, match="Foundry"):
        run_eval.run(
            CASES, ["vector"], k=1, answers=True, collection=collection,
            embedder=lambda t: embedded.append(t) or fake_embed(t),
        )  # fmt: skip
    assert embedded == []


def test_the_reranker_is_only_handed_to_the_rerank_mode(collection):
    # answering in vector mode with a reranker would be rejected by the pipeline
    runs = run_eval.run(
        CASES, ["vector", "rerank"], k=1, answers=True, collection=collection,
        embedder=fake_embed, llm=FakeLLM(), reranker=lambda pairs: [0.0] * len(pairs),
    )  # fmt: skip
    assert [r.mode for r in runs] == ["vector", "rerank"]


def test_with_answers_each_question_is_retrieved_once_per_mode_not_twice(collection):
    question_embeds = []

    def counting_embed(texts):
        question_embeds.extend(t for t in texts if t.startswith("Q "))
        return fake_embed(texts)

    run_eval.run(
        CASES, ["vector", "hybrid"], k=1, answers=True, collection=collection,
        embedder=counting_embed, llm=FakeLLM(),
    )  # fmt: skip
    assert len(question_embeds) == len(CASES) * 2  # 2 cases x 2 modes, not x 4


def test_the_answer_is_generated_from_the_chunks_that_were_graded_for_recall(collection):
    class PromptLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def create(self, **kwargs):
            self.prompts.append(kwargs["messages"][0]["content"])
            return super().create(**kwargs)

    llm = PromptLLM()
    run_eval.run(
        CASES[:1], ["vector"], k=1, answers=True, collection=collection,
        embedder=fake_embed, llm=llm,
    )  # fmt: skip
    assert "alpha text" in llm.prompts[0] and "beta text" not in llm.prompts[0]


def test_a_set_with_no_answerable_case_is_rejected_before_any_work(collection):
    unanswerable = ev.Case.from_dict(
        {"id": "n", "category": "unanswerable", "question": "Q", "expected_keywords": ["x"]}
    )
    for cases in ([], [unanswerable]):
        with pytest.raises(ValueError, match="expected_doc"):
            run_eval.run(
                cases, ["vector"], k=1, answers=False, collection=collection, embedder=fake_embed
            )


# --- main: every failure is a one-line error, not a traceback


def _ok_runs():
    return [run_eval.ModeRun("vector", [ev.RetrievalResult(CASES[0], 1)], None)]


def test_main_reports_a_set_with_nothing_to_score_as_an_error(tmp_path, capsys):
    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    assert run_eval.main(["--set", str(empty)]) == 1
    assert "error:" in capsys.readouterr().err


def test_main_reports_a_failure_while_formatting_as_an_error(monkeypatch, capsys):
    monkeypatch.setattr(run_eval, "run", lambda *a, **k: [run_eval.ModeRun("vector", [], None)])
    assert run_eval.main([]) == 1
    assert "error:" in capsys.readouterr().err


def test_main_reports_a_failed_save_as_an_error_after_printing_the_results(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(run_eval, "run", lambda *a, **k: _ok_runs())
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)  # a directory: the write fails
    assert run_eval.main(["--save"]) == 1
    captured = capsys.readouterr()
    assert "recall@" in captured.out and "error:" in captured.err


def test_main_succeeds_and_saves_when_everything_works(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_eval, "run", lambda *a, **k: _ok_runs())
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path / "results.md")
    assert run_eval.main(["--save"]) == 0
    assert "recall@" in (tmp_path / "results.md").read_text()
