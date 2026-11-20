"""Tests for the output format of scripts/ask.py (screenshot #4 shows this output)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ask.py"
spec = importlib.util.spec_from_file_location("ask", SCRIPT)
ask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ask)

RESULT = {
    "answer": "It is $750.",
    "chunks": [
        {
            "id": "remote::3",
            "text": "Stipend of $750.",
            "metadata": {"doc_name": "remote_work_policy", "section": "4. Equipment"},
            "distance": 0.4312,
        }
    ],
    "sources": [{"doc_name": "remote_work_policy", "section": "4. Equipment", "ranks": [1]}],
    "sources_basis": "cited",
    "invalid_citations": [],
}


def test_output_shows_question_chunks_answer_and_sources_in_order():
    out = ask.format_result("How much?", RESULT, "vector")
    for part in ("How much?", "Stipend of $750.", "0.4312", "It is $750.", "vector"):
        assert part in out
    assert out.index("Question") < out.index("Retrieved chunks") < out.index("Answer")
    assert out.index("Answer") < out.index("Sources")
    assert out.rstrip().endswith("remote_work_policy > 4. Equipment")


def test_negative_or_zero_k_is_rejected_by_the_parser(monkeypatch, capsys):
    import pytest

    for bad in ("0", "-1"):
        with pytest.raises(SystemExit) as exc:
            ask.main(["q", "-k", bad])
        assert exc.value.code == 2
    assert "positive" in capsys.readouterr().err


def test_setup_and_api_errors_become_one_line_messages(monkeypatch, capsys):
    def fail(*a, **kw):
        raise RuntimeError("Foundry is not configured")

    monkeypatch.setattr(ask.pipeline, "answer", fail)
    assert ask.main(["q"]) == 1
    err = capsys.readouterr().err
    assert err.strip() == "error: Foundry is not configured"


def test_api_errors_are_caught_too(monkeypatch, capsys):
    import anthropic
    import httpx

    def fail(*a, **kw):
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))

    monkeypatch.setattr(ask.pipeline, "answer", fail)
    assert ask.main(["q"]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_success_returns_zero_and_prints_the_result(monkeypatch, capsys):
    monkeypatch.setattr(ask.pipeline, "answer", lambda *a, **kw: RESULT)
    assert ask.main(["How much?"]) == 0
    assert "It is $750." in capsys.readouterr().out


def test_empty_sources_are_shown_as_none():
    out = ask.format_result("Who?", {**RESULT, "sources": [], "sources_basis": "none"}, "vector")
    assert out.rstrip().endswith("Sources:\n  none")


def test_a_long_answer_is_wrapped_and_keeps_paragraphs():
    long = ("word " * 60).strip() + "\n\nSecond paragraph."
    out = ask.format_result("q", {**RESULT, "answer": long}, "vector")
    answer = out.split("Answer:\n")[1].split("\n\nSources:")[0]
    assert all(len(line) <= 100 for line in answer.splitlines())
    assert "\n\n  Second paragraph." in answer


def test_hybrid_chunks_show_their_fused_score_and_a_dash_when_there_is_no_distance():
    chunks = [
        {**RESULT["chunks"][0], "score": 0.0328},
        {**RESULT["chunks"][0], "id": "x", "distance": None, "score": 0.0164},
    ]
    out = ask.format_result("Q?", {**RESULT, "chunks": chunks}, "hybrid")
    assert "score=0.0328" in out and "distance=0.4312" in out
    assert "distance=-" in out and "score=0.0164" in out
    assert "None" not in out


def test_the_mode_flag_offers_hybrid():
    assert "hybrid" in ask.pipeline.MODES


def test_reranked_chunks_show_the_rerank_score():
    chunks = [{**RESULT["chunks"][0], "score": 0.0328, "rerank_score": 7.25}]
    out = ask.format_result("Q?", {**RESULT, "chunks": chunks}, "rerank")
    assert "rerank=7.2500" in out


# --- source attribution (RAG-10) ---


def test_cited_sources_show_their_passage_numbers_and_say_they_were_cited():
    out = ask.format_result("q", RESULT, "vector")
    assert "Sources (cited in the answer):" in out
    assert "  - [1] remote_work_policy > 4. Equipment" in out


def test_a_source_behind_several_passages_lists_all_their_numbers():
    sources = [{"doc_name": "d", "section": "S", "ranks": [1, 3]}]
    out = ask.format_result("q", {**RESULT, "sources": sources}, "vector")
    assert "  - [1, 3] d > S" in out


def test_retrieved_sources_say_the_answer_cited_nothing():
    result = {**RESULT, "sources_basis": "retrieved"}
    out = ask.format_result("q", result, "vector")
    assert "the answer cited no passage" in out
    assert "Sources (cited in the answer):" not in out
    assert "  - [1] remote_work_policy > 4. Equipment" in out


def test_invalid_citations_are_shown_as_a_warning():
    out = ask.format_result("q", {**RESULT, "invalid_citations": [7, 9]}, "vector")
    assert "Warning" in out and "7, 9" in out and "not retrieved" in out


def test_no_warning_when_every_citation_is_valid():
    assert "Warning" not in ask.format_result("q", RESULT, "vector")


def test_the_sources_come_after_the_answer():
    out = ask.format_result("q", RESULT, "vector")
    assert out.index("Answer:") < out.index("Sources (cited")
