"""The README is part of the submission, so it is checked like code: it must cover what the task
asks for, point at files that exist, and quote the same numbers as eval/results.md."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
RESULTS = ROOT / "eval" / "results.md"

# paths a README may mention that are deliberately not in the repository
NOT_COMMITTED = ("docs/outputs",)


def section(heading: str) -> str:
    """The text under a `## heading` up to the next `## ` heading."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", README, re.M | re.S)
    assert match, f"README has no '## {heading}' section"
    return match.group(1)


# --- it covers what the task asks for ---


@pytest.mark.parametrize(
    "heading",
    [
        "Quick start",
        "Results at a glance",
        "Status",
        "Tests",
        "Continuous integration",
        "Configuration",
    ],
)
def test_the_readme_has_the_sections_a_reader_needs(heading):
    section(heading)


@pytest.mark.parametrize(
    "topic",
    [
        "Chunking rationale",
        "Hybrid retrieval",
        "Reranking",
        "Evaluation harness",
        "Source attribution",
        "Diagnosing the planted issue",
    ],
)
def test_every_stage_of_the_pipeline_is_described(topic):
    assert topic in README


def test_every_step_in_the_status_table_is_done():
    rows = [r for r in section("Status").splitlines() if r.startswith("| ") and "---" not in r]
    steps = [r for r in rows if not r.startswith("| Step")]
    assert len(steps) >= 11
    assert all(r.rstrip().endswith("| Done |") for r in steps), [
        r for r in steps if "Done" not in r
    ]


def test_the_readme_does_not_read_like_work_in_progress():
    for stale in ("grows as each step lands", "What exists so far", "Not started", "later ticket"):
        assert stale not in README, stale


# --- how to run it ---


def test_the_quick_start_covers_setup_ingest_ask_eval_and_diagnose():
    text = section("Quick start")
    for command in (
        "pip install -r requirements.txt",
        "scripts/02_ingest.py",
        "scripts/ask.py",
        "scripts/run_eval.py",
        "scripts/diagnose.py",
        "pytest",
    ):
        assert command in text, command


def test_every_script_in_the_repo_is_mentioned_in_the_readme():
    for script in (ROOT / "scripts").glob("*.py"):
        assert f"scripts/{script.name}" in README, script.name


def test_the_ci_section_says_how_the_llm_tests_behave_without_the_key():
    text = section("Continuous integration")
    assert "ANTHROPIC_FOUNDRY_API_KEY" in text
    assert "skipped" in text
    assert "RAG_REQUIRE_MODELS" in text  # a model that cannot be downloaded fails, never skips
    assert (
        "run for real" in text
    )  # with the secrets, the answer-accuracy tests really call the model


# --- it points at things that exist ---


def test_every_relative_link_points_at_a_file_that_exists():
    for target in re.findall(r"\]\(([^)\s]+)\)", README):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#")[0]
        assert (ROOT / path).exists(), f"broken link: {target}"


def test_every_path_the_readme_names_exists():
    pattern = r"`((?:scripts|src/rag|docs|eval|tests|data)/[\w./-]+\.(?:py|md|json))`"
    missing = [
        p
        for p in set(re.findall(pattern, README))
        if not p.startswith(NOT_COMMITTED) and "*" not in p and not (ROOT / p).exists()
    ]
    assert not missing, missing


# --- the numbers match the saved eval run ---


def last_summary_rows() -> list[str]:
    text = RESULTS.read_text()
    last_run = text.split("# Eval run ")[-1]
    summary = last_run.split("## Summary")[1].split("\n## ")[0]
    return [r.strip() for r in summary.splitlines() if r.startswith("| ") and "---" not in r]


def test_the_results_table_quotes_the_latest_saved_eval_run():
    rows = last_summary_rows()
    assert len(rows) == 4  # header + vector, hybrid, rerank
    text = section("Results at a glance")
    for row in rows:
        assert row in text, f"README results are out of date with eval/results.md: {row}"


def test_the_results_section_states_the_diagnosis_and_the_headline_wins():
    text = section("Results at a glance")
    assert "source data" in text
    assert "EX-15" in text and "mileage" in text
    assert "docs/diagnosis.md" in text
