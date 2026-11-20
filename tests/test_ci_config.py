"""The CI workflow: what it runs, what it guards, and that it matches the Dockerfile.

These read .github/workflows/ci.yml and the Dockerfile. The image leaves .github out
(.dockerignore), so inside the container there is nothing to check and the tests skip.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DOCKERFILE = ROOT / "Dockerfile"


@pytest.fixture(scope="module")
def ci():
    if not WORKFLOW.exists():
        pytest.skip("the workflow file is not in this checkout (e.g. the Docker image)")
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def ci_text():
    if not WORKFLOW.exists():
        pytest.skip("the workflow file is not in this checkout (e.g. the Docker image)")
    return WORKFLOW.read_text()


def steps_text(job: dict) -> str:
    return " ".join(str(step.get("run", "")) for step in job["steps"])


def torch_pin(text: str) -> str | None:
    match = re.search(r"pip install torch==([\w.+]+)", text)
    return match.group(1) if match else None


# --- what runs ---


def test_the_workflow_has_lint_test_and_docker_jobs(ci):
    assert set(ci["jobs"]) == {"lint", "test", "docker"}


def test_it_runs_on_pushes_to_main_and_on_pull_requests_to_main(ci):
    triggers = ci.get("on") or ci.get(True)  # YAML 1.1 reads a bare `on` as True
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["pull_request"]["branches"] == ["main"]


def test_the_lint_job_checks_both_lint_and_format(ci):
    run = steps_text(ci["jobs"]["lint"])
    assert "ruff check ." in run and "ruff format --check ." in run


def test_the_test_job_runs_the_whole_suite_and_lists_why_anything_was_skipped(ci):
    run = steps_text(ci["jobs"]["test"])
    assert re.search(r"pytest\b.*-rs", run), "pytest -rs prints the reason for every skip"
    assert "-m " not in run and "tests/" not in run, "no subset: the eval tests must run"


def test_the_docker_job_builds_the_image_and_runs_the_whole_suite_in_it(ci):
    run = steps_text(ci["jobs"]["docker"])
    assert "docker build" in run
    assert re.search(r"docker run .*pytest", run)


# --- a green run must mean the eval tests really ran ---


def test_the_test_job_fails_instead_of_skipping_when_a_model_cannot_be_loaded(ci):
    assert str(ci["jobs"]["test"].get("env", {}).get("RAG_REQUIRE_MODELS")) == "1"


def test_the_docker_job_fails_instead_of_skipping_when_a_model_cannot_be_loaded(ci):
    assert "-e RAG_REQUIRE_MODELS=1" in steps_text(ci["jobs"]["docker"])


def test_the_cache_key_changes_with_config_py(ci_text):
    # a hand-edited key goes stale when a model changes: the old cache is restored, never re-saved
    key = next(line for line in ci_text.splitlines() if line.strip().startswith("key: hf-"))
    assert "hashFiles('src/rag/config.py')" in key


# --- secrets and permissions ---


def test_the_token_is_read_only(ci):
    assert ci["permissions"] == {"contents": "read"}


def test_every_job_has_a_timeout_so_a_hung_run_cannot_burn_minutes(ci):
    for name, job in ci["jobs"].items():
        assert 1 <= job.get("timeout-minutes", 0) <= 60, name


def test_the_foundry_secrets_are_used_by_the_test_job_only(ci):
    for name, job in ci["jobs"].items():
        uses = "secrets." in yaml.safe_dump(job)
        assert uses == (name == "test"), name


def test_the_test_job_passes_both_foundry_secrets(ci):
    dumped = yaml.safe_dump(ci["jobs"]["test"])
    assert "secrets.ANTHROPIC_FOUNDRY_API_KEY" in dumped
    assert "secrets.ANTHROPIC_FOUNDRY_BASE_URL" in dumped


def test_no_job_runs_untrusted_pull_request_code_with_secrets(ci_text):
    assert "pull_request_target" not in ci_text


# --- reproducibility ---


def test_torch_is_pinned_in_ci_and_the_dockerfile_to_the_same_version(ci):
    if not DOCKERFILE.exists():
        pytest.skip("no Dockerfile in this checkout")
    in_ci = torch_pin(steps_text(ci["jobs"]["test"]))
    in_docker = torch_pin(DOCKERFILE.read_text())
    assert in_ci and in_docker, "pin torch (CPU wheel) so a new release cannot break CI unseen"
    assert in_ci == in_docker


def test_torch_is_installed_from_the_cpu_index_in_both_places(ci):
    assert "download.pytorch.org/whl/cpu" in steps_text(ci["jobs"]["test"])
    assert "download.pytorch.org/whl/cpu" in DOCKERFILE.read_text()


def test_ci_and_the_dockerfile_use_the_same_python_minor_version(ci):
    setup = [s for s in ci["jobs"]["test"]["steps"] if "setup-python" in str(s.get("uses", ""))]
    ci_version = str(setup[0]["with"]["python-version"])
    from_line = next(
        line for line in DOCKERFILE.read_text().splitlines() if line.startswith("FROM")
    )
    assert f"python:{ci_version}" in from_line


def test_the_direct_dependencies_are_pinned_exactly():
    lines = [x.strip() for x in (ROOT / "requirements.txt").read_text().splitlines() if x.strip()]
    assert lines and all("==" in line for line in lines)
