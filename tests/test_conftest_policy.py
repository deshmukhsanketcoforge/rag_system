"""RAG_REQUIRE_MODELS: in CI a model that cannot be loaded fails the run instead of skipping tests.

A skip would leave CI green with the eval and recall tests silently not run. Locally, with no
network and no cached model, skipping is still the friendly behaviour.
"""

import conftest
import pytest


def test_without_the_variable_an_unavailable_model_skips(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_MODELS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="model unavailable"):
        conftest.skip_or_fail("model unavailable: offline")


def test_with_the_variable_an_unavailable_model_fails(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_MODELS", "1")
    with pytest.raises(pytest.fail.Exception, match="model unavailable"):
        conftest.skip_or_fail("model unavailable: offline")


@pytest.mark.parametrize("value", ["0", "", "false", "no"])
def test_only_the_value_one_turns_the_requirement_on(monkeypatch, value):
    monkeypatch.setenv("RAG_REQUIRE_MODELS", value)
    assert conftest.models_required() is False


def test_the_value_one_turns_the_requirement_on(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_MODELS", "1")
    assert conftest.models_required() is True


# --- end to end: the real fixtures, with no model available ---

import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_offline(tmp_path, require: bool):
    """Run one real-corpus recall test with an empty model cache and no network."""
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "empty_hf_cache"),
        "TRANSFORMERS_OFFLINE": "1",
    }
    env.pop("RAG_REQUIRE_MODELS", None)
    if require:
        env["RAG_REQUIRE_MODELS"] = "1"
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
            "tests/test_retrieval.py::test_recall_at_k_on_the_eval_set[vector]",
        ],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )  # fmt: skip


def test_a_missing_model_skips_the_recall_test_when_nothing_requires_it(tmp_path):
    result = run_offline(tmp_path, require=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout
    assert "embedding model unavailable" in result.stdout


def test_a_missing_model_fails_the_recall_test_when_ci_requires_models(tmp_path):
    result = run_offline(tmp_path, require=True)
    assert result.returncode != 0, "CI would have gone green with the recall test not run"
    assert "embedding model unavailable" in result.stdout
    assert "skipped" not in result.stdout.splitlines()[-1]
