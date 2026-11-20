"""The CI workflow caches the Hugging Face models under a key that follows the model settings."""

from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def test_the_model_cache_key_changes_with_config_py():
    # .dockerignore leaves .github out of the image: inside the container there is nothing to check
    if not WORKFLOW.exists():
        pytest.skip("the workflow file is not in this checkout (e.g. the Docker image)")
    # a hand-edited key goes stale when a model changes: the old cache is restored, never re-saved
    text = WORKFLOW.read_text()
    key = next(line for line in text.splitlines() if line.strip().startswith("key: hf-"))
    assert "hashFiles('src/rag/config.py')" in key
