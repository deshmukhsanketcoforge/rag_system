"""Skip tests marked `llm` unless an API key is available (from the environment or .env)."""

import os

import pytest
from dotenv import load_dotenv

from rag.config import ROOT

load_dotenv(ROOT / ".env")


def pytest_collection_modifyitems(config, items):
    if os.getenv("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY is not set")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
