"""Skip tests marked `llm` unless an API key is available (from the environment or .env)."""

import pytest
from dotenv import load_dotenv

from rag import llm
from rag.config import ROOT

load_dotenv(ROOT / ".env")


def pytest_collection_modifyitems(config, items):
    if not llm.missing_settings():
        return
    skip = pytest.mark.skip(reason="Foundry settings are not set (see .env.example)")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
