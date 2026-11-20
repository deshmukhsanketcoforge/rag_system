"""LLM client factory tests. No network: building a client makes no request."""

import anthropic
import pytest

from rag import llm

KEY, URL = "ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_BASE_URL"
ENDPOINT = "https://example-resource.services.ai.azure.com/anthropic"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (KEY, URL, "ANTHROPIC_API_KEY"):  # the old direct key must be ignored
        monkeypatch.delenv(name, raising=False)


def test_missing_settings_lists_both_when_nothing_is_set():
    assert llm.missing_settings() == [KEY, URL]


def test_missing_settings_lists_only_what_is_missing(monkeypatch):
    monkeypatch.setenv(KEY, "k")
    assert llm.missing_settings() == [URL]
    monkeypatch.setenv(URL, ENDPOINT)
    assert llm.missing_settings() == []


def test_blank_values_count_as_missing(monkeypatch):
    monkeypatch.setenv(KEY, "   ")
    monkeypatch.setenv(URL, "")
    assert llm.missing_settings() == [KEY, URL]


def test_the_old_direct_key_does_not_count(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-old")
    assert llm.missing_settings() == [KEY, URL]


def test_error_names_the_missing_variables_and_where_to_set_them(monkeypatch):
    monkeypatch.setenv(KEY, "k")
    with pytest.raises(RuntimeError) as exc:
        llm.require_settings()
    message = str(exc.value)
    assert URL in message and KEY not in message
    assert ".env" in message


def test_client_is_a_foundry_client_pointed_at_the_endpoint(monkeypatch):
    monkeypatch.setenv(KEY, "test-key")
    monkeypatch.setenv(URL, ENDPOINT)
    client = llm.llm_client()
    assert isinstance(client, anthropic.AnthropicFoundry)
    assert str(client.base_url).startswith(ENDPOINT)
    assert client.api_key == "test-key"


def test_client_refuses_to_build_without_settings():
    with pytest.raises(RuntimeError, match=KEY):
        llm.llm_client()


def test_whitespace_around_pasted_values_is_removed(monkeypatch):
    monkeypatch.setenv(KEY, "  test-key\n")
    monkeypatch.setenv(URL, f"  {ENDPOINT}  \n")
    client = llm.llm_client()
    assert client.api_key == "test-key"
    assert str(client.base_url).startswith(ENDPOINT)
    assert " " not in str(client.base_url)


def test_the_foundry_project_endpoint_is_rejected_with_the_right_hint(monkeypatch):
    monkeypatch.setenv(KEY, "k")
    monkeypatch.setenv(URL, "https://r.services.ai.azure.com/api/projects/proj-default")
    with pytest.raises(RuntimeError, match=r"/anthropic"):
        llm.llm_client()


def test_require_settings_also_rejects_the_project_endpoint(monkeypatch):
    """So the pipeline fails before retrieval, not at the LLM call."""
    monkeypatch.setenv(KEY, "k")
    monkeypatch.setenv(URL, "https://r.services.ai.azure.com/api/projects/proj-default")
    with pytest.raises(RuntimeError, match=r"/anthropic"):
        llm.require_settings()
