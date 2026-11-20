"""The one place that builds the LLM client: Azure AI Foundry only (SAN-26).

Settings come from the environment (`.env` locally, repository secrets in CI). The key is
never written to any tracked file:

    ANTHROPIC_FOUNDRY_API_KEY    the Foundry key
    ANTHROPIC_FOUNDRY_BASE_URL   the endpoint, e.g. https://<resource>.services.ai.azure.com/anthropic

The model sent with each request is the Foundry deployment name (config.LLM_MODEL).
"""

import os

import anthropic

API_KEY_VAR = "ANTHROPIC_FOUNDRY_API_KEY"
BASE_URL_VAR = "ANTHROPIC_FOUNDRY_BASE_URL"
REQUIRED = (API_KEY_VAR, BASE_URL_VAR)


def missing_settings() -> list[str]:
    """Names of the required variables that are unset or blank."""
    return [name for name in REQUIRED if not os.getenv(name, "").strip()]


def _check_endpoint(url: str) -> None:
    # The project endpoint (.../api/projects/<name>) needs an api-version query parameter and is
    # not the Anthropic-compatible one; calls to it fail with a confusing 400.
    if "/api/projects" in url:
        raise RuntimeError(
            f"{BASE_URL_VAR} is the Foundry project endpoint. Use the Anthropic endpoint instead: "
            "https://<resource>.services.ai.azure.com/anthropic"
        )


def require_settings() -> None:
    missing = missing_settings()
    if missing:
        raise RuntimeError(
            f"Foundry is not configured: set {', '.join(missing)} in .env or the environment"
        )
    _check_endpoint(os.environ[BASE_URL_VAR].strip())


def llm_client() -> anthropic.AnthropicFoundry:
    require_settings()
    return anthropic.AnthropicFoundry(
        api_key=os.environ[API_KEY_VAR].strip(), base_url=os.environ[BASE_URL_VAR].strip()
    )
