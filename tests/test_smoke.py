"""Smoke test so CI has something to run from day one."""


def test_third_party_imports():
    import anthropic  # noqa: F401
    import chromadb  # noqa: F401
    import dotenv  # noqa: F401
    import sentence_transformers  # noqa: F401


def test_config_values():
    from rag import config

    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.LLM_MODEL == "claude-opus-5-5-fast"  # the Foundry deployment name
    assert config.CHROMA_PATH.endswith("chroma_db")
