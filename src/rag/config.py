"""Central configuration.

Only values known up front live here. Chunk size, overlap, top-k, RRF and
rerank settings are added by the tickets that decide them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-sonnet-5-5"

CHROMA_PATH = str(ROOT / "chroma_db")
