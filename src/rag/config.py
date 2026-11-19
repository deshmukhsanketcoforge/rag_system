"""Central configuration.

Only values known up front live here. Chunk size, overlap, top-k, RRF and
rerank settings are added by the tickets that decide them.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "claude-sonnet-5-5"

# Overridable so the container can keep the store on a volume (see Dockerfile).
CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", str(ROOT / "chroma_db"))

# --- Ingestion (RAG-4) ---
RAW_DIR = ROOT / "data" / "raw"
COLLECTION_NAME = "policies"
EMBED_BATCH_SIZE = 32

# Chunking. Measured on data/raw: top-level sections are 31-156 words (mean ~90) and each is a
# self-contained topic, so a chunk is one section whenever it fits. The cap is 120 words because
# all-MiniLM-L6-v2 truncates at 256 tokens: ~120 words plus the header line stays well under it.
# Sections over the cap are split on '###' subsection boundaries first; only a single block still
# over the cap is cut into word windows that overlap by ~17% so boundary facts are not lost.
CHUNK_MAX_WORDS = 120
CHUNK_OVERLAP_WORDS = 20

# Everything besides the chunk text that changes a stored vector. Part of every chunk's content
# hash, so changing the model or the chunk settings re-embeds the corpus (see store.sync).
INGEST_FINGERPRINT = f"{EMBEDDING_MODEL}|max={CHUNK_MAX_WORDS}|overlap={CHUNK_OVERLAP_WORDS}"
