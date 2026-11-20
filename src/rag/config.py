"""Central configuration.

Only values known up front live here. Chunk size, overlap, top-k, RRF and
rerank settings are added by the tickets that decide them.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# The Azure AI Foundry deployment name, sent as `model` (see rag/llm.py).
# Override with RAG_LLM_MODEL.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "claude-opus-5-5-fast")

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

# --- Retrieval and generation (RAG-5) ---
# Three chunks is about 300 words of context: enough to cover the two PTO policies plus one more
# section, small enough that the prompt stays focused. Revisit once the eval harness exists.
TOP_K = 3
# Answers are short policy lookups. The cap leaves room for a model that thinks before answering.
MAX_ANSWER_TOKENS = 2000
# Both the direct API and the Foundry deployment reject `temperature` (HTTP 400, checked live), so
# it is not sent and answers may vary slightly from run to run.

# --- Hybrid retrieval (RAG-6) ---
# Reciprocal Rank Fusion adds 1 / (RRF_K + rank) per list a chunk appears in. 60 is the value from
# the original paper: large enough that rank 1 and rank 5 are not wildly different, so a chunk
# near the top of both lists beats one at the very top of only one.
RRF_K = 60
# How many candidates each search (vector, keyword) hands to the fusion step. The corpus is 50
# chunks; 10 is generous and keeps the fused list from being padded with weak keyword matches.
CANDIDATE_POOL = 10
