# RAG System

An internal Q&A assistant over company policy documents, built from scratch.

Stack: `sentence-transformers` (embeddings, reranking), `chromadb` (vector store), the Anthropic API (answer generation), `pytest`, and GitHub Actions CI.

## Status

| Step | State |
|---|---|
| Setup: environment, dependencies, repo layout, CI | Done |
| Policy documents with a planted data quality issue | Done |
| Minimal embed-store-retrieve loop | Done |
| Container (Dockerfile, docker-compose) | Done |
| Chunking, embedding and persistent vector store at full scale | Done |
| RAG pipeline, hybrid search, reranking, eval harness, diagnosis, source attribution | Not started |

This README grows as each step lands. The full plan is tracked in Linear (project "RAG System").

## Prerequisites

- Python 3.13 (for the local venv option)
- Docker Desktop (for the container option)
- An Anthropic API key. It is only needed to generate the documents and, later, to answer questions. The tests and the minimal loop run without it.

## 1. Configure the API key

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY
```

`.env` is gitignored and excluded from the Docker image. Never commit it.

## 2. Run it: pick one

### Option A: Docker (recommended)

```bash
docker compose build

docker compose run --rm app                                   # run the tests (default command)
docker compose run --rm app python scripts/01_minimal_loop.py # minimal loop
```

- The project folder is mounted into the container, so code edits on your machine take effect immediately. No rebuild is needed unless `requirements.txt` or the `Dockerfile` changes.
- The vector store (`/data/chroma_db`) and the downloaded models (`/cache/huggingface`) live on named volumes, so they survive between runs. The first run downloads the embedding model, which takes a while.
- To wipe those volumes: `docker compose down -v`.
- The image installs CPU-only PyTorch. The default Linux wheel bundles CUDA and is several GB larger.

### Option B: local virtual environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -v
python scripts/01_minimal_loop.py
```

The vector store goes to `./chroma_db/` (gitignored). Set `RAG_CHROMA_PATH` to change that.

## What exists so far

### Source documents

`data/raw/` holds five generated policy documents (500-800 words each, numbered section headings, exact form codes):

| File | Topic |
|---|---|
| `remote_work_policy.md` | Remote work, equipment stipend, core hours |
| `expense_reimbursement_policy.md` | Expense submission, approval limits |
| `security_policy.md` | Passwords, MFA, incident reporting |
| `pto_policy.md` | Current paid time off policy |
| `pto_policy_v1_2022.md` | Older PTO policy, kept in the corpus on purpose |

One data quality issue is planted deliberately for a later diagnosis step. It is described in [docs/planted_issue.md](docs/planted_issue.md).

The documents are committed, and CI never regenerates them. To regenerate (needs the API key):

```bash
python data/generate_docs.py             # all documents
python data/generate_docs.py pto_policy  # one document by name
```

### Minimal embed-store-retrieve loop

`scripts/01_minimal_loop.py` is a small proof that the core pieces work before any pipeline code exists. It embeds two unrelated sentences with `all-MiniLM-L6-v2` (384 dimensions), stores them in an in-memory Chroma collection with cosine distance, queries with a question, and asserts that the relevant sentence ranks first.

Expected output:

```
Embedding model: all-MiniLM-L6-v2
Vector dimension: 384
Collection count after first text: 1
Collection count after second text: 2

Question: How many vacation days do employees receive?
  1. distance=0.2113  Employees get 20 days of paid vacation
  2. distance=0.7436  The office cafeteria serves lunch at noon

OK: the relevant text ranks first.
```

On macOS you may see `Redirects are currently not supported in MacOs` from PyTorch. It is harmless.

### Chunking, embedding and vector store

```bash
python scripts/02_ingest.py            # incremental sync of the corpus into the vector store
python scripts/02_ingest.py --rebuild  # drop the collection and re-embed everything
python scripts/03_check_retrieval.py   # ask a few questions against the real collection
# in the container:
docker compose run --rm app python scripts/02_ingest.py
docker compose run --rm app python scripts/03_check_retrieval.py
```

Code: `src/rag/chunking.py`, `embeddings.py`, `store.py`. All five documents are ingested, including the older PTO policy.

**Incremental ingest.** `store.sync` makes the collection match `data/raw/` with the least work:

- Each stored chunk carries a `content_hash` of (embedding model + chunk settings, section, text).
- Unchanged chunks are skipped and never embedded again. With no edits, the model is not even loaded.
- New and changed chunks are embedded and upserted.
- Chunks that no longer exist (an edited or removed document) are deleted, so removing a document removes it from the index. The control test in the diagnosis step relies on this.
- Changing the embedding model or the chunk size changes every hash, so everything is re-embedded instead of leaving stale vectors. `--rebuild` forces a clean slate.
- IDs stay `<doc_name>::<chunk_index>`. Inserting a section mid-document shifts later indexes, so those chunks are re-embedded.

**Chunking rationale** (values in `src/rag/config.py`):

- **Section-aware.** The documents have numbered `##` sections that are short (31-156 words, mean about 90) and each covers one topic. A chunk is one section whenever it fits, so a fact stays with its context, and the section name gives clean source attribution later.
- **Cap of 120 words.** The embedding model truncates input at 256 tokens. 120 words plus the header line measured at most 197 tokens, so nothing is cut off. Smaller chunks also retrieve more precisely than whole documents (about 650 words).
- **Splitting.** A section over the cap is packed from its `###` subsections and never cut inside one. This splits 5 of the 50 sections.
- **Overlap of 20 words (about 17%).** Applies only when a single block is still over the cap and has to be cut into word windows, so a fact on the boundary appears in both neighbours. No block in the current corpus needs this. The behaviour is covered by a unit test on synthetic text.
- **Header line.** Each chunk starts with `<document title> - <section>`, so it reads on its own and keyword search can match the section name.
- **Metadata:** `doc_name`, `section`, `chunk_index`. IDs are `<doc_name>::<chunk_index>`.

Result: 50 chunks (expense 10, PTO 10, PTO v1 11, remote work 9, security 10).

## Tests

```bash
pytest -v                          # local venv
docker compose run --rm app        # container
```

- `test_smoke.py`: imports and config values.
- `test_chunking.py`: size cap, subsection splitting, overlap, metadata, header chunk, no empty chunks, deterministic IDs.
- `test_store.py`: incremental sync (first ingest, no-op re-run embeds nothing, only the changed chunk is re-embedded, new chunks added, removed chunks deleted, fingerprint change and `--rebuild` re-embed all) and nearest-first query. Uses a fake embedder, so no model download is needed.

Retrieval recall and answer accuracy tests arrive with the evaluation ticket.

## Lint and formatting

Config is in [ruff.toml](ruff.toml): line length 100, rules `E`, `F`, `I` (import order) and `B` (bugbear). Ruff is not in `requirements.txt`. Install it for local use:

```bash
pip install ruff==0.17.6
ruff check .            # lint
ruff check . --fix      # auto-fix what it can
ruff format .           # format
ruff format --check .   # what CI runs
```

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests targeting `main`:

- `lint`: runs `ruff check .` and `ruff format --check .`. It is fast and does not install the heavy dependencies.
- `test`: installs CPU-only PyTorch and the pinned requirements, then runs `pytest -v`.
- `docker`: builds the image and runs `pytest -v` inside it.

## Configuration

Settings live in [src/rag/config.py](src/rag/config.py):

| Setting | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` |
| Reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `claude-sonnet-5-5` |
| Vector store path | `./chroma_db`, or `RAG_CHROMA_PATH` if set (`/data/chroma_db` in the container) |

## Repo layout

```
.
├── Dockerfile, docker-compose.yml, .dockerignore
├── requirements.txt, pytest.ini, ruff.toml, .env.example
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                 # generated policy documents
│   └── generate_docs.py     # LLM document generator
├── docs/planted_issue.md    # the planted data quality issue
├── scripts/
│   ├── 01_minimal_loop.py   # two-text proof
│   ├── 02_ingest.py         # chunk + embed + store the corpus
│   └── 03_check_retrieval.py
├── src/rag/                 # config, chunking, embeddings, store
└── tests/                   # smoke, chunking, store
```

Empty directories `eval/` and `docs/screenshots/` are placeholders for later steps.
