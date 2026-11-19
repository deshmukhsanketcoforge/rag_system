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
| Chunking, full vector store, RAG pipeline, hybrid search, reranking, eval harness, diagnosis, source attribution | Not started |

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

## Tests

```bash
pytest -v                          # local venv
docker compose run --rm app        # container
```

Currently these are smoke tests (imports and config values). Tests for chunking, retrieval recall and answer accuracy arrive with their tickets.

## Continuous integration

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests targeting `main`:

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
├── requirements.txt, pytest.ini, .env.example
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                 # generated policy documents
│   └── generate_docs.py     # LLM document generator
├── docs/planted_issue.md    # the planted data quality issue
├── scripts/01_minimal_loop.py
├── src/rag/config.py
└── tests/test_smoke.py
```

Empty directories `eval/` and `docs/screenshots/` are placeholders for later steps.
