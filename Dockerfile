FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    RAG_CHROMA_PATH=/data/chroma_db

WORKDIR /app

# CPU-only torch first: the default Linux wheel bundles CUDA and is several GB.
# sentence-transformers then finds torch already installed.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# Dependencies before code, so code edits do not invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

# No secrets here: .env is excluded by .dockerignore and passed at runtime.
COPY . .

CMD ["pytest", "-v"]
