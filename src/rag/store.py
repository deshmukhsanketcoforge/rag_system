"""Persistent chromadb store with incremental sync.

`sync` makes the collection match the chunks you pass in, doing the least work possible:

- Each stored chunk carries a `content_hash` of (fingerprint, section, text).
- Chunks whose hash is unchanged are skipped, so they are never embedded again.
- New and changed chunks are embedded and upserted.
- Stored chunks that no longer exist (an edited or removed document) are deleted. This is
  what the RAG-9 control test relies on: remove a doc, sync, and its chunks are gone.

The fingerprint identifies everything besides the text that changes a vector: the embedding
model and the chunking settings (see config.INGEST_FINGERPRINT). Change either and every chunk
hash changes, so everything is re-embedded instead of leaving stale vectors behind.
`rebuild=True` drops the collection first, for a clean slate.

Chunk IDs are deterministic ("<doc>::<index>"). Inserting a section in the middle of a document
shifts the later indexes, so those chunks are re-embedded: simple IDs are kept on purpose.
"""

import contextlib
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import chromadb

from rag.chunking import Chunk
from rag.config import CHROMA_PATH, COLLECTION_NAME

Embedder = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class SyncResult:
    added: int
    updated: int
    deleted: int
    unchanged: int


def get_client(path: str = CHROMA_PATH) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=path)


def get_collection(client: chromadb.ClientAPI | None = None):
    client = client or get_client()
    return client.get_collection(COLLECTION_NAME)


def reset_collection(client: chromadb.ClientAPI):
    """Drop the collection if present and create it fresh (cosine distance)."""
    # chroma raises a different error type per version when the collection is missing
    with contextlib.suppress(Exception):
        client.delete_collection(COLLECTION_NAME)
    return _open_collection(client)


def _open_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def content_hash(chunk: Chunk, fingerprint: str) -> str:
    payload = "\x1f".join([fingerprint, chunk.section, chunk.text])
    return hashlib.sha256(payload.encode()).hexdigest()


def sync(
    chunks: list[Chunk],
    embedder: Embedder,
    fingerprint: str,
    client: chromadb.ClientAPI | None = None,
    rebuild: bool = False,
):
    """Bring the collection in line with `chunks`. Returns (collection, SyncResult)."""
    client = client or get_client()
    collection = reset_collection(client) if rebuild else _open_collection(client)

    stored = collection.get(include=["metadatas"])
    old_hashes = {
        i: (m or {}).get("content_hash")
        for i, m in zip(stored["ids"], stored["metadatas"], strict=True)
    }
    new_hashes = {c.id: content_hash(c, fingerprint) for c in chunks}

    stale = [i for i in old_hashes if i not in new_hashes]
    todo = [c for c in chunks if old_hashes.get(c.id) != new_hashes[c.id]]
    added = sum(1 for c in todo if c.id not in old_hashes)

    if stale:
        collection.delete(ids=stale)
    if todo:
        vectors = embedder([c.text for c in todo])
        if len(vectors) != len(todo):
            raise ValueError("the embedder must return one vector per text")
        collection.upsert(
            ids=[c.id for c in todo],
            documents=[c.text for c in todo],
            embeddings=vectors,
            metadatas=[{**c.metadata, "content_hash": new_hashes[c.id]} for c in todo],
        )

    return collection, SyncResult(
        added=added,
        updated=len(todo) - added,
        deleted=len(stale),
        unchanged=len(chunks) - len(todo),
    )


def query(collection, query_embedding: list[float], k: int = 3) -> list[dict]:
    """Top-k chunks for an embedded query: [{id, text, metadata, distance}, ...]."""
    res = collection.query(query_embeddings=[query_embedding], n_results=k)
    return [
        {"id": i, "text": t, "metadata": m, "distance": d}
        for i, t, m, d in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        )
    ]
