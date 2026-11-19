"""Vector store tests. A fake embedder replaces the model, so nothing is downloaded."""

import chromadb
import pytest

from rag import store
from rag.chunking import Chunk


class FakeEmbedder:
    """Deterministic embedder that records every batch it is asked to embed."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[1.0, float(len(t)), float(sum(map(ord, t)) % 97)] for t in texts]

    @property
    def embedded(self) -> list[str]:
        return [t for batch in self.calls for t in batch]


def _chunks(n, text="text"):
    return [Chunk(f"doc::{i}", f"{text} {i}", "doc", f"{i}. Section", i) for i in range(n)]


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path))


def _sync(chunks, embedder, client, fingerprint="fp1", rebuild=False):
    return store.sync(chunks, embedder, fingerprint, client=client, rebuild=rebuild)


def test_first_sync_stores_everything_with_metadata(client):
    emb = FakeEmbedder()
    col, res = _sync(_chunks(3), emb, client)
    assert col.count() == 3
    assert (res.added, res.updated, res.deleted, res.unchanged) == (3, 0, 0, 0)
    meta = col.get(ids=["doc::1"])["metadatas"][0]
    assert meta["doc_name"] == "doc"
    assert meta["section"] == "1. Section"
    assert meta["chunk_index"] == 1
    assert meta["content_hash"]


def test_resync_unchanged_embeds_nothing_and_adds_no_duplicates(client):
    emb = FakeEmbedder()
    _sync(_chunks(3), emb, client)
    emb.calls.clear()
    col, res = _sync(_chunks(3), emb, client)
    assert emb.calls == []  # no model work at all
    assert col.count() == 3
    assert (res.added, res.updated, res.deleted, res.unchanged) == (0, 0, 0, 3)


def test_changed_chunk_is_the_only_one_re_embedded(client):
    emb = FakeEmbedder()
    _sync(_chunks(3), emb, client)
    emb.calls.clear()
    changed = _chunks(3)
    changed[1] = Chunk("doc::1", "brand new text", "doc", "1. Section", 1)
    col, res = _sync(changed, emb, client)
    assert emb.embedded == ["brand new text"]
    assert (res.added, res.updated, res.deleted, res.unchanged) == (0, 1, 0, 2)
    assert col.get(ids=["doc::1"])["documents"] == ["brand new text"]
    assert col.count() == 3


def test_new_chunk_is_added_and_only_it_is_embedded(client):
    emb = FakeEmbedder()
    _sync(_chunks(2), emb, client)
    emb.calls.clear()
    col, res = _sync(_chunks(3), emb, client)
    assert emb.embedded == ["text 2"]
    assert (res.added, res.updated, res.deleted, res.unchanged) == (1, 0, 0, 2)
    assert col.count() == 3


def test_removed_chunks_are_deleted_from_the_index(client):
    emb = FakeEmbedder()
    _sync(_chunks(3), emb, client)
    emb.calls.clear()
    col, res = _sync(_chunks(2), emb, client)
    assert emb.calls == []
    assert (res.added, res.updated, res.deleted, res.unchanged) == (0, 0, 1, 2)
    assert col.count() == 2
    assert col.get(ids=["doc::2"])["ids"] == []


def test_new_fingerprint_re_embeds_everything(client):
    emb = FakeEmbedder()
    _sync(_chunks(3), emb, client, fingerprint="model-A")
    emb.calls.clear()
    col, res = _sync(_chunks(3), emb, client, fingerprint="model-B")
    assert len(emb.embedded) == 3
    assert (res.updated, res.unchanged) == (3, 0)
    assert col.count() == 3


def test_rebuild_re_embeds_everything_even_if_unchanged(client):
    emb = FakeEmbedder()
    _sync(_chunks(3), emb, client)
    emb.calls.clear()
    col, res = _sync(_chunks(3), emb, client, rebuild=True)
    assert len(emb.embedded) == 3
    assert (res.added, res.unchanged) == (3, 0)
    assert col.count() == 3


def test_mismatched_embedder_output_is_rejected(client):
    with pytest.raises(ValueError, match="one vector per text"):
        _sync(_chunks(2), lambda texts: [[1.0, 0.0, 0.0]], client)


def test_query_returns_nearest_first(client):
    vectors = {"text 0": [1, 0, 0], "text 1": [0, 1, 0], "text 2": [0, 0, 1]}
    col, _ = _sync(_chunks(3), lambda texts: [vectors[t] for t in texts], client)
    hits = store.query(col, [0, 1, 0], k=2)
    assert hits[0]["id"] == "doc::1"
    assert hits[0]["distance"] <= hits[1]["distance"]
    assert hits[0]["metadata"]["section"] == "1. Section"
