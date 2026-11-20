"""Vector store tests. A fake embedder replaces the model, so nothing is downloaded."""

import chromadb
import pytest

from rag import store
from rag.chunking import Chunk
from rag.config import COLLECTION_NAME


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


# --- review fixes ---


def test_embedder_failure_leaves_the_old_chunks_in_place(client):
    """Embed and upsert happen before stale deletes, so a failure loses nothing."""
    _sync(_chunks(3), FakeEmbedder(), client)

    def exploding(texts):
        raise RuntimeError("model download failed")

    edited = _chunks(2)
    edited[0] = Chunk("doc::0", "edited text", "doc", "0. Section", 0)  # doc::2 is now stale
    with pytest.raises(RuntimeError):
        _sync(edited, exploding, client)

    col = store.get_collection(client)
    assert col.count() == 3
    assert col.get(ids=["doc::2"])["ids"] == ["doc::2"]
    assert col.get(ids=["doc::0"])["documents"] == ["text 0"]


def test_retry_after_a_failed_sync_completes_the_update(client):
    _sync(_chunks(3), FakeEmbedder(), client)
    edited = _chunks(2)
    edited[0] = Chunk("doc::0", "edited text", "doc", "0. Section", 0)
    with pytest.raises(RuntimeError):
        _sync(edited, lambda texts: (_ for _ in ()).throw(RuntimeError("boom")), client)
    col, res = _sync(edited, FakeEmbedder(), client)
    assert (res.updated, res.deleted) == (1, 1)
    assert col.count() == 2


class _LockedClient:
    def delete_collection(self, name):
        raise PermissionError("database is locked")

    def get_or_create_collection(self, *args, **kwargs):
        raise AssertionError("must not carry on after a failed delete")


def test_reset_collection_surfaces_real_delete_errors():
    with pytest.raises(PermissionError):
        store.reset_collection(_LockedClient())


def test_rebuild_on_an_empty_store_works(client):
    col, res = _sync(_chunks(2), FakeEmbedder(), client, rebuild=True)
    assert col.count() == 2 and res.added == 2


def test_sync_rejects_a_collection_with_the_wrong_distance_space(client):
    client.create_collection(COLLECTION_NAME)  # chroma default is l2, not cosine
    with pytest.raises(RuntimeError, match="cosine"):
        _sync(_chunks(2), FakeEmbedder(), client)


def test_rebuild_fixes_a_wrong_distance_space(client):
    client.create_collection(COLLECTION_NAME)
    col, _ = _sync(_chunks(2), FakeEmbedder(), client, rebuild=True)
    assert col.configuration["hnsw"]["space"] == "cosine"


def test_get_collection_rejects_the_wrong_distance_space(client):
    client.create_collection(COLLECTION_NAME)
    with pytest.raises(RuntimeError, match="cosine"):
        store.get_collection(client)


def test_metadata_changes_count_as_changes(client):
    _sync(_chunks(2), FakeEmbedder(), client)
    moved = [
        Chunk("doc::0", "text 0", "doc", "0. Section", 0),
        Chunk("doc::1", "text 1", "doc", "1. Section", 7),  # same id, text, section; new index
    ]
    col, res = _sync(moved, FakeEmbedder(), client)
    assert (res.updated, res.unchanged) == (1, 1)
    assert col.get(ids=["doc::1"])["metadatas"][0]["chunk_index"] == 7


def test_renamed_doc_with_same_id_and_text_is_updated(client):
    _sync([Chunk("x::0", "same", "old_name", "1. S", 0)], FakeEmbedder(), client)
    col, res = _sync([Chunk("x::0", "same", "new_name", "1. S", 0)], FakeEmbedder(), client)
    assert res.updated == 1
    assert col.get(ids=["x::0"])["metadatas"][0]["doc_name"] == "new_name"
