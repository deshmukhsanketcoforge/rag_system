"""Chunk, embed and store the corpus in the persistent chromadb collection (RAG-4).

    python scripts/02_ingest.py             # incremental: only new/changed chunks are embedded
    python scripts/02_ingest.py --rebuild   # drop the collection and re-embed everything

Re-running is safe and cheap: unchanged chunks are skipped, so with no edits the embedding model
is never even loaded. Chunks of a removed or edited document are deleted from the index.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag import store  # noqa: E402
from rag.chunking import chunk_corpus  # noqa: E402
from rag.config import CHROMA_PATH, INGEST_FINGERPRINT, RAW_DIR  # noqa: E402


def main() -> None:
    rebuild = "--rebuild" in sys.argv[1:]
    chunks = chunk_corpus(RAW_DIR)
    print(f"Read {len({c.doc_name for c in chunks})} documents, {len(chunks)} chunks")

    # Imported here so a run with nothing to embed never pays for loading the model.
    from rag.embeddings import embed

    collection, res = store.sync(chunks, embed, INGEST_FINGERPRINT, rebuild=rebuild)

    print(f"{'Rebuilt' if rebuild else 'Synced'} {CHROMA_PATH} (collection '{collection.name}')")
    print(
        f"  embedded: {res.added} added, {res.updated} updated | "
        f"deleted: {res.deleted} | skipped (unchanged): {res.unchanged}"
    )
    print("\nChunks per document:")
    for doc, n in sorted(Counter(c.doc_name for c in chunks).items()):
        print(f"  {doc:32s} {n:3d}")
    print(f"  {'TOTAL':32s} {collection.count():3d}")
    assert collection.count() == len(chunks), "stored count differs from chunk count"


if __name__ == "__main__":
    main()
