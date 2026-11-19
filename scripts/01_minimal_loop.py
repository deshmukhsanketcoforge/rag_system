"""Minimal embed-store-retrieve loop on two known texts (RAG-3).

Proves embeddings + chromadb + similarity ranking work before any pipeline code.

    python scripts/01_minimal_loop.py
"""
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag.config import EMBEDDING_MODEL  # noqa: E402

RELEVANT = "Employees get 20 days of paid vacation"
UNRELATED = "The office cafeteria serves lunch at noon"
QUESTION = "How many vacation days do employees receive?"


def main() -> None:
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 1. Embed one short text and print the vector dimension.
    vec = model.encode(RELEVANT)
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Vector dimension: {len(vec)}")

    # 2. Store it in an in-memory chromadb collection (cosine distance) and print the count.
    client = chromadb.EphemeralClient()
    collection = client.create_collection("minimal_loop", metadata={"hnsw:space": "cosine"})
    collection.add(ids=["doc-0"], documents=[RELEVANT], embeddings=[vec.tolist()])
    print(f"Collection count after first text: {collection.count()}")

    # 3. Add a second, clearly different text.
    collection.add(
        ids=["doc-1"], documents=[UNRELATED], embeddings=[model.encode(UNRELATED).tolist()]
    )
    print(f"Collection count after second text: {collection.count()}")

    # 4. Query with a question that matches one of them; print results and distances.
    print(f"\nQuestion: {QUESTION}")
    result = collection.query(query_embeddings=[model.encode(QUESTION).tolist()], n_results=2)
    docs, dists = result["documents"][0], result["distances"][0]
    for rank, (doc, dist) in enumerate(zip(docs, dists), start=1):
        print(f"  {rank}. distance={dist:.4f}  {doc}")

    # 5. Assert the relevant text ranks first (smaller distance = more similar).
    assert docs[0] == RELEVANT, f"expected the vacation text first, got: {docs[0]!r}"
    assert dists[0] < dists[1]
    print("\nOK: the relevant text ranks first.")


if __name__ == "__main__":
    main()
