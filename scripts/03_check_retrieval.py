"""Repeat the RAG-3 embed-store-retrieve check at full scale (RAG-4).

Run scripts/02_ingest.py first. Asks a few questions against the real collection, prints the
top-3 chunks with distances, and asserts the expected document appears within a per-question
rank limit.

    python scripts/03_check_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag import store  # noqa: E402
from rag.embeddings import embed  # noqa: E402

K = 3
# (question, expected document, worst acceptable rank of the first chunk from that document).
# Unambiguous questions must put the right document first. The PTO question is allowed rank 3
# because the corpus holds two near-identical PTO policies (the planted issue), and either may
# legitimately come first.
CHECKS = [
    ("How much is the home office equipment stipend?", "remote_work_policy", 1),
    ("How soon must I submit an expense report?", "expense_reimbursement_policy", 1),
    ("How long must my password be?", "security_policy", 1),
    ("How many days do I have to report a security incident?", "security_policy", 1),
    ("How many PTO days do employees get per year?", "pto_policy", 3),
]


def first_rank(hits: list[dict], doc: str) -> int | None:
    return next((r for r, h in enumerate(hits, 1) if h["metadata"]["doc_name"] == doc), None)


def main() -> None:
    collection = store.get_collection()
    print(f"Collection '{collection.name}' holds {collection.count()} chunks\n")
    failures = 0
    for question, expected_doc, max_rank in CHECKS:
        hits = store.query(collection, embed([question])[0], k=K)
        print(f"Q: {question}")
        for rank, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            print(f"  {rank}. {hit['distance']:.4f}  {meta['doc_name']} > {meta['section']}")
        rank = first_rank(hits, expected_doc)
        ok = rank is not None and rank <= max_rank
        print(
            f"  -> {'OK' if ok else 'MISS'}: {expected_doc} at rank {rank} (need <= {max_rank})\n"
        )
        failures += not ok
    assert not failures, f"{failures} question(s) failed"
    print("OK: every question found its document within its allowed rank")


if __name__ == "__main__":
    main()
