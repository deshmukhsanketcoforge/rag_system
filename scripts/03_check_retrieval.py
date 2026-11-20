"""Repeat the RAG-3 embed-store-retrieve check at full scale (RAG-4).

Run scripts/02_ingest.py first. Asks a few questions against the real collection, prints the
top-3 chunks with distances, and judges each result (see `judge`).

    python scripts/03_check_retrieval.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rag import store  # noqa: E402
from rag.embeddings import embed  # noqa: E402

K = 3


@dataclass(frozen=True)
class Check:
    question: str
    top1_docs: set[str]  # the first hit must come from one of these documents
    must_appear: set[str]  # every one of these documents must appear somewhere in the top K
    top1_section: str  # the first hit's section must contain this text


# Unambiguous questions need the right document AND section first. The PTO question is about the
# planted issue: the corpus holds two near-identical PTO policies, so either may rank first, but
# both must come back (otherwise the planted-issue setup is broken) and the first hit must be the
# Annual Allowance section.
CHECKS = [
    Check(
        "How much is the home office equipment stipend?",
        {"remote_work_policy"},
        {"remote_work_policy"},
        "Equipment",
    ),
    Check(
        "How soon must I submit an expense report?",
        {"expense_reimbursement_policy"},
        {"expense_reimbursement_policy"},
        "Submission",
    ),
    Check("How long must my password be?", {"security_policy"}, {"security_policy"}, "Account"),
    Check(
        "How many days do I have to report a security incident?",
        {"security_policy"},
        {"security_policy"},
        "Incident",
    ),
    Check(
        "How many PTO days do employees get per year?",
        {"pto_policy", "pto_policy_v1_2022"},
        {"pto_policy", "pto_policy_v1_2022"},
        "Annual Allowance",
    ),
]


def judge(hits: list[dict], check: Check) -> str | None:
    """Return None if the hits satisfy the check, else a short reason it failed."""
    if not hits:
        return "no results"
    top = hits[0]["metadata"]
    if top["doc_name"] not in check.top1_docs:
        return f"rank 1 is {top['doc_name']}, expected one of {sorted(check.top1_docs)}"
    if check.top1_section not in top["section"]:
        return (
            f"top hit section is '{top['section']}', expected it to contain '{check.top1_section}'"
        )
    missing = check.must_appear - {h["metadata"]["doc_name"] for h in hits}
    if missing:
        return f"missing from the top {len(hits)}: {sorted(missing)}"
    return None


def main() -> None:
    collection = store.get_collection()
    print(f"Collection '{collection.name}' holds {collection.count()} chunks\n")
    failures = 0
    for check in CHECKS:
        hits = store.query(collection, embed([check.question])[0], k=K)
        print(f"Q: {check.question}")
        for rank, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            print(f"  {rank}. {hit['distance']:.4f}  {meta['doc_name']} > {meta['section']}")
        reason = judge(hits, check)
        print(f"  -> {'OK' if reason is None else 'MISS: ' + reason}\n")
        failures += reason is not None
    assert not failures, f"{failures} question(s) failed"
    print("OK: every question passed")


if __name__ == "__main__":
    main()
