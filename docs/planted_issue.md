# Planted Data Quality Issue

**Type:** outdated duplicate document (stale version with conflicting facts, no "superseded" marker).

## Affected file

`data/raw/pto_policy_v1_2022.md`, an old (effective 2022-01-01) copy of the PTO policy.
The current policy is `data/raw/pto_policy.md` (effective 2024-01-01).
Both are ingested. Nothing in the old file says it is outdated, and the corpus has no version or effective-date metadata that the retriever uses.

## What changed

`pto_policy_v1_2022.md` was generated from the same topic as the current policy, with the numbers deliberately changed. It keeps the same section layout (`## 3. Annual Allowance`, `## 4. Carryover`), so both documents match the same queries.

## Conflicting facts

| Fact | pto_policy.md (current, 2024) | pto_policy_v1_2022.md (stale, 2022) |
|---|---|---|
| Annual PTO, full-time | **20 days** (Section 3.1) | **15 days** (Section 3.1) |
| Carryover limit | **5 days** (Section 4.1) | **10 days** (Section 4.1) |
| Notice for Form PTO-1 | 10 business days, for absences of 3+ days (Section 5.2) | 5 business days (Section 5.1) |
| New hire eligibility | Accrues from day one (Section 3.2) | Only after a 90-day probation (Section 2.2) |
| Paid company holidays | 10 (Section 6.1) | 9 (Section 6.1) |
| Sick leave | 8 days (Section 6.2) | 5 days (Section 7.1) |

## Questions that should expose it

1. **Primary:** "How many PTO days do employees get per year?" (20 vs 15)
2. "How many unused PTO days can I carry over into next year?" (5 vs 10)

A correct answer is 20 days and 5 days carryover (the current policy). A flawed answer states 15 or 10, mixes both, or hedges between the two.

## Expected diagnosis (used in RAG-9)

- Retrieval is not at fault: chunks from both documents are legitimately relevant and both are retrieved.
- The source data is at fault: the corpus holds two contradictory policies with nothing marking one as current.
- Control test: remove `pto_policy_v1_2022.md`, re-ingest, ask again. The answer becomes 20 days with no code change.
- Fixes: versioning, effective-date metadata, removing stale documents, ownership and lineage.
