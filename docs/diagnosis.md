# Diagnosis: the planted data quality issue

**Verdict: the cause is the source data, not the retrieval logic and not the generation logic.**
The corpus holds two PTO policies that contradict each other, and nothing in either document or in
the index says which one is current. Retrieval returned both, as it should. The model then had
two equally authoritative answers and could only report the conflict. Removing the outdated
document fixed every answer with no change to the retrieval or generation code.

Everything below is reproduced by `python scripts/diagnose.py` (the run quoted here is saved in
`docs/outputs/09_diagnosis_run.txt`, which is not committed) and checked by `tests/test_diagnosis.py`
and `tests/test_diagnose_script.py`.

## 1. The questions and the flawed answers

Two questions expose the issue (see `docs/planted_issue.md`). Each was asked in all three
retrieval modes (vector, hybrid, hybrid + rerank), top 3 chunks, model `claude-opus-5-5-fast`.

| Question | Current policy (`pto_policy`) | Stale policy (`pto_policy_v1_2022`) |
|---|---|---|
| How many PTO days do employees get per year? | 20 days | 15 days |
| How many unused PTO days can I carry over into next year? | 5 days | 10 days |

**All six answers mixed both policies.** None is a clean answer. Two typical ones:

> **Carryover, vector mode.** *"The policy documents give conflicting limits, so I can't say for
> certain which applies to you: **"pto_policy" (Section 4.1):** up to 5 days. **"pto_policy_v1_2022"
> (Section 4.1):** up to 10 days. The "v1_2022" label suggests that version may be older, but the
> documents don't say which one is current. Please check with HR before relying on either limit."*

> **Annual allowance, vector mode.** *"Full-time employees receive **20 days of PTO per leave year**
> ... An older 2022 version of the policy gives 15 days per year. The passages don't say which
> version is in effect, but the 20-day figure comes from the document that isn't labeled with a
> version or date."*

How flawed they are varies by answer:
- **Carryover** in vector and rerank mode gives **no definitive number**. A user asking this gets
  sent to HR.
- **Annual allowance**, and carryover in hybrid mode, lead with the right value (20 and 5 days) but
  still present the outdated value as a live possibility and ask the user to confirm.
- No answer stated only the outdated value. The failure is an unresolved conflict, not a wrong number.

The answers vary slightly between runs (the API rejects `temperature`), so the exact wording will
differ. What the tests rely on is only that the stale value or document is named.

## 2. Q1: was it retrieval?

**No. Retrieval returned what the corpus contains.** For both questions and all three modes, the
matching section of **both** policies is in the top 3:

| Question | Mode | Current policy | Stale policy |
|---|---|---|---|
| annual | vector | rank 1 | rank 2 |
| annual | hybrid | rank 3 | rank 2 |
| annual | rerank | rank 2 | rank 1 |
| carryover | vector | rank 2 | rank 1 |
| carryover | hybrid | rank 2 | rank 1 |
| carryover | rerank | rank 2 | rank 1 |

Vector-mode distances (smaller is closer) show how little separates them:

| Question | Rank | Distance | Chunk |
|---|---|---|---|
| annual | 1 | 0.2924 | `pto_policy` > 3. Annual Allowance |
| annual | 2 | 0.2935 | `pto_policy_v1_2022` > 3. Annual Allowance |
| annual | 3 | 0.3655 | `pto_policy` > 6. Company Holidays and Sick Leave |
| carryover | 1 | 0.2197 | `pto_policy_v1_2022` > 4. Carryover |
| carryover | 2 | 0.2358 | `pto_policy` > 4. Carryover |
| carryover | 3 | 0.3762 | `pto_policy_v1_2022` > 3. Annual Allowance |

Both documents are legitimately relevant to the question: they are two versions of the same
policy with the same title and the same section layout. A retriever that ranked the stale chunk
first (as it does for carryover) is not malfunctioning; the two chunks are near-identical in
meaning. Retrieval was also checked in the eval harness (RAG-8): recall@3 is 92% to 100%, and the
misses there are unrelated questions.

**Was it the generation logic?** No. The system prompt (`generate.SYSTEM_PROMPT`) says only to
answer from the numbered passages, not to guess, and to state specific figures. It says nothing
about conflicts, versions or which document to prefer (a test checks that those words are absent),
so it neither causes the hedging nor hides it. The model received both contradictory passages
(a test checks that both the 20-day and the 15-day text are in the prompt it is given) and
reported what it was given. The prompt and the model are identical in the control test below.

## 3. Q2: was it the source data?

**Yes.** Six facts conflict. Each row was checked against the chunked source text by
`dg.verify_conflicts` (a test fails if a row is not really in the documents):

| Fact | `pto_policy` (effective 2024-01-01) | `pto_policy_v1_2022` (effective 2022-01-01) |
|---|---|---|
| Annual PTO, full-time | 20 days of PTO per leave year | 15 days of PTO per year |
| Carryover limit | maximum of 5 days | maximum of 10 days |
| Notice for Form PTO-1 | at least 10 business days in advance | at least 5 business days in advance |
| New hire eligibility | accrue PTO monthly starting from their first day | 90-day probation period |
| Paid company holidays | 10 paid company holidays | 9 paid company holidays |
| Sick leave | 8 days of sick leave | 5 days of sick leave |

And nothing marks one as current:
- The stale document contains none of the words a document uses to say it is out of date
  (superseded, obsolete, deprecated, replaced, no longer, outdated, archived, draft, previous or old
  version). Tested.
- Chunk metadata is `doc_name`, `section` and `chunk_index`. There is no version, status or date
  field, so neither retrieval nor the prompt can prefer the newer policy.
- Both documents have the same title (`Paid Time Off (PTO) Policy`) and largely the same section
  layout (`3. Annual Allowance`, `4. Carryover`, `5. Requesting PTO` exist in both, so the same
  questions match both). They differ in the numbers, some wording, and the "Effective date" line
  in each header chunk; nothing in the retrieval or the prompt reads that line.
- The file name is the only hint, and the model used it: *"the 'v1_2022' label suggests that
  version may be older"*. That is a guess from a file name, not a fact the system holds, so it
  cannot be relied on.

So no retriever or prompt could pick the right value without information the corpus does not
contain.

## 4. Control test: remove the stale document

Same questions, same retrieval and generation code, same model. The only change: the 11 chunks of
`pto_policy_v1_2022` are removed from the store (through the normal incremental `store.sync`,
which deletes chunks whose document is gone). The script checks from the sync result that exactly
those 11 were deleted and that the other 39 chunks were left as they were, and stops with an error
otherwise, so a control that did not really remove the document cannot be reported as one.

| Question | Mode | Full corpus | Stale document removed |
|---|---|---|---|
| annual | vector | mixes both | **correct** |
| annual | hybrid | mixes both | **correct** |
| annual | rerank | mixes both | **correct** |
| carryover | vector | mixes both | **correct** |
| carryover | hybrid | mixes both | **correct** |
| carryover | rerank | mixes both | **correct** |

Example, carryover, vector mode, stale document removed: *"You can carry over **up to 5 days** of
unused PTO into the next leave year. Any balance above 5 days on December 31 is forfeited."*

This is what proves the root cause is the data: the answers become correct without touching
retrieval or generation.

"Correct" means the answer states the current number of days and mentions neither the stale number
nor the stale policy (`dg.classify`). The judge accepts "20 days", "twenty days", "a 20-day cap" and
"20 PTO days", does not count "120" or "twenty-five" as 20 or 5, does not count "10 business days
of notice" as the 10-day carryover, and treats naming `pto_policy_v1_2022`, "v1" or "2022" as
mentioning the stale policy. It is a heuristic, not a language model, so I read the answers too.

**Repo state.** The diagnosis builds its own temporary store from `data/raw/`, so the corpus
files and your `chroma_db/` are never modified. Nothing needed restoring, and the repository
submitted is the original corpus with both policies.

## 5. Why the evaluation harness did not catch this

RAG-8's harness reported **100% answer accuracy in rerank mode**, including both PTO cases. Those
cases check that the answer *contains* `20 days` or `5 days`, and every flawed answer above does.
A keyword check cannot see an answer that also quotes the outdated value or refuses to choose.

That is itself a data-quality lesson: a green dashboard is not evidence of clean data.
`dg.classify` is the stricter check this diagnosis uses. It is not yet part of the eval gate.

## 6. Fixes (what a real system should do)

1. **Remove or archive the stale document.** The control test shows this alone fixes the answers.
   Keep it somewhere that is not ingested.
2. **Version and date metadata.** Store `effective_date`, `version` and `status` (current,
   superseded) per document as chunk metadata, parsed from the header at ingest. The planted
   document even has `Effective date: 2022-01-01`, which is not used.
3. **Retrieval-time rule.** Filter on `status = current`, or when two chunks share a
   (title, section) prefer the latest effective date, and show the date in the prompt label.
4. **Ingestion gate.** Flag near-duplicate documents at ingest (same title, mostly the same
   sections, conflicting numbers) and require an explicit supersede decision before indexing.
5. **Ownership and lineage.** Every document records an owner and what it replaces. Here the owner
   is People Operations for both, so the owner could have retired v1 when v2 shipped.
6. **Quality checks in evaluation.** Add forbidden-value checks to the eval cases (the answer must
   not mention `15 days`) so a conflict fails the gate.

None of 2 to 6 is implemented in this repository; the lab asks for the diagnosis, and item 1 is
demonstrated by the control test.

## Reproduce

```bash
python scripts/diagnose.py                  # the full report above, about 12 LLM calls
python scripts/diagnose.py --no-answers     # retrieval and source evidence only, no LLM
pytest tests/test_diagnosis.py tests/test_diagnose_script.py -v
```
