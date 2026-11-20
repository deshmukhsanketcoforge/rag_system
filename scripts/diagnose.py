"""Diagnose the planted data quality issue (RAG-9): retrieval or source data?

    python scripts/diagnose.py                     # all modes, with answers (about 12 LLM calls)
    python scripts/diagnose.py --no-answers        # retrieval evidence only, no LLM
    python scripts/diagnose.py --modes rerank

The two-question approach:
  Q1 Was it retrieval?   Both policies' matching section is retrieved: the retriever did its job.
  Q2 Was it the source?  The two documents contradict each other; nothing says which is current.
Then a control test: remove the stale document and ask again. Same code, same questions.

Everything runs on a temporary store built from data/raw, so neither the corpus nor your own
chroma_db is touched and there is nothing to restore afterwards.
"""

import argparse
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import anthropic  # noqa: E402
import chromadb  # noqa: E402

from rag import diagnosis as dg  # noqa: E402
from rag import pipeline, store  # noqa: E402
from rag import rerank as rr  # noqa: E402
from rag.chunking import Chunk, chunk_corpus  # noqa: E402
from rag.config import INGEST_FINGERPRINT, RAW_DIR, TOP_K  # noqa: E402
from rag.llm import llm_client, require_settings  # noqa: E402


@dataclass(frozen=True)
class Run:
    question: dg.Question
    mode: str
    chunks: list[dict]
    current_rank: int | None  # rank of the current policy's matching section in the top k
    stale_rank: int | None  # rank of the stale policy's matching section
    answer: str | None
    verdict: str | None


@dataclass(frozen=True)
class Diagnosis:
    full: list[Run]
    control: list[Run]
    modes: list[str]
    k: int
    removed: int  # chunks deleted when the stale document was taken out
    unchanged: int  # chunks the removal left exactly as they were
    metadata_keys: list[str]  # every key chunk metadata has (sorted)
    problems: list[str]  # conflicts NOT found in the source (empty = all six are really there)
    markers: list[str]  # status markers found in the stale document (empty = nothing marks it)


def _rank(chunks: list[dict], doc: str, section: str) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        if meta["doc_name"] == doc and meta["section"] == section:
            return rank
    return None


def _phase(collection, questions, modes, k, embedder, llm, reranker, answers) -> list[Run]:
    runs = []
    for q in questions:
        for mode in modes:
            chunks = pipeline.retrieve(
                q.question, collection, embedder, k=k, mode=mode,
                reranker=reranker if mode == "rerank" else None,
            )  # fmt: skip
            answer = (
                pipeline.answer_from_chunks(q.question, chunks, llm=llm)["answer"]
                if answers
                else None
            )
            runs.append(
                Run(
                    q,
                    mode,
                    chunks,
                    _rank(chunks, dg.CURRENT_DOC, q.section),
                    _rank(chunks, dg.STALE_DOC, q.section),
                    answer,
                    dg.classify(answer, q) if answers else None,
                )  # fmt: skip
            )
    return runs


def run(
    chunks: list[Chunk],
    questions,
    modes: list[str],
    k: int,
    answers: bool,
    client: chromadb.ClientAPI,
    embedder=None,
    llm=None,
    reranker=None,
) -> Diagnosis:
    """Ask each question in each mode on the full corpus, then with the stale document removed."""
    for mode in modes:
        if mode not in pipeline.MODES:
            raise ValueError(f"unknown mode '{mode}', expected one of {pipeline.MODES}")
    if not any(c.doc_name == dg.STALE_DOC for c in chunks):
        raise ValueError(
            f"the corpus has no '{dg.STALE_DOC}' chunks, so there is nothing to remove"
        )
    if answers and llm is None:
        require_settings()
        llm = llm_client()  # one client for the whole run
    if embedder is None:
        from rag.embeddings import embed as embedder
    if "rerank" in modes and reranker is None:
        reranker = rr.load_scorer()

    collection, _ = store.sync(chunks, embedder, INGEST_FINGERPRINT, client=client)
    args = (questions, modes, k, embedder, llm, reranker, answers)
    full = _phase(collection, *args)

    kept = [c for c in chunks if c.doc_name != dg.STALE_DOC]
    stale_count = len(chunks) - len(kept)
    collection, result = store.sync(kept, embedder, INGEST_FINGERPRINT, client=client)
    if result.deleted != stale_count or result.added or result.updated:
        raise RuntimeError(
            f"the control is invalid: removing {dg.STALE_DOC} should delete its {stale_count} "
            f"chunks and change nothing else, but the sync deleted {result.deleted}, "
            f"added {result.added} and updated {result.updated}"
        )
    control = _phase(collection, *args)

    stale_text = " ".join(c.text for c in chunks if c.doc_name == dg.STALE_DOC)
    return Diagnosis(
        full,
        control,
        modes,
        k,
        result.deleted,
        result.unchanged,
        sorted({key for c in chunks for key in c.metadata}),
        dg.verify_conflicts(chunks),
        dg.status_markers(stale_text),
    )


def _wrap(text: str, indent: str = "    ") -> str:
    return "\n".join(
        textwrap.fill(line, 96, initial_indent=indent, subsequent_indent=indent) if line else ""
        for line in text.strip().splitlines()
    )


def _rank_text(rank: int | None) -> str:
    return "not retrieved" if rank is None else f"rank {rank}"


def format_report(d: Diagnosis) -> str:
    out = ["# Diagnosis: the planted data quality issue", ""]

    out += ["## 1. Flawed answers (full corpus: both PTO policies ingested)", ""]
    for r in d.full:
        out += [f"Question: {r.question.question}", f"Mode: {r.mode}"]
        if r.answer is None:
            out += ["(answers not requested)", ""]
            continue
        out += [f"Verdict: {r.verdict}", "Answer:", _wrap(r.answer), ""]

    out += ["## 2. Was it retrieval?", ""]
    out += ["| question | mode | current policy | stale policy |", "|---|---|---|---|"]
    for r in d.full:
        cells = [r.question.id, r.mode, _rank_text(r.current_rank), _rank_text(r.stale_rank)]
        out.append("| " + " | ".join(cells) + " |")
    found = sum(r.current_rank is not None and r.stale_rank is not None for r in d.full)
    total, where = len(d.full), f"in the top {d.k} in"
    runs = f"{found} of {total} runs ({', '.join(d.modes)})"
    out += [
        "",
        f"The matching section of BOTH policies was {where} {runs}: "
        + (
            "retrieval returned what the corpus contains, so it did its job."
            if found == total
            else "look at retrieval first."
        ),
        "",
    ]

    out += ["## 3. Was it the source data?", ""]
    out += ["| fact | current policy | stale policy |", "|---|---|---|"]
    for c in dg.CONFLICTS:
        out.append(f"| {c.name} | {c.current.snippet} | {c.stale.snippet} |")
    out += [""]
    if d.problems:
        out += ["NOT found in the source:", *[f"- {p}" for p in d.problems]]
    else:
        out += ["Every row above was checked against the source text: all found."]
    if d.markers:
        markers = f"{', '.join(d.markers)} (the document itself says it is outdated: weaker case)"
    else:
        markers = "none (nothing says it is outdated)"
    keys = d.metadata_keys
    versioning = dg.versioning_keys(keys)
    if versioning:
        keys_note = f"Version or date keys present: {', '.join(versioning)}."
    else:
        keys_note = "There is no version, status or date among them, so nothing to rank by."
    out += [
        f"Status markers in the stale document: {markers}",
        f"Chunk metadata keys: {', '.join(keys)}. {keys_note}",
        "",
    ]

    out += [f"## 4. Control test: remove {dg.STALE_DOC}", ""]
    out += [
        f"Removing it deleted {d.removed} chunks (all of its chunks); "
        f"the other {d.unchanged} chunks were left as they were (checked from the sync result).",
        "",
    ]
    out += ["| question | mode | full corpus | stale removed |", "|---|---|---|---|"]
    for f, c in zip(d.full, d.control, strict=True):
        out.append(f"| {f.question.id} | {f.mode} | {f.verdict or 'n/a'} | {c.verdict or 'n/a'} |")
    out += [""]
    for c in d.control:
        if c.answer is not None:
            out += [
                f"Question: {c.question.question}",
                f"Mode: {c.mode}",
                f"Verdict: {c.verdict}",
                "Answer:",
                _wrap(c.answer),
                "",
            ]
    return "\n".join(out)


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+", choices=pipeline.MODES, default=list(pipeline.MODES))
    parser.add_argument("-k", type=_positive_int, default=TOP_K)
    parser.add_argument("--no-answers", action="store_true", help="retrieval evidence only")
    args = parser.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            d = run(
                chunk_corpus(RAW_DIR), dg.QUESTIONS, args.modes, args.k, not args.no_answers,
                client=chromadb.PersistentClient(path=tmp),
            )  # fmt: skip
            report = format_report(d)
    except (ValueError, RuntimeError, OSError, anthropic.APIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
