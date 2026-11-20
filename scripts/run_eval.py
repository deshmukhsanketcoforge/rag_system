"""Run the evaluation harness (RAG-8): retrieval recall and answer accuracy per retrieval mode.

    python scripts/run_eval.py                       # recall@k only, all modes, no LLM
    python scripts/run_eval.py --answers             # also generate answers and check keywords
    python scripts/run_eval.py --modes hybrid rerank --answers
    python scripts/run_eval.py --answers --save      # append the run to eval/results.md

Run scripts/02_ingest.py first. Answers need the Foundry settings in .env (see .env.example).
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import anthropic  # noqa: E402

from rag import evaluation as ev  # noqa: E402
from rag import pipeline  # noqa: E402
from rag.config import EVAL_SET, ROOT, TOP_K  # noqa: E402
from rag.llm import require_settings  # noqa: E402

DEFAULT_MODES = pipeline.MODES
RESULTS = ROOT / "eval" / "results.md"


@dataclass(frozen=True)
class ModeRun:
    mode: str
    retrieval: list[ev.RetrievalResult]
    answers: list[ev.AnswerResult] | None


def run(
    cases: list[ev.Case],
    modes: list[str],
    k: int,
    answers: bool,
    collection=None,
    embedder=None,
    llm=None,
    reranker=None,
) -> list[ModeRun]:
    """Evaluate every mode on the same cases. Arguments left as None default to the real ones."""
    for mode in modes:
        if mode not in pipeline.MODES:
            raise ValueError(f"unknown mode '{mode}', expected one of {pipeline.MODES}")
    if not any(c.expected_doc for c in cases):
        raise ValueError("the test set has no case with an expected_doc, so recall has no target")
    if answers and llm is None:
        require_settings()
    collection, embedder, reranker = pipeline.resolve_defaults(
        collection, embedder, reranker, rerank="rerank" in modes
    )

    runs = []
    for mode in modes:
        scorer = reranker if mode == "rerank" else None
        retrieved: dict[str, list[dict]] = {}

        def search(question, mode=mode, scorer=scorer, retrieved=retrieved):
            # once per question: the chunks graded for recall are the ones the answer is made from
            if question not in retrieved:
                retrieved[question] = pipeline.retrieve(
                    question, collection, embedder, k=k, mode=mode, reranker=scorer
                )
            return retrieved[question]

        def ask(question, search=search):
            return pipeline.answer_from_chunks(question, search(question), llm=llm)["answer"]

        retrieval = ev.evaluate_retrieval(cases, search, k=k)
        graded = ev.evaluate_answers(cases, ask) if answers else None
        runs.append(ModeRun(mode, retrieval, graded))
    return runs


def format_summary(runs: list[ModeRun], k: int) -> str:
    with_answers = all(r.answers is not None for r in runs)
    header = ["mode", f"recall@{k}"] + (["answer accuracy"] if with_answers else [])
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in runs:
        cells = [r.mode, f"{ev.recall(r.retrieval):.0%}"]
        if with_answers:
            cells.append(f"{ev.accuracy(r.answers):.0%}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def format_output(runs: list[ModeRun], k: int) -> str:
    parts = [f"## Summary\n\n{format_summary(runs, k)}"]
    parts += [ev.format_report(r.mode, r.retrieval, r.answers, k) for r in runs]
    return "\n\n".join(parts)


def save(path: Path, output: str, when: str | None = None) -> None:
    """Append a dated run, so the history of runs stays in one file."""
    when = when or datetime.now().strftime("%Y-%m-%d %H:%M")
    with Path(path).open("a") as f:
        f.write(f"# Eval run {when}\n\n{output}\n\n---\n\n")


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("-k", type=_positive_int, default=TOP_K)
    parser.add_argument("--answers", action="store_true", help="also grade generated answers")
    parser.add_argument("--save", action="store_true", help="append the run to eval/results.md")
    parser.add_argument("--set", type=Path, default=EVAL_SET, help="the test set (JSON)")
    args = parser.parse_args(argv)
    try:
        cases = ev.load_cases(args.set)
        runs = run(cases, args.modes, args.k, args.answers)
        output = format_output(runs, args.k)
        print(output)
        if args.save:
            save(RESULTS, output)
    except (ValueError, RuntimeError, OSError, anthropic.APIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
