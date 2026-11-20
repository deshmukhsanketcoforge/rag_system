"""Ask the RAG pipeline a question (RAG-5).

    python scripts/ask.py "How much is the home office equipment stipend?"
    python scripts/ask.py "..." -k 5
    python scripts/ask.py "..." --mode vector

Run scripts/02_ingest.py first, and set the Foundry variables in .env (see .env.example).
"""

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import anthropic  # noqa: E402

from rag import pipeline  # noqa: E402
from rag.config import TOP_K  # noqa: E402


def _wrap_answer(text: str, width: int = 96, indent: str = "  ") -> str:
    """Wrap line by line, so paragraph breaks and bullet lists from the model survive."""
    return "\n".join(
        textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent) if line else ""
        for line in text.strip().splitlines()
    )


def format_result(question: str, result: dict, mode: str) -> str:
    lines = [f"Question: {question}", f"Retrieval: {mode}", "", "Retrieved chunks:"]
    for rank, chunk in enumerate(result["chunks"], start=1):
        meta = chunk["metadata"]
        distance = "-" if chunk.get("distance") is None else f"{chunk['distance']:.4f}"
        score = f"  score={chunk['score']:.4f}" if "score" in chunk else ""
        lines.append(
            f"  [{rank}] distance={distance}{score}  {meta['doc_name']} > {meta['section']}"
        )
        lines.append(textwrap.indent(textwrap.fill(chunk["text"], 96), "      "))
    lines += ["", "Answer:", _wrap_answer(result["answer"]), "", "Sources:"]
    lines += [f"  - {s['doc_name']} > {s['section']}" for s in result["sources"]] or ["  none"]
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument(
        "-k", type=_positive_int, default=TOP_K, help=f"chunks to retrieve (default {TOP_K})"
    )
    parser.add_argument("--mode", default="vector", choices=pipeline.MODES)
    args = parser.parse_args(argv)
    try:
        result = pipeline.answer(args.question, k=args.k, mode=args.mode)
    except (ValueError, RuntimeError, anthropic.APIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_result(args.question, result, args.mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
