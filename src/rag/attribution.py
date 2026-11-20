"""Source attribution (RAG-10): which retrieved passages informed an answer.

The model is asked to cite the passages it used by number, like [1] or [1, 3] (see
generate.SYSTEM_PROMPT). Those numbers are checked against the passages it was actually given:

- a cited passage that exists becomes a source;
- a cited number that is not one of the passages is reported as invalid and is never a source,
  so a made-up citation cannot put a document into the answer's sources;
- if the answer cites nothing valid, the sources are the retrieved chunks and the basis says so
  ("retrieved"), instead of passing them off as cited;
- the fixed "not enough information" answer has no sources ("none"), since nothing supports it.

A source is a (document, section) pair with the passage numbers that point at it.
"""

import re

from rag.generate import NOT_FOUND_ANSWER

_CITATION = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def parse_citations(text: str) -> list[int]:
    """Passage numbers cited as [1] or [1, 3], in order of first appearance, without repeats."""
    seen: dict[int, None] = {}
    for match in _CITATION.finditer(text):
        for number in match.group(1).split(","):
            seen.setdefault(int(number), None)
    return list(seen)


def group_sources(chunks: list[dict], ranks: list[int]) -> list[dict]:
    """Distinct (document, section) pairs for the given passage numbers, in passage order.

    `ranks` are 1-based positions in `chunks`. Each source lists the passage numbers behind it.
    """
    grouped: dict[tuple[str, str], list[int]] = {}
    for rank in sorted(set(ranks)):
        meta = chunks[rank - 1]["metadata"]
        grouped.setdefault((meta["doc_name"], meta["section"]), []).append(rank)
    return [{"doc_name": d, "section": s, "ranks": r} for (d, s), r in grouped.items()]


def _is_fallback(answer: str) -> bool:
    plain = answer.replace("’", "'").strip()
    return plain == NOT_FOUND_ANSWER


def attribute(answer: str, chunks: list[dict]) -> dict:
    """{"sources", "basis", "invalid"} for an answer generated from `chunks`.

    basis is "cited" (the answer's valid citations), "retrieved" (it cited nothing valid, so these
    are the chunks it was given) or "none" (the fallback answer, or nothing was retrieved).
    """
    if not chunks or _is_fallback(answer):
        return {"sources": [], "basis": "none", "invalid": []}
    cited = parse_citations(answer)
    valid = [n for n in cited if 1 <= n <= len(chunks)]
    invalid = [n for n in cited if not 1 <= n <= len(chunks)]
    if valid:
        return {"sources": group_sources(chunks, valid), "basis": "cited", "invalid": invalid}
    everything = list(range(1, len(chunks) + 1))
    return {"sources": group_sources(chunks, everything), "basis": "retrieved", "invalid": invalid}
