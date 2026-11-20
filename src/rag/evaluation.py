"""Evaluation harness logic (RAG-8): a fixed question set, retrieval recall and answer accuracy.

Two things are measured, on purpose separately, so a failure points at the right stage:
- retrieval recall@k: is the expected (document, section) among the top-k retrieved chunks? No LLM.
- answer accuracy: does the generated answer contain every expected keyword?

The functions take plain callables (`search(question) -> chunks`, `ask(question) -> answer`), so
the same code runs against any retrieval mode and the tests run it against fakes.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag.config import EVAL_SET

_FIELDS = {
    "id",
    "category",
    "question",
    "expected_doc",
    "expected_section",
    "expected_keywords",
    "note",
}
_REQUIRED = ("id", "category", "question", "expected_keywords")


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    question: str
    expected_doc: str | None  # document name as stored (no .md); None for an unanswerable question
    expected_section: str | None  # the section title exactly, as stored
    expected_keywords: tuple[str, ...]
    note: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "Case":
        unknown = set(raw) - _FIELDS
        if unknown:
            raise ValueError(f"case {raw.get('id')!r}: unknown field(s) {sorted(unknown)}")
        for name in _REQUIRED:
            if name not in raw:
                raise ValueError(f"case {raw.get('id')!r}: missing field '{name}'")
        if not raw["expected_keywords"]:
            raise ValueError(f"case {raw['id']!r}: expected_keywords must not be empty")
        doc, section = raw.get("expected_doc"), raw.get("expected_section")
        if (doc is None) != (section is None):
            raise ValueError(
                f"case {raw['id']!r}: expected_doc and expected_section must be given together"
            )
        return cls(
            id=raw["id"],
            category=raw["category"],
            question=raw["question"],
            expected_doc=doc,
            expected_section=section,
            expected_keywords=tuple(raw["expected_keywords"]),
            note=raw.get("note", ""),
        )


@dataclass(frozen=True)
class RetrievalResult:
    case: Case
    rank: int | None  # 1-based position of the first matching chunk, None if not retrieved

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass(frozen=True)
class AnswerResult:
    case: Case
    answer: str
    missing: list[str]

    @property
    def passed(self) -> bool:
        return not self.missing


def load_cases(path: Path = EVAL_SET) -> list[Case]:
    cases = [Case.from_dict(raw) for raw in json.loads(Path(path).read_text())]
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate case id(s): {duplicates}")
    return cases


# --- matching ---


def retrieval_rank(chunks: list[dict], case: Case) -> int | None:
    """Rank of the first chunk from the expected document and with exactly the expected section."""
    if case.expected_doc is None:
        raise ValueError(f"case {case.id!r} has no expected_doc, so it has no retrieval target")
    for rank, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        if meta["doc_name"] == case.expected_doc and case.expected_section == meta["section"]:
            return rank
    return None


def normalize(text: str) -> str:
    """Lowercase, drop markdown bold, make any run of whitespace (even non-breaking) one space."""
    return re.sub(r"\s+", " ", text.replace("*", "").lower()).strip()


def contains(text: str, keyword: str) -> bool:
    """Case-insensitive, whole-token match: '5 days' is not found in '15 days'."""
    pattern = r"(?<![a-z0-9])" + re.escape(normalize(keyword)) + r"(?![a-z0-9])"
    return re.search(pattern, normalize(text)) is not None


def missing_keywords(answer: str, case: Case) -> list[str]:
    return [k for k in case.expected_keywords if not contains(answer, k)]


# --- running ---


def evaluate_retrieval(
    cases: list[Case], search: Callable[[str], list[dict]], k: int
) -> list[RetrievalResult]:
    """Rank of the expected chunk within the top k, for every case that has one.

    Unanswerable cases (no expected document) are skipped. Chunks beyond k are ignored, so the
    result is recall@k whatever the search returns.
    """
    return [
        RetrievalResult(c, retrieval_rank(search(c.question)[:k], c))
        for c in cases
        if c.expected_doc
    ]


def evaluate_answers(cases: list[Case], ask: Callable[[str], str]) -> list[AnswerResult]:
    results = []
    for c in cases:
        answer = ask(c.question)
        results.append(AnswerResult(c, answer, missing_keywords(answer, c)))
    return results


def _fraction(flags: list[bool]) -> float:
    if not flags:
        raise ValueError("no cases to score")
    return sum(flags) / len(flags)


def recall(results: list[RetrievalResult]) -> float:
    return _fraction([r.hit for r in results])


def accuracy(results: list[AnswerResult]) -> float:
    return _fraction([r.passed for r in results])


# --- report ---


def format_report(
    title: str,
    retrieval: list[RetrievalResult],
    answers: list[AnswerResult] | None,
    k: int,
) -> str:
    """Markdown: one row per case, then the totals."""
    ranks = {r.case.id: r for r in retrieval}
    by_answer = {r.case.id: r for r in answers or []}
    ids = list(dict.fromkeys([*ranks, *by_answer]))
    header = ["id", "question", f"retrieved (rank, top {k})"]
    if answers is not None:
        header += ["answer", "missing"]
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for case_id in ids:
        case = (ranks.get(case_id) or by_answer[case_id]).case
        r = ranks.get(case_id)
        cells = [case_id, case.question]
        if r is None:
            cells.append("n/a")
        else:
            cells.append(f"PASS ({r.rank})" if r.hit else "FAIL (-)")
        if answers is not None:
            a = by_answer[case_id]
            cells += ["PASS" if a.passed else "FAIL", ", ".join(a.missing) or "-"]
        rows.append("| " + " | ".join(cells) + " |")
    lines = [f"## {title}", "", *rows, ""]
    hits = sum(r.hit for r in retrieval)
    lines.append(f"recall@{k}: {recall(retrieval):.0%} ({hits}/{len(retrieval)})")
    if answers is not None:
        passed = sum(a.passed for a in answers)
        lines.append(f"answer accuracy: {accuracy(answers):.0%} ({passed}/{len(answers)})")
    return "\n".join(lines)
