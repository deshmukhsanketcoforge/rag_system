"""The planted data quality issue (RAG-9): what is wrong, and how to recognise a flawed answer.

The corpus holds two PTO policies, `pto_policy` (effective 2024) and `pto_policy_v1_2022`, that
contradict each other, and nothing in either says which is current. This module records the six
conflicting facts and the two questions that expose them, so tests can check the claims the
diagnosis makes (see docs/diagnosis.md) instead of just asserting them in prose.

`classify` is the judge the RAG-8 keyword check could not be: that check only looks for the
expected value, so an answer that also quotes the outdated value still passes.
"""

import re
from dataclasses import dataclass

from rag.evaluation import contains, normalize

CURRENT_DOC = "pto_policy"
STALE_DOC = "pto_policy_v1_2022"


@dataclass(frozen=True)
class Fact:
    doc: str
    section: str  # the exact section title
    snippet: str  # text that must appear in that section


@dataclass(frozen=True)
class Conflict:
    name: str
    current: Fact
    stale: Fact


CONFLICTS = (
    Conflict(
        "Annual PTO, full-time",
        Fact(CURRENT_DOC, "3. Annual Allowance", "20 days of PTO per leave year"),
        Fact(STALE_DOC, "3. Annual Allowance", "15 days of PTO per year"),
    ),
    Conflict(
        "Carryover limit",
        Fact(CURRENT_DOC, "4. Carryover", "maximum of 5 days"),
        Fact(STALE_DOC, "4. Carryover", "maximum of 10 days"),
    ),
    Conflict(
        "Notice for Form PTO-1",
        Fact(CURRENT_DOC, "5. Requesting PTO", "at least 10 business days in advance"),
        Fact(STALE_DOC, "5. Requesting PTO", "at least 5 business days in advance"),
    ),
    Conflict(
        "New hire eligibility",
        Fact(
            CURRENT_DOC, "3. Annual Allowance", "accrue PTO monthly starting from their first day"
        ),
        Fact(STALE_DOC, "2. Scope and Eligibility", "90-day probation period"),
    ),
    Conflict(
        "Paid company holidays",
        Fact(CURRENT_DOC, "6. Company Holidays and Sick Leave", "10 paid company holidays"),
        Fact(STALE_DOC, "6. Company Holidays", "9 paid company holidays"),
    ),
    Conflict(
        "Sick leave",
        Fact(CURRENT_DOC, "6. Company Holidays and Sick Leave", "8 days of sick leave"),
        Fact(STALE_DOC, "7. Sick Leave", "5 days of sick leave"),
    ),
)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    section: str  # the section both documents answer it in
    current_snippet: str  # as in CONFLICTS
    stale_snippet: str
    current_number: int  # the number of days a correct answer states
    stale_number: int  # the number of days an outdated answer states


QUESTIONS = (
    Question(
        "annual",
        "How many PTO days do employees get per year?",
        "3. Annual Allowance",
        "20 days of PTO per leave year",
        "15 days of PTO per year",
        20,
        15,
    ),
    Question(
        "carryover",
        "How many unused PTO days can I carry over into next year?",
        "4. Carryover",
        "maximum of 5 days",
        "maximum of 10 days",
        5,
        10,
    ),
)

# Words a document would use to say it is no longer current. Neither policy contains any.
STATUS_MARKERS = (
    "superseded",
    "obsolete",
    "deprecated",
    "replaced",
    "no longer",
    "outdated",
    "archived",
    "draft",
    "previous version",
    "old version",
)


_NUMBER_WORDS = {5: "five", 10: "ten", 15: "fifteen", 20: "twenty"}
# Ways an answer can name the stale policy without giving its number.
STALE_NAMES = ("v1", "2022")


def mentions(text: str, number: int) -> bool:
    """Does `text` state `number` days? "20 days", "twenty days", "a 20-day cap", "20 PTO days".

    The number must stand alone: "20" is not found in "120" or "2.20", and "five" is not found in
    "twenty-five". Only PTO, unused and paid may sit between the number and "days", so "10 business
    days of notice" is not "10 days". This is a heuristic judge, not a language model.
    """
    forms = [str(number)] + ([_NUMBER_WORDS[number]] if number in _NUMBER_WORDS else [])
    pattern = (
        r"(?<![a-z0-9.\-])(?:" + "|".join(forms) + r")(?:-| )(?:(?:pto|unused|paid) ){0,2}days?"
        r"(?![a-z0-9])"
    )
    return re.search(pattern, normalize(text)) is not None


def classify(answer: str, question: Question) -> str:
    """ "correct", "mixes both", "outdated" or "no answer".

    Naming the stale policy (pto_policy_v1_2022, the 2022 version) counts as mentioning the
    stale value: an answer that says so is still presenting two policies as live candidates.
    """
    current = mentions(answer, question.current_number)
    stale = mentions(answer, question.stale_number) or any(
        contains(answer, name) for name in STALE_NAMES
    )
    if current and stale:
        return "mixes both"
    if current:
        return "correct"
    if stale:
        return "outdated"
    return "no answer"


def status_markers(text: str) -> list[str]:
    """The markers from STATUS_MARKERS found in `text`, in list order, case-insensitive."""
    lowered = text.lower()
    return [m for m in STATUS_MARKERS if re.search(rf"(?<![a-z]){re.escape(m)}(?![a-z])", lowered)]


def verify_conflicts(chunks) -> list[str]:
    """Problems found checking CONFLICTS against the source chunks; empty when all six hold."""
    problems = []
    for conflict in CONFLICTS:
        for fact in (conflict.current, conflict.stale):
            found = any(
                c.doc_name == fact.doc
                and c.section == fact.section
                and " ".join(fact.snippet.split()) in " ".join(c.text.split())
                for c in chunks
            )
            if not found:
                problems.append(
                    f"{conflict.name}: '{fact.snippet}' not found in {fact.doc} > {fact.section}"
                )
    return problems


_VERSIONING = re.compile(r"version|status|date|effective|supersed", re.IGNORECASE)


def versioning_keys(keys) -> list[str]:
    """The metadata keys that would let a retriever tell a current document from an old one."""
    return [k for k in keys if _VERSIONING.search(k)]
