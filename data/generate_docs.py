"""Generate the source policy corpus with an LLM and plant ONE data quality issue.

Run once, review the output, and commit data/raw/*.md. CI must never run this.

    python data/generate_docs.py            # generate everything
    python data/generate_docs.py pto_policy # regenerate a single doc by name

Planted issue (see docs/planted_issue.md): pto_policy_v1_2022.md is an outdated
duplicate of pto_policy.md with conflicting numbers and no "superseded" marker.
"""

import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag.config import LLM_MODEL  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
MIN_WORDS, MAX_WORDS = 500, 800
MAX_ATTEMPTS = 3

COMPANY = "Northwind Analytics, a 400-person software company"

FORMAT_RULES = f"""\
Write an internal company policy document for {COMPANY}.
Rules:
- Output Markdown only, no preamble and no code fences.
- Between {MIN_WORDS + 40} and {MAX_WORDS - 60} words in total (hard limit {MIN_WORDS}-{MAX_WORDS}).
- First line: a '# ' title. Then a line with 'Effective date: <date>' and a line with
  'Owner: <team>'.
- Use numbered section headings exactly like '## 1. Purpose' and '## 4.2 Equipment Stipend'
  (top-level sections '## N. Title', subsections '### N.M Title').
- Include concrete numbers (amounts, day counts, deadlines) and exact reference codes
  (form codes like 'Form EX-14', cross-references like 'Section 7.3').
- Every cross-reference must point to a section that exists in the document.
- Plain, realistic corporate tone. No placeholders like [Company].
"""

# Each entry: file stem -> (topic prompt). Facts are pinned so they are known and checkable.
POLICIES = {
    "remote_work_policy": """\
Topic: Remote Work Policy.
Effective date: 2024-03-01. Owner: People Operations.
Must include: eligibility (after 90 days of employment), core hours 10:00-15:00 local time,
home office equipment stipend of $750 one time (Section 4.2 Equipment Stipend), monthly internet
reimbursement of $50 claimed with Form RW-7, a requirement to be on-site at least 2 days per
week for hybrid roles, VPN use, and an approval process via Form RW-3.""",
    "expense_reimbursement_policy": """\
Topic: Expense Reimbursement Policy.
Effective date: 2024-01-15. Owner: Finance.
Must include: submission deadline of 30 days after the expense, expenses are submitted on
Form EX-14, receipts required for any expense over $25, meal per-diem of $60 per day when
travelling, manager approval up to $1,000 and director approval above $1,000 (Section 5.3 Approval
Limits), the reimbursement timeline of 10 business days, and non-reimbursable items.""",
    "security_policy": """\
Topic: Information Security Policy.
Effective date: 2024-02-01. Owner: Security Team.
Must include: password minimum 14 characters, multi-factor authentication on all company
accounts, mandatory laptop disk encryption, reporting a suspected incident within 4 hours using
Form SEC-2 (Section 6.1 Incident Reporting), annual security training, and handling of
confidential data classifications.""",
    # Current PTO policy. The v1 doc below deliberately conflicts with these numbers.
    "pto_policy": """\
Topic: Paid Time Off (PTO) Policy.
Effective date: 2024-01-01. Owner: People Operations.
Must include exactly these facts: full-time employees receive 20 days of PTO per year;
unused PTO may be carried over up to a maximum of 5 days into the next year; PTO requests
are made with Form PTO-1 at least 10 business days in advance for absences of 3 days or more;
new hires accrue PTO monthly starting from their first day; 10 paid company holidays per year;
sick leave is separate (8 days per year). Include a section '## 3. Annual Allowance' and a
section '## 4. Carryover'.""",
    # PLANTED ISSUE: outdated duplicate of the PTO policy with conflicting facts.
    "pto_policy_v1_2022": """\
Topic: Paid Time Off (PTO) Policy.
Effective date: 2022-01-01. Owner: People Operations.
This is an older version of the company PTO policy. Do NOT mention that it is old, outdated,
replaced or superseded anywhere. Write it as a normal, current-looking policy.
Must include exactly these facts: full-time employees receive 15 days of PTO per year;
unused PTO may be carried over up to a maximum of 10 days into the next year; PTO requests
are made with Form PTO-1 at least 5 business days in advance; new hires become eligible to use
PTO only after a 90-day probation period; 9 paid company holidays per year; sick leave is
separate (5 days per year). Use the same section structure as the current policy, including
'## 3. Annual Allowance' and '## 4. Carryover'.""",
}


def word_count(text: str) -> int:
    return len(text.split())


def generate(client: anthropic.Anthropic, name: str, topic: str) -> str:
    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": f"{FORMAT_RULES}\n{topic}{feedback}"}],
        )
        # The response may start with a thinking block; keep only the text blocks.
        text = "".join(b.text for b in msg.content if b.type == "text").strip() + "\n"
        n = word_count(text)
        print(f"  {name}: attempt {attempt}, {n} words")
        if MIN_WORDS <= n <= MAX_WORDS:
            return text
        feedback = (
            f"\n\nYour previous draft had {n} words. Rewrite it so it is between "
            f"{MIN_WORDS + 40} and {MAX_WORDS - 60} words."
        )
    raise RuntimeError(f"{name}: no draft within {MIN_WORDS}-{MAX_WORDS} words")


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (see .env.example)")
    client = anthropic.Anthropic()

    wanted = sys.argv[1:] or list(POLICIES)
    unknown = [w for w in wanted if w not in POLICIES]
    if unknown:
        sys.exit(f"unknown doc(s): {unknown}; choose from {list(POLICIES)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        print(f"Generating {name} ...")
        text = generate(client, name, POLICIES[name])
        (RAW_DIR / f"{name}.md").write_text(text)
    print("Done. Review each file, then commit data/raw/.")


if __name__ == "__main__":
    main()
