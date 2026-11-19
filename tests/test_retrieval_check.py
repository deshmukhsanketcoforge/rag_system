"""Tests for how scripts/03_check_retrieval.py judges a result (pure logic, no model)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "03_check_retrieval.py"
spec = importlib.util.spec_from_file_location("check_retrieval", SCRIPT)
check_retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_retrieval)

Check = check_retrieval.Check
judge = check_retrieval.judge


def hit(doc, section="3. Annual Allowance"):
    return {"metadata": {"doc_name": doc, "section": section}, "distance": 0.3}


PTO = Check(
    "How many PTO days?",
    top1_docs={"pto_policy", "pto_policy_v1_2022"},
    must_appear={"pto_policy", "pto_policy_v1_2022"},
    top1_section="Annual Allowance",
)
SINGLE = Check("Stipend?", {"remote_work_policy"}, {"remote_work_policy"}, "Equipment")


def test_passes_when_everything_matches():
    hits = [hit("pto_policy_v1_2022"), hit("pto_policy"), hit("pto_policy", "6. Holidays")]
    assert judge(hits, PTO) is None


def test_fails_when_one_of_the_required_documents_is_missing():
    hits = [hit("pto_policy"), hit("pto_policy", "6. Holidays"), hit("security_policy")]
    assert "pto_policy_v1_2022" in judge(hits, PTO)


def test_fails_when_the_right_document_is_only_at_rank_three():
    hits = [hit("security_policy"), hit("expense_reimbursement_policy"), hit("pto_policy")]
    assert judge(hits, PTO) is not None


def test_fails_when_the_top_hit_is_the_wrong_section():
    hits = [hit("remote_work_policy", "2. Eligibility"), hit("remote_work_policy", "4. Equipment")]
    assert "section" in judge(hits, SINGLE)


def test_single_document_check_needs_it_first():
    hits = [hit("security_policy", "Equipment"), hit("remote_work_policy", "4. Equipment")]
    assert "rank 1" in judge(hits, SINGLE)


def test_empty_results_fail():
    assert judge([], SINGLE) is not None
