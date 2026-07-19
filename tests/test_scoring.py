import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from scoring import retrieval_metrics, score_answer, text_error_rates


def test_numeric_pass_with_formatting():
    s = score_answer("The loan amount is $162,000.00.", "162000", "numeric")
    assert s.passed


def test_numeric_fail():
    s = score_answer("The loan amount is $150,000.", "162000", "numeric")
    assert not s.passed


def test_numeric_tolerance():
    assert score_answer("about 4249.9", "4250", "numeric").passed
    assert not score_answer("about 4000", "4250", "numeric").passed


def test_adversarial_distractor_detection():
    s = score_answer("The corrected loan amount is $150,000.", "162000", "numeric",
                     distractor="150000")
    assert not s.passed
    assert s.matched_distractor

    s = score_answer("The loan amount is $162,000 (a corrected copy claims $150,000 "
                     "but it is not authentic).", "162000", "numeric", distractor="150000")
    assert s.passed
    assert not s.matched_distractor


def test_contains_and_fuzzy():
    assert score_answer("Lender: Ficus Bank, NA", "Ficus Bank", "contains").passed
    assert score_answer("Michael Jones & Mary Stone are the borrowers",
                        "Michael Jones and Mary Stone", "fuzzy").passed
    assert not score_answer("No idea", "Michael Jones and Mary Stone", "fuzzy").passed


def test_retrieval_metrics():
    hit, rr = retrieval_metrics(["a.pdf", "b.pdf", "gold.pdf"], "gold.pdf")
    assert hit and abs(rr - 1 / 3) < 1e-9
    hit, rr = retrieval_metrics(["a.pdf"], "gold.pdf")
    assert not hit and rr == 0.0


def test_text_error_rates():
    r = text_error_rates("Loan Amount $162,000", "Loan Amount $162,000")
    assert r["cer"] == 0.0
    r = text_error_rates("Loan Amount $162,000", "Loan Arnount $l62,OOO")
    assert 0 < r["cer"] < 0.5
