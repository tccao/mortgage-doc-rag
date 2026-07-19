"""Typed, deterministic scoring for the eval harness.

Answer scoring is per answer_type (numeric with tolerance, exact, contains,
fuzzy) with explicit distractor detection for adversarial cases. Text-level
scoring (CER/WER) uses jiwer against frozen ground truth. Nothing here calls
a model — given the same strings, scores are identical forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import jiwer

NUMERIC_TOLERANCE = 0.005  # 0.5% relative
FUZZY_THRESHOLD = 0.6

_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


@dataclass
class AnswerScore:
    passed: bool
    reason: str
    matched_distractor: bool = False


def _parse_numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_RE.findall(text):
        try:
            out.append(float(m.replace("$", "").replace(",", "")))
        except ValueError:
            pass
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _numbers_match(a: float, b: float, tol: float = NUMERIC_TOLERANCE) -> bool:
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= tol


def score_answer(
    response: str,
    expected: str,
    answer_type: str,
    distractor: str | None = None,
) -> AnswerScore:
    """PASS/FAIL for one case. Adversarial rule: if the response contains the
    distractor's value and not the expected one, that is an explicit FAIL even
    when fuzzy metrics would look acceptable."""
    matched_distractor = False
    if distractor:
        if answer_type == "numeric":
            d = float(distractor.replace(",", ""))
            nums = _parse_numbers(response)
            expected_f = float(expected.replace(",", ""))
            matched_distractor = any(
                _numbers_match(n, d) for n in nums
            ) and not any(_numbers_match(n, expected_f) for n in nums)
        else:
            matched_distractor = _norm(distractor) in _norm(response)

    if answer_type == "numeric":
        expected_f = float(expected.replace(",", ""))
        nums = _parse_numbers(response)
        hit = any(_numbers_match(n, expected_f) for n in nums)
        if matched_distractor:
            return AnswerScore(False, "answered with distractor value", True)
        return AnswerScore(hit, "expected number found" if hit else "expected number absent")

    if answer_type == "exact":
        hit = _norm(response) == _norm(expected)
        return AnswerScore(hit, "exact match" if hit else "no exact match", matched_distractor)

    if answer_type == "contains":
        hit = _norm(expected) in _norm(response)
        if matched_distractor and not hit:
            return AnswerScore(False, "answered with distractor value", True)
        return AnswerScore(hit, "expected string present" if hit else "expected string absent",
                           matched_distractor)

    if answer_type == "fuzzy":
        ratio = SequenceMatcher(None, _norm(response), _norm(expected)).ratio()
        contained = _norm(expected) in _norm(response)
        hit = contained or ratio >= FUZZY_THRESHOLD
        return AnswerScore(hit, f"similarity {ratio:.2f}", matched_distractor)

    raise ValueError(f"Unknown answer_type: {answer_type}")


def retrieval_metrics(retrieved_doc_ids: list[str], gold_doc_id: str) -> tuple[bool, float]:
    """(hit@k, reciprocal rank) — did the gold document appear, and how high."""
    for rank, doc_id in enumerate(retrieved_doc_ids):
        if doc_id == gold_doc_id:
            return True, 1.0 / (rank + 1)
    return False, 0.0


_CER_TRANSFORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def text_error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    """CER/WER between frozen ground truth and pipeline-extracted text."""
    ref = _CER_TRANSFORM(reference)
    hyp = _CER_TRANSFORM(hypothesis)
    if not ref.strip():
        return {"cer": 0.0, "wer": 0.0}
    return {
        "cer": jiwer.cer(ref, hyp),
        "wer": jiwer.wer(ref, hyp),
    }
