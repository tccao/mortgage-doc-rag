"""Typed, deterministic scoring for the eval harness.

Answer scoring is per answer_type (numeric with tolerance, exact, contains,
fuzzy) with explicit distractor detection for adversarial cases. Text-level
scoring (CER/WER) uses jiwer against frozen ground truth. Nothing here calls
a model — given the same strings, scores are identical forever.

Also holds the aggregate statistics the report needs: a confusion matrix with
per-class precision/recall/F1 (accuracy alone hides which class is misfiring),
Wilson score intervals, and a cluster bootstrap. Golden cases sharing a source
document fail together, so resampling cases would understate the uncertainty —
the bootstrap resamples documents instead.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import jiwer

NUMERIC_TOLERANCE = 0.005  # 0.5% relative
FUZZY_THRESHOLD = 0.6
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 42

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


def citation_hit(
    citations: list[dict], gold_doc_id: str, gold_page: int, page_tolerance: int = 1
) -> bool | None:
    """Did any returned citation actually point at the gold page?

    Answer scoring checks the *value*; this checks the *provenance* behind it. A
    right answer with citations pointing elsewhere is a right answer for the wrong
    reason, and at scale that is the failure a reviewer cannot audit.

    Returns None when the case carries no page-level gold (bundle cases use
    gold_page 0), so unscoreable cases are excluded rather than counted as misses.
    Pages are 1-indexed to match the citation metadata and what a PDF viewer shows.
    """
    if not gold_page or gold_page < 1:
        return None
    gold_name = gold_doc_id.rsplit("/", 1)[-1]
    for c in citations:
        if not str(c.get("filename", "")).endswith(gold_name):
            continue
        start = int(c.get("page_start", 0)) - page_tolerance
        end = int(c.get("page_end", 0)) + page_tolerance
        if start <= gold_page <= end:
            return True
    return False


# ---------------- Aggregate statistics ----------------


def confusion_matrix(pairs: list[tuple[str, str]]) -> dict:
    """Full confusion matrix + per-class precision/recall/F1 from (expected, predicted).

    Accuracy over an imbalanced label set hides *which* class is wrong. A class
    that is never correct but frequently predicted has precision 0 while overall
    accuracy still looks healthy — that is a keyword sink, and only the matrix
    shows it.
    """
    labels = sorted({e for e, _ in pairs} | {p for _, p in pairs})
    matrix = {e: dict.fromkeys(labels, 0) for e in labels}
    for expected, predicted in pairs:
        matrix[expected][predicted] += 1

    per_class = {}
    for lab in labels:
        tp = matrix[lab][lab]
        fn = sum(matrix[lab].values()) - tp
        fp = sum(matrix[e][lab] for e in labels) - tp
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        # Precision is genuinely undefined for a class that was never predicted,
        # and that is reported as None rather than 0. F1 is still 0 there: the
        # class was missed entirely. Keeping F1 None would let a never-predicted
        # class vanish from macro F1 instead of penalizing it.
        if tp == 0:
            f1 = 0.0 if (tp + fp + fn) else None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[lab] = {
            "support": tp + fn,  # how many truly belong to this class
            "predicted": tp + fp,  # how many times it was guessed
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        }

    scored = [lab for lab in labels if per_class[lab]["support"] > 0]
    macro_f1 = (
        sum(per_class[lab]["f1"] or 0.0 for lab in scored) / len(scored) if scored else None
    )
    correct = sum(matrix[lab][lab] for lab in labels)
    total = len(pairs)
    return {
        "labels": labels,
        "matrix": matrix,
        "per_class": per_class,
        "accuracy": round(correct / total, 4) if total else None,
        "macro_f1": round(macro_f1, 4) if macro_f1 is not None else None,
        "n": total,
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion.

    Preferred over the Wald interval because Wald collapses to zero width at
    p=0 or p=1 — it would report 29/29 retrieval hits as [1.0, 1.0], claiming
    certainty from 29 observations.
    """
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def cluster_bootstrap_ci(
    items: list[dict],
    value_key: str,
    cluster_key: str,
    n_boot: int = BOOTSTRAP_ITERS,
    alpha: float = 0.05,
    seed: int = BOOTSTRAP_SEED,
) -> dict | None:
    """Percentile bootstrap CI for a mean, resampling *clusters* not items.

    Golden cases are not independent: three cases target the same refi closing
    disclosure and fail together when that document extracts badly. Resampling
    cases would treat those as three independent observations and understate the
    interval. Resampling documents keeps the correlation intact.

    Seeded, so the interval is reproducible like every other number here.
    """
    usable = [it for it in items if it.get(value_key) is not None]
    if not usable:
        return None

    groups: dict[str, list[float]] = {}
    for it in usable:
        groups.setdefault(str(it.get(cluster_key, it.get("id", ""))), []).append(
            float(it[value_key])
        )
    keys = list(groups)
    observed = sum(sum(v) for v in groups.values()) / sum(len(v) for v in groups.values())

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = count = 0.0
        for _ in keys:
            vals = groups[keys[rng.randrange(len(keys))]]
            total += sum(vals)
            count += len(vals)
        if count:
            means.append(total / count)
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]

    out = {
        "mean": round(observed, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n_items": len(usable),
        "n_clusters": len(keys),
        "iters": n_boot,
    }
    # Between-cluster variance is what this bootstrap resamples. With one cluster
    # there is none, so every resample returns the same mean and the interval
    # collapses to zero width — narrow for the worst possible reason. Report it
    # as unavailable rather than as certainty.
    if len(keys) < 2:
        out.update({"ci_low": None, "ci_high": None,
                    "note": "single cluster — between-cluster variance not estimable"})
    elif len(keys) < 5:
        out["note"] = f"only {len(keys)} clusters — interval is coarse"
    return out


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
