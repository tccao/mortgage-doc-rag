"""Backfill confusion matrices into the committed baseline without a re-run.

The confusion matrix landed after the 2026-07-20 benchmark, so `latest.json`
carries accuracy and a per-file `mistakes` list but no matrix. Both variants are
exactly reconstructible from what was already recorded:

  clean     recomputed from frozen ground-truth text — deterministic, no OCR, no
            LLM, so it reproduces the recorded run exactly
  degraded  rebuilt from the recorded `mistakes` list plus the label distribution;
            every off-diagonal cell is a logged error and the diagonal is the
            remainder. Re-running it would cost ~1470s of OCR to recover numbers
            already implied by the record.

The script verifies its own arithmetic against the recorded correct/scored counts
and refuses to write if they disagree. Run:

    uv run python scripts/backfill_confusion.py            # check only
    uv run python scripts/backfill_confusion.py --write    # update baseline JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "src"))

from scoring import confusion_matrix, wilson_interval  # noqa: E402

DATA = ROOT / "data"
BASELINE = ROOT / "evals" / "baselines" / "latest.json"

TYPE_MAP = {
    "loan_estimate": "Loan Estimate",
    "closing_disclosure": "Closing Disclosure",
    "settlement_statement": "Settlement Statement",
    "loan_application": "Loan Application",
    "appraisal": "Appraisal",
    "tax_form": "Tax Document",
    "pay_slip": "Pay Slip",
    "resume": "Resume",
    "escrow_notice": None,
    "service_providers": None,
    "va_form": None,
    "ftc_sample": None,
    "circular": None,
}


def manifest_types() -> dict[str, str]:
    types = {}
    with open(DATA / "manifest.csv") as f:
        for row in csv.DictReader(f):
            types[row["filename"]] = row["doc_type"]
    for sub in ("pay_slip", "resume"):
        for p in (DATA / "clean" / sub).glob("*.pdf"):
            types[p.name] = sub
    return types


def ground_truth(pdf: Path) -> str:
    gt = DATA / "ground_truth" / pdf.parent.name / f"{pdf.stem}.txt"
    return gt.read_text() if gt.exists() else ""


def clean_pairs() -> list[tuple[str, str]]:
    """Recompute clean predictions from frozen ground truth (deterministic)."""
    from mortgage_rag.chunking import classify_page_content

    types = manifest_types()
    pairs = []
    for pdf in sorted((DATA / "clean").rglob("*.pdf")):
        expected = TYPE_MAP.get(types.get(pdf.name, pdf.parent.name))
        if expected is None:
            continue
        predicted, _ = classify_page_content(ground_truth(pdf)[:4000])
        pairs.append((expected, predicted))
    return pairs


def degraded_pairs(mistakes: list[dict]) -> list[tuple[str, str]]:
    """Rebuild degraded pairs from the label distribution + recorded errors."""
    types = manifest_types()
    support: Counter = Counter()
    for scan in sorted((DATA / "degraded").rglob("*_scan.pdf")):
        expected = TYPE_MAP.get(types.get(scan.name.replace("_scan.pdf", ".pdf"),
                                          scan.parent.name))
        if expected is not None:
            support[expected] += 1

    errors = [(m["expected"], m["predicted"]) for m in mistakes if m["variant"] == "degraded"]
    pairs = list(errors)
    wrong_by_class = Counter(e for e, _ in errors)
    for label, total in support.items():
        pairs += [(label, label)] * (total - wrong_by_class[label])
    return pairs


def backfill_rag_intervals(baseline: dict) -> None:
    """Add Wilson + cluster-bootstrap intervals to the recorded RAG results.

    These are pure functions of the per-case PASS/FAIL record, so they need no
    re-run. The bootstrap needs a cluster key, which is joined in from the golden
    set by case id — cases sharing a source document are the correlated group.
    """
    from scoring import cluster_bootstrap_ci

    rag = baseline.get("rag")
    if not rag:
        return

    golden = {}
    with open(ROOT / "evals" / "golden_set.jsonl") as f:
        for line in f:
            if line.strip():
                case = json.loads(line)
                golden[case["id"]] = case
    for case in rag["per_case"]:
        case.setdefault("doc_id", golden.get(case["id"], {}).get("doc_id", case["id"]))

    per_case = rag["per_case"]
    n = len(per_case)
    hits = sum(1 for c in per_case if c["retrieval_hit"])
    rag["retrieval"]["hit_at_k_ci"] = wilson_interval(hits, n)
    rag["retrieval"]["mrr_ci"] = cluster_bootstrap_ci(per_case, "rr", "doc_id")

    answered = [c for c in per_case if "answer_pass" in c]
    if not answered:
        return
    ans = rag["answer"]
    passes = sum(1 for c in answered if c["answer_pass"])
    ans["pass_rate_ci"] = wilson_interval(passes, len(answered))
    ans["pass_rate_bootstrap"] = cluster_bootstrap_ci(
        [{**c, "_pass": float(c["answer_pass"])} for c in answered], "_pass", "doc_id"
    )
    for cat, v in ans["by_category"].items():
        cases = [c for c in answered if c["category"] == cat]
        v["pass_rate_ci"] = wilson_interval(
            sum(1 for c in cases if c["answer_pass"]), len(cases)
        )
    adv = [c for c in answered if c["category"] == "adversarial"]
    if adv:
        ans["adversarial_resistance_ci"] = wilson_interval(
            sum(1 for c in adv if not c["matched_distractor"]), len(adv)
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update the baseline JSON in place")
    ap.add_argument("--report", action="store_true",
                    help="re-render evals/report.md from the updated baseline")
    args = ap.parse_args()

    baseline = json.loads(BASELINE.read_text())
    clf = baseline["classification"]
    built = {"clean": clean_pairs(), "degraded": degraded_pairs(clf["mistakes"])}

    ok = True
    for variant, pairs in built.items():
        recorded = clf[variant]
        conf = confusion_matrix(pairs)
        correct = sum(conf["matrix"][lab][lab] for lab in conf["labels"])
        match = (correct == recorded["correct"] and conf["n"] == recorded["scored"])
        ok &= match
        flag = "OK " if match else "MISMATCH"
        print(f"[{flag}] {variant}: reconstructed {correct}/{conf['n']}, "
              f"recorded {recorded['correct']}/{recorded['scored']}, "
              f"macro F1 {conf['macro_f1']:.4f}")
        if not match:
            continue
        sinks = [
            lab for lab, pc in conf["per_class"].items()
            if pc["predicted"] > 0 and pc["tp"] == 0
        ]
        if sinks:
            print(f"          never-correct but predicted: {sinks}")
        if args.write:
            recorded["confusion"] = conf
            recorded["accuracy_ci"] = wilson_interval(
                recorded["correct"], recorded["scored"]
            )

    if not ok:
        print("\nRefusing to write: reconstruction disagrees with recorded counts.")
        return 1

    if args.write or args.report:
        backfill_rag_intervals(baseline)
        bs = baseline.get("rag", {}).get("answer", {}).get("pass_rate_bootstrap")
        if bs:
            print(f"\nAnswer pass rate {bs['mean']:.3f}, cluster bootstrap 95% CI "
                  f"[{bs['ci_low']:.3f}, {bs['ci_high']:.3f}] "
                  f"({bs['n_clusters']} documents, {bs['n_items']} cases)")

    if args.write:
        BASELINE.write_text(json.dumps(baseline, indent=2))
        print(f"Baseline updated: {BASELINE}")
    if args.report:
        from run_eval import write_report

        write_report(baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
