"""Layered regression eval for the mortgage-doc-rag pipeline.

Layers (each independently runnable, all deterministic except LLM generation,
which is pinned to temperature 0):

  ocr             CER/WER of OCR on degraded scans vs frozen ground truth
  classification  doc-type accuracy over the full corpus (manifest = labels)
  retrieval       hit@k / MRR over the golden set (no LLM)
  answer          PASS/FAIL per golden case via the orchestrator (needs LLM)

Usage:
  uv run python evals/run_eval.py --layers retrieval                # CI-safe
  uv run python evals/run_eval.py --retrieval-only                  # same
  uv run python evals/run_eval.py --all --backend llama_cpp         # full run
  uv run python evals/run_eval.py --all --save-baseline
  uv run python evals/run_eval.py --layers retrieval --compare evals/baselines/latest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from retrievers import BM25Retriever, answer_without_retrieval  # noqa: E402
from scoring import (  # noqa: E402
    citation_hit,
    cluster_bootstrap_ci,
    confusion_matrix,
    retrieval_metrics,
    score_answer,
    text_error_rates,
    wilson_interval,
)

DATA = ROOT / "data"
BASELINES = ROOT / "evals" / "baselines"
REPORT = ROOT / "evals" / "report.md"
GOLDEN = ROOT / "evals" / "golden_set.jsonl"

DEGRADED_MAX_PAGES = 4  # keep in sync with scripts/degrade_scans.py

# manifest doc_type -> classifier label (None = not classifiable by design)
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


def load_golden() -> list[dict]:
    with open(GOLDEN) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_manifest_types() -> dict[str, str]:
    types = {}
    with open(DATA / "manifest.csv") as f:
        for row in csv.DictReader(f):
            types[row["filename"]] = row["doc_type"]
    # synthetic files are not in the manifest
    for p in (DATA / "clean" / "pay_slip").glob("*.pdf"):
        types[p.name] = "pay_slip"
    for p in (DATA / "clean" / "resume").glob("*.pdf"):
        types[p.name] = "resume"
    return types


def read_ground_truth(pdf_rel: Path) -> str:
    gt = DATA / "ground_truth" / pdf_rel.parent.name / f"{pdf_rel.stem}.txt"
    return gt.read_text() if gt.exists() else ""


# ---------------- OCR layer ----------------


def eval_ocr_layer() -> dict:
    from mortgage_rag.ocr import extract_text_pipeline

    per_file = []
    for scan in sorted((DATA / "degraded").rglob("*_scan.pdf")):
        clean_stem = scan.stem.removesuffix("_scan")
        gt_path = DATA / "ground_truth" / scan.parent.name / f"{clean_stem}.txt"
        if not gt_path.exists():
            continue
        gt_pages = gt_path.read_text().split("\f")[:DEGRADED_MAX_PAGES]
        reference = "\n".join(gt_pages)
        if not reference.strip():
            continue

        hypothesis, _ = extract_text_pipeline(str(scan))
        rates = text_error_rates(reference, hypothesis)
        per_file.append({"file": f"{scan.parent.name}/{scan.name}", **rates})

    n = len(per_file)
    return {
        "files": n,
        "mean_cer": round(sum(f["cer"] for f in per_file) / n, 4) if n else None,
        "mean_wer": round(sum(f["wer"] for f in per_file) / n, 4) if n else None,
        "per_file": per_file,
    }


# ---------------- Classification layer ----------------


def eval_classification_layer(include_degraded: bool = True, cfg=None) -> dict:
    from mortgage_rag.chunking import classify_page_content, compute_keyword_idf
    from mortgage_rag.ocr import extract_text_pipeline

    types = load_manifest_types()

    # IDF is fitted on the clean ground-truth corpus only — the same text the
    # labels derive from — so the degraded arm is scored with weights that never
    # saw a degraded document.
    idf = None
    if cfg is not None and getattr(cfg, "use_idf_classifier", False):
        idf = compute_keyword_idf([
            read_ground_truth(p.relative_to(DATA / "clean"))
            for p in sorted((DATA / "clean").rglob("*.pdf"))
        ])
    results = {"clean": defaultdict(int), "degraded": defaultdict(int)}
    pairs: dict[str, list[tuple[str, str]]] = {"clean": [], "degraded": []}
    mistakes = []

    for pdf in sorted((DATA / "clean").rglob("*.pdf")):
        expected = TYPE_MAP.get(types.get(pdf.name, pdf.parent.name))
        if expected is None:
            results["clean"]["unmapped"] += 1
            continue
        text = read_ground_truth(pdf.relative_to(DATA / "clean"))
        predicted, _ = classify_page_content(text[:4000], idf)
        results["clean"]["correct" if predicted == expected else "wrong"] += 1
        pairs["clean"].append((expected, predicted))
        if predicted != expected:
            mistakes.append({"file": pdf.name, "variant": "clean",
                            "expected": expected, "predicted": predicted})

    if include_degraded:
        for scan in sorted((DATA / "degraded").rglob("*_scan.pdf")):
            clean_name = scan.name.replace("_scan.pdf", ".pdf")
            expected = TYPE_MAP.get(types.get(clean_name, scan.parent.name))
            if expected is None:
                results["degraded"]["unmapped"] += 1
                continue
            text, _ = extract_text_pipeline(str(scan))
            predicted, _ = classify_page_content(text[:4000], idf)
            results["degraded"]["correct" if predicted == expected else "wrong"] += 1
            pairs["degraded"].append((expected, predicted))
            if predicted != expected:
                mistakes.append({"file": scan.name, "variant": "degraded",
                                "expected": expected, "predicted": predicted})

    out = {}
    for variant, counts in results.items():
        scored = counts["correct"] + counts["wrong"]
        out[variant] = {
            "scored": scored,
            "correct": counts["correct"],
            "unmapped": counts["unmapped"],
            "accuracy": round(counts["correct"] / scored, 4) if scored else None,
            # Accuracy over an imbalanced label set cannot show *which* class
            # misfires; the matrix and per-class precision can.
            "confusion": confusion_matrix(pairs[variant]) if pairs[variant] else None,
            "accuracy_ci": wilson_interval(counts["correct"], scored) if scored else None,
        }
    out["mistakes"] = mistakes
    return out


# ---------------- Retrieval + answer layers ----------------


def _chunk_pairs(result) -> list[tuple[str, object]]:
    """(filename, chunk) pairs — the same units the dense index is built from,
    so BM25 and dense retrieval compete over identical inputs."""
    return [(f.filename, c) for f in result.files for c in f.chunks]


def _build_corpus_index(variant: str, needed_files: set[str], cfg, retriever: str = "dense"):
    """Index the full clean corpus, or just the needed degraded files (OCR cost).

    Returns (index_or_None, chunk_pairs). BM25 and no-retrieval skip embedding
    entirely, which is also why the BM25 baseline is cheap to run.
    """
    from mortgage_rag.pipeline import process_files

    if variant == "clean":
        paths = [str(p) for p in sorted((DATA / "clean").rglob("*.pdf"))]
    else:
        paths = [
            str(p) for p in sorted((DATA / "degraded").rglob("*_scan.pdf"))
            if p.name in needed_files
        ]
    result = process_files(paths, cfg, build_vector_index=(retriever == "dense"))
    return result.index, _chunk_pairs(result)


def _build_bundle_index(bundle: str, cfg, retriever: str = "dense"):
    from mortgage_rag.pipeline import process_files

    result = process_files([str(DATA / "loan_files" / f"{bundle}.pdf")], cfg,
                           build_vector_index=(retriever == "dense"))
    sidecar = json.loads((DATA / "loan_files" / f"{bundle}.json").read_text())
    return result.index, _chunk_pairs(result), sidecar


def _retrieved_doc_ids(nodes, scope: str, sidecar: dict | None) -> list[str]:
    """Map retrieved nodes to comparable doc ids (filename, or bundle source file)."""
    ids = []
    if scope == "corpus":
        ids = [n.metadata.get("filename", "?") for n in nodes]
    else:
        page_to_source = {p["page"]: p["source"] for p in sidecar["pages"]}
        for n in nodes:
            ids.append(page_to_source.get(n.metadata.get("page_start", -1), "?"))
    return ids


def eval_retrieval_and_answer(
    cases: list[dict], cfg, run_answers: bool, backend=None, retriever: str = "dense"
) -> dict:
    """Score retrieval and (optionally) answers over the golden set.

    ``retriever`` selects the reference baseline: ``dense`` (shipped pipeline),
    ``bm25`` (classical sparse), or ``none`` (no context — measures how much the
    model already knows about these public forms).

    Retrieval is scored twice for the dense path: once on the bi-encoder's
    candidate list and once after the cross-encoder rerank. The delta is the
    reranker's measured contribution, which is otherwise assumed rather than known.
    """
    from mortgage_rag.orchestrator import ClassicalRAG
    from mortgage_rag.rag import build_reranker, retrieve

    by_scope: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_scope[c["scope"] + "|" + c["variant"]].append(c)

    # One reranker for the whole run: loading the cross-encoder per scope would
    # dominate the layer's runtime and change nothing about the scores.
    reranker = build_reranker(cfg) if (cfg.use_reranker and retriever == "dense") else None

    per_case = []
    for scope_key, scoped_cases in sorted(by_scope.items()):
        scope, variant = scope_key.split("|")
        sidecar = None
        if retriever == "none":
            index, chunks = None, []
        elif scope == "corpus":
            needed = {c["doc_id"] for c in scoped_cases}
            index, chunks = _build_corpus_index(variant, needed, cfg, retriever)
        else:
            index, chunks, sidecar = _build_bundle_index(
                scope.removeprefix("bundle:"), cfg, retriever
            )

        bm25 = BM25Retriever(chunks) if retriever == "bm25" else None
        orch = (
            ClassicalRAG(index, cfg, backend=backend)
            if run_answers and retriever == "dense"
            else None
        )

        for c in scoped_cases:
            entry = {
                "id": c["id"],
                "category": c["category"],
                "variant": c["variant"],
                "doc_id": c["doc_id"],  # bootstrap clusters on this
            }

            if retriever == "none":
                entry.update({"retrieval_hit": False, "rr": 0.0})
            else:
                if bm25 is not None:
                    nodes = bm25.retrieve(c["question"], top_k=cfg.top_k)
                else:
                    nodes = retrieve(index, c["question"], cfg)
                hit, rr = retrieval_metrics(
                    _retrieved_doc_ids(nodes, scope, sidecar), c["doc_id"]
                )
                entry.update({"retrieval_hit": hit, "rr": round(rr, 3)})

                # Post-rerank scoring on the same candidate list. The reranker
                # returns rerank_top_n nodes while retrieval produced top_k, so
                # the pre-rerank list is truncated to the same depth before
                # comparing — otherwise the delta would conflate reordering
                # (what the cross-encoder does) with truncation (what the cutoff
                # does), and the cross-encoder would be blamed for both.
                if reranker is not None and nodes:
                    from llama_index.core import QueryBundle

                    depth = cfg.rerank_top_n
                    pre_hit, pre_rr = retrieval_metrics(
                        _retrieved_doc_ids(nodes[:depth], scope, sidecar), c["doc_id"]
                    )
                    reranked = reranker.postprocess_nodes(
                        nodes, query_bundle=QueryBundle(c["question"])
                    )
                    r_hit, r_rr = retrieval_metrics(
                        _retrieved_doc_ids(reranked, scope, sidecar), c["doc_id"]
                    )
                    entry.update({
                        "retrieval_hit_at_n": pre_hit,   # dense, truncated to top_n
                        "rr_at_n": round(pre_rr, 3),
                        "retrieval_hit_reranked": r_hit,  # reranked, same depth
                        "rr_reranked": round(r_rr, 3),
                    })

            if run_answers and c["answer_type"] != "retrieval_only":
                t0 = time.time()
                if orch is not None:
                    res = orch.answer(c["question"])
                    answer_text, citations = res.answer, [vars(x) for x in res.citations]
                elif retriever == "none":
                    answer_text, citations = answer_without_retrieval(backend, c["question"]), []
                else:  # bm25: same prompt and context budget, different retriever
                    answer_text = _answer_from_nodes(backend, c["question"], nodes, cfg)
                    citations = [dict(n.metadata, score=n.score) for n in nodes]
                latency = time.time() - t0

                s = score_answer(answer_text, c["expected_answer"], c["answer_type"],
                                 c.get("distractor_answer"))
                entry.update({
                    "answer_pass": s.passed,
                    "matched_distractor": s.matched_distractor,
                    "reason": s.reason,
                    "latency_s": round(latency, 2),
                    "response": answer_text[:200],
                    # Right value, wrong provenance is still a defect at scale.
                    "citation_hit": citation_hit(
                        citations, c["doc_id"], c.get("gold_page", 0)
                    ),
                })
            per_case.append(entry)

    return _aggregate_rag(per_case, retriever)


def _answer_from_nodes(backend, question: str, nodes, cfg) -> str:
    """Generate from a node list using the orchestrator's own prompt and budget.

    Keeps the BM25 arm comparable to the dense arm: identical prompt, identical
    number of context chunks, so the only variable is which chunks were selected.
    """
    from mortgage_rag.orchestrator import QA_PROMPT

    if not nodes:
        return "No relevant information found."
    context = "\n\n".join(
        f"[{n.metadata.get('type', '?')}, pages "
        f"{n.metadata.get('page_start', '?')}-{n.metadata.get('page_end', '?')}]:\n{n.text}"
        for n in nodes[: cfg.rerank_top_n]
    )
    return backend.complete(QA_PROMPT.format(context=context, question=question))


def _aggregate_rag(per_case: list[dict], retriever: str = "dense") -> dict:
    agg: dict = {"per_case": per_case, "retriever": retriever}
    n = len(per_case)
    agg["retrieval"] = {
        "cases": n,
        "hit_at_k": round(sum(1 for c in per_case if c["retrieval_hit"]) / n, 4) if n else None,
        "mrr": round(sum(c["rr"] for c in per_case) / n, 4) if n else None,
        "hit_at_k_ci": wilson_interval(sum(1 for c in per_case if c["retrieval_hit"]), n),
        "mrr_ci": cluster_bootstrap_ci(per_case, "rr", "doc_id"),
    }

    # The reranker's measured value. Compared at equal depth (top_n vs top_n) so
    # the delta isolates reordering; the truncation cost from top_k to top_n is
    # reported separately because it is the cutoff's doing, not the model's.
    reranked = [c for c in per_case if "rr_reranked" in c]
    if reranked:
        m = len(reranked)
        pre_at_n = sum(c["rr_at_n"] for c in reranked) / m
        post_mrr = sum(c["rr_reranked"] for c in reranked) / m
        agg["retrieval"]["reranked"] = {
            "cases": m,
            "hit_at_n_dense": round(sum(1 for c in reranked if c["retrieval_hit_at_n"]) / m, 4),
            "mrr_at_n_dense": round(pre_at_n, 4),
            "hit_at_k": round(sum(1 for c in reranked if c["retrieval_hit_reranked"]) / m, 4),
            "mrr": round(post_mrr, 4),
            "mrr_delta": round(post_mrr - pre_at_n, 4),
            "improved": sum(1 for c in reranked if c["rr_reranked"] > c["rr_at_n"]),
            "worsened": sum(1 for c in reranked if c["rr_reranked"] < c["rr_at_n"]),
            # Gold documents lost purely by cutting top_k down to top_n.
            "truncation_losses": sum(
                1 for c in reranked if c["retrieval_hit"] and not c["retrieval_hit_at_n"]
            ),
        }

    answered = [c for c in per_case if "answer_pass" in c]
    if answered:
        by_cat: dict[str, list] = defaultdict(list)
        for c in answered:
            by_cat[c["category"]].append(c)
        passes = sum(1 for c in answered if c["answer_pass"])
        agg["answer"] = {
            "cases": len(answered),
            "pass_rate": round(passes / len(answered), 4),
            "pass_rate_ci": wilson_interval(passes, len(answered)),
            "pass_rate_bootstrap": cluster_bootstrap_ci(
                [{**c, "_pass": float(c["answer_pass"])} for c in answered], "_pass", "doc_id"
            ),
            "mean_latency_s": round(sum(c["latency_s"] for c in answered) / len(answered), 2),
            "by_category": {
                cat: {
                    "cases": len(cs),
                    "pass_rate": round(sum(1 for c in cs if c["answer_pass"]) / len(cs), 4),
                    "pass_rate_ci": wilson_interval(
                        sum(1 for c in cs if c["answer_pass"]), len(cs)
                    ),
                }
                for cat, cs in sorted(by_cat.items())
            },
            "adversarial_resistance": None,
        }
        adv = by_cat.get("adversarial", [])
        if adv:
            resisted = sum(1 for c in adv if not c["matched_distractor"])
            agg["answer"]["adversarial_resistance"] = round(resisted / len(adv), 4)
            agg["answer"]["adversarial_resistance_ci"] = wilson_interval(resisted, len(adv))

        # Scored only where the golden case carries a page-level gold label.
        cited = [c for c in answered if c.get("citation_hit") is not None]
        if cited:
            faithful = sum(1 for c in cited if c["citation_hit"])
            agg["answer"]["citation_faithfulness"] = {
                "cases": len(cited),
                "rate": round(faithful / len(cited), 4),
                "ci": wilson_interval(faithful, len(cited)),
                # A right answer whose citations point elsewhere is unauditable.
                "passed_but_uncited": sum(
                    1 for c in cited if c["answer_pass"] and not c["citation_hit"]
                ),
            }
    return agg


# ---------------- Reporting / regression ----------------


def corpus_stats() -> dict:
    return {
        "clean_pdfs": len(list((DATA / "clean").rglob("*.pdf"))),
        "degraded_pdfs": len(list((DATA / "degraded").rglob("*.pdf"))),
        "bundles": len(list((DATA / "loan_files").glob("*.pdf"))),
        "doc_types": len([d for d in (DATA / "clean").iterdir() if d.is_dir()]),
    }


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["cuda"] = "unknown"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
        info["commit"] = commit
    except Exception:
        pass
    return info


def _ci(bounds) -> str:
    """Render a confidence interval inline, or nothing when it is unavailable."""
    if not bounds:
        return ""
    return f" [95% CI {bounds[0]:.3f}–{bounds[1]:.3f}]"


def _confusion_section(conf: dict, variant: str) -> list[str]:
    """Full confusion matrix + per-class precision/recall/F1.

    Reported because accuracy alone cannot distinguish "one class is a keyword
    sink that absorbs everything" from "errors are spread evenly" — those have
    completely different fixes.
    """
    labels = conf["labels"]
    short = {lab: lab[:14] for lab in labels}
    lines = [
        "", f"### Confusion matrix — {variant} (rows = expected, columns = predicted)", "",
        "| expected \\ predicted | " + " | ".join(short[c] for c in labels) + " | total |",
        "|---" * (len(labels) + 2) + "|",
    ]
    for exp in labels:
        row = conf["matrix"][exp]
        total = sum(row.values())
        if total == 0 and conf["per_class"][exp]["predicted"] == 0:
            continue
        cells = [
            (f"**{row[p]}**" if p == exp and row[p] else (str(row[p]) if row[p] else "·"))
            for p in labels
        ]
        lines.append(f"| {short[exp]} | " + " | ".join(cells) + f" | {total} |")

    lines += ["", f"### Per-class scores — {variant}", "",
              "| class | support | predicted | precision | recall | F1 |", "|---|---|---|---|---|---|"]
    for lab in labels:
        pc = conf["per_class"][lab]
        if pc["support"] == 0 and pc["predicted"] == 0:
            continue
        fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(
            f"| {lab} | {pc['support']} | {pc['predicted']} | "
            f"{fmt(pc['precision'])} | {fmt(pc['recall'])} | {fmt(pc['f1'])} |"
        )
    lines.append("")
    lines.append(
        f"Accuracy {conf['accuracy']:.1%} over {conf['n']} files; macro F1 "
        f"{conf['macro_f1']:.4f}. Macro F1 weights every class equally, so a class "
        "that is frequently predicted but never correct drags it down while accuracy "
        "stays high — that gap is the signal to look for."
    )
    return lines


def write_report(results: dict) -> None:
    lines = ["# Evaluation report", ""]
    hw = results["meta"]["hardware"]
    cs = results["meta"]["corpus"]
    lines += [
        f"- Run: {results['meta']['timestamp']} | mode: {results['meta']['mode']} | "
        f"backend: {results['meta']['backend']} | model: {results['meta'].get('model', '?')} | "
        f"device: {hw.get('cuda')} | commit: {hw.get('commit', '?')}",
        f"- Corpus: {cs['clean_pdfs']} clean + {cs['degraded_pdfs']} degraded PDFs, "
        f"{cs['bundles']} adversarial bundles, {cs['doc_types']} doc types",
    ]
    secs = results["meta"].get("layer_seconds") or {}
    if secs:
        lines.append("- Layer runtime: " + ", ".join(f"{k} {v}s" for k, v in secs.items()))
    lines += [
        "",
        "| Layer | Metric | Value |",
        "|---|---|---|",
    ]

    if "ocr" in results:
        o = results["ocr"]
        lines.append(f"| OCR (degraded scans) | mean CER / WER over {o['files']} files | "
                     f"{o['mean_cer']} / {o['mean_wer']} |")
    if "classification" in results:
        for variant in ("clean", "degraded"):
            c = results["classification"].get(variant)
            if c and c["scored"]:
                lines.append(f"| Doc classification ({variant}) | accuracy | "
                             f"{c['accuracy']:.1%} ({c['correct']}/{c['scored']}){_ci(c.get('accuracy_ci'))} |")
                conf = c.get("confusion") or {}
                if conf.get("macro_f1") is not None:
                    lines.append(f"| Doc classification ({variant}) | macro F1 | "
                                 f"{conf['macro_f1']:.4f} |")
    if "rag" in results:
        r = results["rag"]["retrieval"]
        retr = results["rag"].get("retriever", "dense")
        label = "Retrieval" if retr == "dense" else f"Retrieval [{retr}]"
        lines.append(f"| {label} | hit@k / MRR over {r['cases']} cases | "
                     f"{r['hit_at_k']:.1%} / {r['mrr']}{_ci(r.get('hit_at_k_ci'))} |")
        rr = r.get("reranked")
        if rr:
            lines.append(f"| {label} (dense, truncated to top_n) | hit / MRR | "
                         f"{rr['hit_at_n_dense']:.1%} / {rr['mrr_at_n_dense']} |")
            lines.append(f"| {label} (post-rerank, same depth) | hit / MRR | "
                         f"{rr['hit_at_k']:.1%} / {rr['mrr']} |")
            lines.append(f"| Reranker contribution | MRR delta at equal depth | "
                         f"{rr['mrr_delta']:+.4f} ({rr['improved']} improved, "
                         f"{rr['worsened']} worsened) |")
            lines.append(f"| Truncation cost (top_k→top_n) | gold docs dropped | "
                         f"{rr['truncation_losses']} of {rr['cases']} |")
        a = results["rag"].get("answer")
        if a:
            lines.append(f"| Answer | pass rate over {a['cases']} cases | "
                         f"{a['pass_rate']:.1%}{_ci(a.get('pass_rate_ci'))} |")
            for cat, v in a["by_category"].items():
                lines.append(f"| Answer ({cat}) | pass rate | "
                             f"{v['pass_rate']:.1%} ({v['cases']} cases)"
                             f"{_ci(v.get('pass_rate_ci'))} |")
            if a["adversarial_resistance"] is not None:
                lines.append(f"| Adversarial resistance | distractor rejected | "
                             f"{a['adversarial_resistance']:.1%}"
                             f"{_ci(a.get('adversarial_resistance_ci'))} |")
            cf = a.get("citation_faithfulness")
            if cf:
                lines.append(f"| Citation faithfulness | gold page cited "
                             f"({cf['cases']} scoreable) | {cf['rate']:.1%}"
                             f"{_ci(cf.get('ci'))} |")
            lines.append(f"| Latency | mean per answered case | {a['mean_latency_s']}s |")

        bs = a.get("pass_rate_bootstrap") if a else None
        if bs:
            interval = (
                f"95% CI [{bs['ci_low']:.3f}, {bs['ci_high']:.3f}]"
                if bs.get("ci_low") is not None else f"CI unavailable ({bs.get('note')})"
            )
            note = f" {bs['note']}." if bs.get("note") and bs.get("ci_low") is not None else ""
            lines += [
                "",
                f"Cluster bootstrap over source documents ({bs['n_clusters']} clusters, "
                f"{bs['n_items']} cases, {bs['iters']} resamples): answer pass rate "
                f"{bs['mean']:.3f}, {interval}.{note} Cases sharing a document fail "
                "together, so documents are resampled rather than cases — resampling "
                "cases would treat correlated failures as independent evidence.",
            ]

    for variant in ("clean", "degraded"):
        conf = (results.get("classification", {}).get(variant) or {}).get("confusion")
        if conf:
            lines += _confusion_section(conf, variant)

    if "rag" in results and any("answer_pass" in c for c in results["rag"]["per_case"]):
        lines += ["", "## Per-case results", "",
                  "| Case | Category | Variant | Retrieval | Answer | Note |", "|---|---|---|---|---|---|"]
        for c in results["rag"]["per_case"]:
            ans = ("PASS" if c.get("answer_pass") else "FAIL") if "answer_pass" in c else "—"
            lines.append(f"| {c['id']} | {c['category']} | {c['variant']} | "
                         f"{'hit' if c['retrieval_hit'] else 'miss'} | {ans} | "
                         f"{c.get('reason', '')} |")

    path = REPORT if not results["meta"].get("tag") else (
        REPORT.with_name(f"report-{results['meta']['tag']}.md")
    )
    path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {path}")


def compare_to_baseline(results: dict, baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text())
    regressions = []

    base_cases = {c["id"]: c for c in baseline.get("rag", {}).get("per_case", [])}
    for c in results.get("rag", {}).get("per_case", []):
        b = base_cases.get(c["id"])
        if not b:
            continue
        if b["retrieval_hit"] and not c["retrieval_hit"]:
            regressions.append(f"{c['id']}: retrieval hit -> miss")
        if b.get("answer_pass") and c.get("answer_pass") is False:
            regressions.append(f"{c['id']}: answer PASS -> FAIL")

    for layer, metric in (("ocr", "mean_cer"),):
        if layer in results and layer in baseline:
            new, old = results[layer].get(metric), baseline[layer].get(metric)
            if new is not None and old is not None and new > old * 1.10:
                regressions.append(f"{layer}.{metric}: {old} -> {new} (>10% worse)")

    if regressions:
        print("\nREGRESSIONS DETECTED:")
        for r in regressions:
            print(f"  - {r}")
        return 1
    print("\nNo regressions vs baseline.")
    return 0


def _parse_overrides(pairs: list[str]) -> dict:
    """Turn --set key=value into typed PipelineConfig kwargs.

    Types are read off the dataclass field so an ablation arm cannot silently
    pass the string "false" where a bool is expected — that would evaluate truthy
    and quietly invalidate the arm.
    """
    from dataclasses import fields

    from mortgage_rag.config import PipelineConfig

    types = {f.name: f.type for f in fields(PipelineConfig)}
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE, got: {pair}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in types:
            raise SystemExit(f"--set: unknown config field '{key}'")
        t = types[key]
        if t in ("bool", bool):
            out[key] = raw.strip().lower() in ("1", "true", "yes")
        elif t in ("int", int):
            out[key] = int(raw)
        elif t in ("float", float):
            out[key] = float(raw)
        else:
            out[key] = raw
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="retrieval",
                    help="comma list: ocr,classification,retrieval,answer")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--all", action="store_true", help="all layers over the full corpus")
    ap.add_argument("--mode", default="classical", choices=["classical", "agentic"])
    ap.add_argument("--backend", default=None,
                    help="override llm backend: llama_cpp | openai_compat | mock")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="reuse layers already completed by a previous (crashed) run's "
                         "checkpoint instead of recomputing them")
    ap.add_argument("--classification-clean-only", action="store_true",
                    help="skip OCR-heavy degraded classification (CI)")
    ap.add_argument("--retriever", default="dense", choices=["dense", "bm25", "none"],
                    help="reference baseline: dense (shipped), bm25 (sparse), "
                         "none (no context — separates retrieval from parametric recall)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any PipelineConfig field; repeatable. Drives the "
                         "ablation grid, e.g. --set use_reranker=false --set chunk_size=250")
    ap.add_argument("--tag", default=None,
                    help="label this run in the report/baseline filename (ablation arms)")
    args = ap.parse_args()

    layers = {x.strip() for x in args.layers.split(",")}
    if args.retrieval_only:
        layers = {"retrieval"}
    if args.all:
        layers = {"ocr", "classification", "retrieval", "answer"}

    from mortgage_rag.config import PipelineConfig

    cfg = PipelineConfig.load(mode=args.mode, llm_backend=args.backend,
                              **_parse_overrides(args.set))

    results: dict = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": cfg.mode,
            "backend": cfg.llm_backend if "answer" in layers else "none",
            "model": (cfg.llm_model_name or cfg.llm_filename) if "answer" in layers else "none",
            "layers": sorted(layers),
            "retriever": args.retriever,
            "tag": args.tag,
            "overrides": _parse_overrides(args.set),
            "corpus": corpus_stats(),
            "hardware": hardware_info(),
        }
    }

    timings: dict[str, float] = {}
    results["meta"]["layer_seconds"] = timings

    # A crashed run leaves its completed layers in the checkpoint; --resume
    # reuses them so an answer-layer failure never costs the OCR pass again.
    suffix = f"-{args.tag}" if args.tag else ""
    checkpoint = BASELINES / f"checkpoint{suffix}.json"
    prior: dict = {}
    if args.resume and checkpoint.exists():
        prior = json.loads(checkpoint.read_text())
        # A checkpoint from a different arm holds results for a different system;
        # reusing it would silently mix configurations into one report.
        prior_meta = prior.get("meta", {})
        same_arm = (
            prior_meta.get("retriever", "dense") == args.retriever
            and (prior_meta.get("overrides") or {}) == _parse_overrides(args.set)
        )
        if not same_arm:
            print("   (checkpoint is from a different config arm — not reusing)")
            prior = {}

    def save_checkpoint() -> None:
        BASELINES.mkdir(exist_ok=True)
        checkpoint.write_text(json.dumps(results, indent=2))

    def reuse(layer: str) -> bool:
        if layer not in prior:
            return False
        results[layer] = prior[layer]
        timings[layer] = prior.get("meta", {}).get("layer_seconds", {}).get(layer, 0.0)
        results["meta"].setdefault("resumed_layers", []).append(layer)
        print(f"   (reused from checkpoint, originally {timings[layer]}s)")
        return True

    if "ocr" in layers:
        print("== OCR layer (degraded scans vs ground truth) ==")
        if not reuse("ocr"):
            t0 = time.time()
            results["ocr"] = eval_ocr_layer()
            timings["ocr"] = round(time.time() - t0, 1)
            save_checkpoint()
        print(f"   mean CER {results['ocr']['mean_cer']} | mean WER {results['ocr']['mean_wer']} "
              f"over {results['ocr']['files']} files [{timings['ocr']}s]")

    if "classification" in layers:
        print("== Classification layer ==")
        if not reuse("classification"):
            t0 = time.time()
            results["classification"] = eval_classification_layer(
                include_degraded=not args.classification_clean_only, cfg=cfg
            )
            timings["classification"] = round(time.time() - t0, 1)
            save_checkpoint()
        for v in ("clean", "degraded"):
            c = results["classification"][v]
            if c["scored"]:
                print(f"   {v}: {c['accuracy']:.1%} ({c['correct']}/{c['scored']})")
        print(f"   [{timings['classification']}s]")

    if "retrieval" in layers or "answer" in layers:
        print("== Retrieval" + (" + answer" if "answer" in layers else "") + " layer ==")
        # A retrieval-only checkpoint cannot satisfy a run that needs answers.
        rag_reusable = "answer" not in layers or "answer" in prior.get("rag", {})
        if not (rag_reusable and reuse("rag")):
            t0 = time.time()
            cases = load_golden()
            backend = None
            if "answer" in layers and cfg.llm_backend == "mock":
                from mortgage_rag.backends import MockBackend

                backend = MockBackend()
            if args.retriever != "dense" and "answer" in layers and backend is None:
                from mortgage_rag.backends import build_backend

                backend = build_backend(cfg)  # baseline arms bypass the orchestrator
            results["rag"] = eval_retrieval_and_answer(
                cases, cfg, run_answers="answer" in layers, backend=backend,
                retriever=args.retriever,
            )
            timings["rag"] = round(time.time() - t0, 1)
            save_checkpoint()
        r = results["rag"]["retrieval"]
        print(f"   retrieval: hit@k {r['hit_at_k']:.1%} | MRR {r['mrr']}")
        a = results["rag"].get("answer")
        if a:
            print(f"   answer: pass {a['pass_rate']:.1%} | "
                  f"adversarial resistance {a['adversarial_resistance']}")
        print(f"   [{timings['rag']}s]")

    write_report(results)

    if args.save_baseline:
        BASELINES.mkdir(exist_ok=True)
        out = BASELINES / (f"{args.tag}.json" if args.tag else "latest.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"Baseline saved to {out}")
        checkpoint.unlink(missing_ok=True)

    if args.compare:
        return compare_to_baseline(results, Path(args.compare))
    return 0


if __name__ == "__main__":
    sys.exit(main())
