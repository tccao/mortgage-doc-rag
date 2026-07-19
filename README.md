# Mortgage Doc RAG

RAG pipeline for messy, multi-document mortgage loan files — dual-engine OCR, document-type
separation, deterministic validation — wrapped in a **layered evaluation harness** that
benchmarks every stage of the pipeline against frozen ground truth, including adversarial
"gamed" loan files.

> Originated during an AI externship; fully rebuilt on public data.

## Results

<!-- RESULTS:BEGIN -->
_Populated by `evals/run_eval.py --all --save-baseline`; see `evals/report.md` for the
latest committed run._
<!-- RESULTS:END -->

Every number is regenerable: deterministic scoring against frozen references, temperature-0
generation, seeded data generators, locked dependencies (`uv.lock`), checksummed corpus.

## Why this exists

Mortgage processing runs on unstructured PDF bundles: closing disclosures, loan estimates,
pay stubs, W-2s, appraisals — often scanned badly, sometimes stapled together into one
file, occasionally containing content designed to game automated processing. Extracting
reliable answers requires more than "chunk and retrieve": you need document separation,
provenance-aware retrieval, deterministic validation, and — above all — **evals that tell
you which stage broke when quality moves**.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the diagram and
[docs/design.md](docs/design.md) for the ADR-style decision log (every trade-off, with
what I'd do differently at scale).

Pipeline: PDF → text-layer detection → OCR when needed (ChandraOCR → Tesseract 5 fallback)
→ keyword doc-type classification → boundary detection → metadata-preserving chunking →
regex validation (amounts, SSN exposure, cross-document consistency) → vector index
(bge-small) → orchestrator (classical v1 now; agentic v2 on the roadmap, same interface).

## The eval harness (the point of this repo)

```
evals/
  golden_set.jsonl   # 29 typed cases: extraction / retrieval / adversarial, clean + degraded
  scoring.py         # numeric-with-tolerance, exact, contains, fuzzy + distractor detection
  run_eval.py        # layered runner with regression compare vs committed baseline
  baselines/         # committed results; CI fails on PASS->FAIL flips
```

Four independent layers:

| Layer | Reference | LLM needed |
|---|---|---|
| OCR | frozen text of clean originals vs OCR of degraded scans (CER/WER) | no |
| Classification | `data/manifest.csv` doc types over the full corpus | no |
| Retrieval | gold document per question (hit@k, MRR) | no |
| Answer | typed expected answers, explicit distractor FAILs | yes |

Adversarial bundles (`data/loan_files/`) inject a fake "corrected" closing disclosure,
an instruction-injection page, and a conflicting stated-income affidavit into otherwise
authentic files — sidecar JSON labels every page, and an answer that parrots a distractor
value fails even when fuzzy metrics look good.

```bash
uv run python evals/run_eval.py --retrieval-only        # deterministic, CI-safe
uv run python evals/run_eval.py --layers ocr,classification
uv run python evals/run_eval.py --all --save-baseline   # full run (needs LLM)
uv run python evals/run_eval.py --retrieval-only --compare evals/baselines/latest.json
```

## Data corpus

130+ PDFs across 13 document types, all rebuildable from `data/manifest.csv`
(SHA-256-checksummed downloads; US-government works — CFPB, HUD, IRS, VA, USDA, EPA, FTC —
plus seeded synthetic pay stubs, since no public-domain filled pay stubs exist):

- `data/clean/` — 60+ source documents (filled samples + blank forms, English + Spanish)
- `data/degraded/` — scan-simulated variants (skew, noise, blur, JPEG artifacts; seeded)
- `data/loan_files/` — adversarial multi-doc bundles + page-role sidecars
- `data/ground_truth/` — frozen plain text per clean PDF (the deterministic reference)

## Quickstart

```bash
uv sync                          # core (no LLM): pipeline, evals, tests
uv sync --extra llm --extra ui   # + llama.cpp backend and Gradio app

uv run pytest -q
uv run python -m mortgage_rag.pipeline data/loan_files/loan_file_01_conflicting_cd.pdf
uv run python -m mortgage_rag.app          # Gradio UI (needs llm+ui extras)
```

Self-contained on purpose: OCR uses the `tesserocr` wheel + committed `tools/tessdata`
(no system tesseract), rendering uses PyMuPDF (no poppler). A Colab demo lives in
[notebooks/demo.ipynb](notebooks/demo.ipynb) — it picks the model tier from the detected GPU.

## Configuration

One `PipelineConfig` object drives everything (see `src/mortgage_rag/config.py`):
defaults < `config.yaml` < `MRAG_*` env vars < explicit kwargs. Swapping the LLM is one
field (`llm_backend`: local GGUF via llama.cpp, any OpenAI-compatible endpoint, or a mock
for tests). `mode: classical | agentic` selects the orchestrator behind a single interface.

## Roadmap

- **v2 — agentic RAG** (same `Orchestrator` interface, same evals): doc-type routing,
  tool loop (retrieval / validation / consistency checks), grounding self-check, multi-hop
  cross-document questions. The eval harness will publish a classical-vs-agentic comparison
  table — accuracy, adversarial resistance, latency, tokens — from the same golden set.
- OCR-layer red-team growth: distractor synthesis, real phone-capture samples.
- Bootstrap confidence intervals over the golden set.

## Docs

- [docs/design.md](docs/design.md) — decision log: every trade-off + limitations
- [docs/architecture.md](docs/architecture.md) — pipeline + eval diagram
- [docs/ai-assisted.md](docs/ai-assisted.md) — how AI assistance was used and verified

## License

MIT. Corpus documents are US-government works (public domain); synthetic documents are
clearly watermarked as such.
