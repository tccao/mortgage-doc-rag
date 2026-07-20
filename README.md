# Mortgage Doc RAG

RAG pipeline for messy, multi-document mortgage loan files — dual-engine OCR, document-type
separation, deterministic validation — wrapped in a **layered evaluation harness** that
benchmarks every stage of the pipeline against frozen ground truth, including adversarial
"gamed" loan files.

> Originated during an AI externship; fully rebuilt on public data.

## Results

<!-- RESULTS:BEGIN -->
Full-corpus run, Ornith-1.0-35B (Q4_K_M) via llama.cpp server on an A100 40GB, reranker off
per [ADR-10](docs/design.md) ([evals/report.md](evals/report.md) for per-case detail):

| Layer | Metric | Value |
| --- | --- | --- |
| OCR (degraded scans) | mean CER / WER over 64 files | 0.703 / 0.931 |
| Doc classification | accuracy, clean / degraded | 94.6% / 92.9% |
| Retrieval | hit@k / MRR over 29 cases | 100% / 0.862 |
| Answer | pass rate over 26 cases | 65.4% |
| Adversarial resistance | distractor rejected | 80.0% |
| Citation faithfulness | gold page cited, 21 scoreable cases | 95.2% |
| Latency | mean per answered case | 6.8s |

One adversarial case answered with the injected "corrected" closing-disclosure value,
exactly the failure mode the distractor scoring exists to expose. A fuzzy-similarity
metric alone would have scored it a near-pass.

Answer pass rate moved 69.2% to 65.4% when I turned the reranker off, the opposite of what
I predicted in ADR-10. I keep the prediction and the refutation in the log rather than
quietly restating the number: see [ADR-10](docs/design.md) for the two reasons, a depth
confound I introduced and a generation token budget that is doing more damage than
retrieval ever was.
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

```text
evals/
  golden_set.jsonl   # 29 typed cases: extraction / retrieval / adversarial, clean + degraded
  scoring.py         # typed answer scoring + distractor detection; confusion matrix,
                     #   Wilson intervals, cluster bootstrap
  retrievers.py      # reference baselines: BM25 (self-contained) and no-retrieval
  run_eval.py        # layered runner, regression compare, ablation arms
  baselines/         # committed results; CI fails on PASS->FAIL flips
```

Four independent layers:

| Layer | Reference | LLM needed |
| --- | --- | --- |
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

### Reference baselines and ablations

The committed baselines are *regression* baselines — they answer "did this change break
anything?". They cannot answer "does the complexity earn its place?". `--retriever` swaps
in a simpler system and `--set` overrides any config field, so an ablation arm is a config
change rather than a code branch. Arms write `report-<tag>.md` and `baselines/<tag>.json`
and never overwrite the shipped baseline.

```bash
uv run python evals/run_eval.py --layers retrieval --retriever bm25 --tag bm25
uv run python evals/run_eval.py --layers retrieval --set use_reranker=false --tag no-rerank
uv run python evals/run_eval.py --all --retriever none --tag no-retrieval   # needs LLM
./scripts/run_ablations.sh retrieval                                        # whole grid
```

Retrieval layer, 29 cases, identical chunks for both retrievers:

| Retriever | hit@k | MRR | vs dense |
| --- | --- | --- | --- |
| Dense (bge-small) | 100.0% | 0.862 | — |
| BM25 | 89.7% | 0.784 | worse on 7 cases, equal on 20, **better on 2** |

Dense retrieval earns its cost — but not uniformly. Both cases BM25 wins are adversarial:
it ranks the authentic closing disclosure first while dense ranks it *behind* the planted
fake "corrected" one. The forgery is semantically near-identical, so cosine similarity
cannot separate the two; lexical scoring can. See [ADR-9](docs/design.md).

The reranker, scored at equal depth for the first time, is **not** earning its place:

| Stage | hit | MRR |
| --- | --- | --- |
| Dense, truncated to `top_n` | 100.0% | 0.862 |
| Cross-encoder reranked, same depth | 89.7% | 0.828 |

Truncation costs 0 of 29 cases, so the whole drop is the cross-encoder demoting gold
documents. I then predicted that turning it off would lift the answer pass rate, because two
failures looked like cases where reranking hid the gold document before generation. I ran it
and the pass rate fell instead, 69.2% to 65.4%, with both predicted recoveries still failing.
The prediction and its refutation both stay in [ADR-10](docs/design.md), because the reason it
failed is the more useful finding.

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
- **Hybrid dense+sparse retrieval**, gated on the ablation grid — BM25 already beats
  dense on the adversarial cases where a forged document is semantically identical
  to the authentic one (see above).
- **No-retrieval baseline on the answer layer**, to separate retrieval from the
  model's parametric recall of these public forms. Wired (`--retriever none`) but
  not yet run against an LLM.
- OCR-layer red-team growth: distractor synthesis, real phone-capture samples.
- Bootstrap confidence intervals over the golden set — clustered by document, since
  cases sharing a source document are not independent.

## Docs

- [docs/design.md](docs/design.md) — decision log: every trade-off + limitations
- [docs/architecture.md](docs/architecture.md) — pipeline + eval diagram
- [docs/ai-assisted.md](docs/ai-assisted.md) — how AI assistance was used and verified

## License

MIT. Corpus documents are US-government works (public domain); synthetic documents are
clearly watermarked as such.
