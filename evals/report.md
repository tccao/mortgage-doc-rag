# Evaluation report

- Run: 2026-07-20 00:20:34 | mode: classical | backend: openai_compat | model: ornith-1.0-35b-Q4_K_M.gguf | device: NVIDIA A100-SXM4-40GB | commit: fde7c36
- Corpus: 64 clean + 64 degraded PDFs, 3 adversarial bundles, 13 doc types
- Layer runtime: ocr 1469.5s, classification 0.0s, rag 254.9s

| Layer | Metric | Value |
|---|---|---|
| OCR (degraded scans) | mean CER / WER over 64 files | 0.7033 / 0.9308 |
| Doc classification (clean) | accuracy | 94.6% (53/56) |
| Doc classification (degraded) | accuracy | 92.9% (52/56) |
| Retrieval | hit@k / MRR over 29 cases | 100.0% / 0.862 |
| Answer | pass rate over 26 cases | 69.2% |
| Answer (adversarial) | pass rate | 80.0% (5 cases) |
| Answer (extraction) | pass rate | 66.7% (21 cases) |
| Adversarial resistance | distractor rejected | 80.0% |
| Latency | mean per answered case | 6.57s |

## Per-case results

| Case | Category | Variant | Retrieval | Answer | Note |
|---|---|---|---|---|---|
| adv-b1-loan-amount | adversarial | clean | hit | PASS | expected number found |
| adv-b1-rate | adversarial | clean | hit | FAIL | answered with distractor value |
| adv-b2-loan-amount | adversarial | clean | hit | PASS | expected number found |
| adv-b2-income | adversarial | clean | hit | PASS | expected number found |
| adv-b3-income | adversarial | clean | hit | PASS | expected number found |
| clean-cd-loan-amount | extraction | clean | hit | PASS | expected number found |
| clean-cd-rate | extraction | clean | hit | PASS | expected number found |
| clean-cd-pi | extraction | clean | hit | FAIL | expected number absent |
| clean-le-lender | extraction | clean | hit | FAIL | expected string absent |
| clean-cd-borrowers | extraction | clean | hit | FAIL | similarity 0.02 |
| clean-le-sale-price | extraction | clean | hit | PASS | expected number found |
| clean-refi-loan-amount | extraction | clean | hit | FAIL | expected number absent |
| clean-refi-rate | extraction | clean | hit | FAIL | expected number absent |
| clean-refi-apr | extraction | clean | hit | FAIL | expected number absent |
| clean-paystub-rivera-gross | extraction | clean | hit | PASS | expected number found |
| clean-paystub-rivera-net | extraction | clean | hit | PASS | expected number found |
| clean-paystub-okafor-id | extraction | clean | hit | PASS | expected string present |
| clean-paystub-okafor-ytd | extraction | clean | hit | PASS | expected number found |
| clean-paystub-lindqvist-employer | extraction | clean | hit | PASS | expected string present |
| clean-cd-prepay-penalty | extraction | clean | hit | FAIL | expected number absent |
| clean-retrieval-refi | retrieval | clean | hit | — |  |
| clean-retrieval-urla | retrieval | clean | hit | — |  |
| clean-retrieval-hud1 | retrieval | clean | hit | — |  |
| degraded-cd-loan-amount | extraction | degraded | hit | PASS | expected number found |
| degraded-cd-rate | extraction | degraded | hit | PASS | expected number found |
| degraded-refi-loan-amount | extraction | degraded | hit | PASS | expected number found |
| degraded-paystub-rivera-gross | extraction | degraded | hit | PASS | expected number found |
| degraded-paystub-okafor-id | extraction | degraded | hit | PASS | expected string present |
| degraded-paystub-lindqvist-employer | extraction | degraded | hit | PASS | expected string present |
