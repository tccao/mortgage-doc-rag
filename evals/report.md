# Evaluation report

- Run: 2026-07-20 00:20:34 | mode: classical | backend: openai_compat | model: ornith-1.0-35b-Q4_K_M.gguf | device: NVIDIA A100-SXM4-40GB | commit: fde7c36
- Corpus: 64 clean + 64 degraded PDFs, 3 adversarial bundles, 13 doc types
- Layer runtime: ocr 1469.5s, classification 0.0s, rag 254.9s

| Layer | Metric | Value |
|---|---|---|
| OCR (degraded scans) | mean CER / WER over 64 files | 0.7033 / 0.9308 |
| Doc classification (clean) | accuracy | 94.6% (53/56) [95% CI 0.854–0.982] |
| Doc classification (clean) | macro F1 | 0.9505 |
| Doc classification (degraded) | accuracy | 92.9% (52/56) [95% CI 0.830–0.972] |
| Doc classification (degraded) | macro F1 | 0.9439 |
| Retrieval | hit@k / MRR over 29 cases | 100.0% / 0.862 [95% CI 0.883–1.000] |
| Answer | pass rate over 26 cases | 69.2% [95% CI 0.500–0.835] |
| Answer (adversarial) | pass rate | 80.0% (5 cases) [95% CI 0.376–0.964] |
| Answer (extraction) | pass rate | 66.7% (21 cases) [95% CI 0.454–0.828] |
| Adversarial resistance | distractor rejected | 80.0% [95% CI 0.376–0.964] |
| Latency | mean per answered case | 6.57s |

Cluster bootstrap over source documents (15 clusters, 26 cases, 2000 resamples): answer pass rate 0.692, 95% CI [0.500, 0.950]. Cases sharing a document fail together, so documents are resampled rather than cases — resampling cases would treat correlated failures as independent evidence.

### Confusion matrix — clean (rows = expected, columns = predicted)

| expected \ predicted | Appraisal | Closing Disclo | Loan Applicati | Loan Estimate | Mortgage Contr | Pay Slip | Resume | Settlement Sta | Tax Document | total |
|---|---|---|---|---|---|---|---|---|---|---|
| Appraisal | **3** | · | · | · | · | · | · | · | · | 3 |
| Closing Disclo | · | **18** | · | · | · | · | · | · | · | 18 |
| Loan Applicati | · | · | **1** | · | 1 | · | · | · | · | 2 |
| Loan Estimate | · | · | · | **15** | 2 | · | · | · | · | 17 |
| Mortgage Contr | · | · | · | · | · | · | · | · | · | 0 |
| Pay Slip | · | · | · | · | · | **3** | · | · | · | 3 |
| Resume | · | · | · | · | · | · | **1** | · | · | 1 |
| Settlement Sta | · | · | · | · | · | · | · | **2** | · | 2 |
| Tax Document | · | · | · | · | · | · | · | · | **10** | 10 |

### Per-class scores — clean

| class | support | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| Appraisal | 3 | 3 | 1.000 | 1.000 | 1.000 |
| Closing Disclosure | 18 | 18 | 1.000 | 1.000 | 1.000 |
| Loan Application | 2 | 1 | 1.000 | 0.500 | 0.667 |
| Loan Estimate | 17 | 15 | 1.000 | 0.882 | 0.938 |
| Mortgage Contract | 0 | 3 | 0.000 | — | 0.000 |
| Pay Slip | 3 | 3 | 1.000 | 1.000 | 1.000 |
| Resume | 1 | 1 | 1.000 | 1.000 | 1.000 |
| Settlement Statement | 2 | 2 | 1.000 | 1.000 | 1.000 |
| Tax Document | 10 | 10 | 1.000 | 1.000 | 1.000 |

Accuracy 94.6% over 56 files; macro F1 0.9505. Macro F1 weights every class equally, so a class that is frequently predicted but never correct drags it down while accuracy stays high — that gap is the signal to look for.

### Confusion matrix — degraded (rows = expected, columns = predicted)

| expected \ predicted | Appraisal | Closing Disclo | Loan Applicati | Loan Estimate | Mortgage Contr | Pay Slip | Resume | Settlement Sta | Tax Document | total |
|---|---|---|---|---|---|---|---|---|---|---|
| Appraisal | **3** | · | · | · | · | · | · | · | · | 3 |
| Closing Disclo | · | **18** | · | · | · | · | · | · | · | 18 |
| Loan Applicati | · | · | **1** | · | 1 | · | · | · | · | 2 |
| Loan Estimate | · | · | · | **15** | 2 | · | · | · | · | 17 |
| Mortgage Contr | · | · | · | · | · | · | · | · | · | 0 |
| Pay Slip | · | · | · | · | · | **3** | · | · | · | 3 |
| Resume | · | · | · | · | · | · | **1** | · | · | 1 |
| Settlement Sta | · | · | · | · | · | · | · | **2** | · | 2 |
| Tax Document | · | · | · | · | 1 | · | · | · | **9** | 10 |

### Per-class scores — degraded

| class | support | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| Appraisal | 3 | 3 | 1.000 | 1.000 | 1.000 |
| Closing Disclosure | 18 | 18 | 1.000 | 1.000 | 1.000 |
| Loan Application | 2 | 1 | 1.000 | 0.500 | 0.667 |
| Loan Estimate | 17 | 15 | 1.000 | 0.882 | 0.938 |
| Mortgage Contract | 0 | 4 | 0.000 | — | 0.000 |
| Pay Slip | 3 | 3 | 1.000 | 1.000 | 1.000 |
| Resume | 1 | 1 | 1.000 | 1.000 | 1.000 |
| Settlement Statement | 2 | 2 | 1.000 | 1.000 | 1.000 |
| Tax Document | 10 | 9 | 1.000 | 0.900 | 0.947 |

Accuracy 92.9% over 56 files; macro F1 0.9439. Macro F1 weights every class equally, so a class that is frequently predicted but never correct drags it down while accuracy stays high — that gap is the signal to look for.

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
