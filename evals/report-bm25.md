# Evaluation report

- Run: 2026-07-19 23:21:11 | mode: classical | backend: none | model: none | device: cpu | commit: 15c0336
- Corpus: 64 clean + 64 degraded PDFs, 3 adversarial bundles, 13 doc types
- Layer runtime: rag 14.2s

| Layer | Metric | Value |
|---|---|---|
| Retrieval [bm25] | hit@k / MRR over 29 cases | 89.7% / 0.7839 [95% CI 0.736–0.964] |
