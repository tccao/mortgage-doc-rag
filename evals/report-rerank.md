# Evaluation report

- Run: 2026-07-19 23:21:27 | mode: classical | backend: none | model: none | device: cpu | commit: 15c0336
- Corpus: 64 clean + 64 degraded PDFs, 3 adversarial bundles, 13 doc types
- Layer runtime: rag 92.4s

| Layer | Metric | Value |
|---|---|---|
| Retrieval | hit@k / MRR over 29 cases | 100.0% / 0.862 [95% CI 0.883–1.000] |
| Retrieval (dense, truncated to top_n) | hit / MRR | 100.0% / 0.862 |
| Retrieval (post-rerank, same depth) | hit / MRR | 89.7% / 0.8276 |
| Reranker contribution | MRR delta at equal depth | -0.0344 (2 improved, 4 worsened) |
| Truncation cost (top_k→top_n) | gold docs dropped | 0 of 29 |
