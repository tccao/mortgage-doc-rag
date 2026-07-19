# AI-assisted development

This project was built with an AI coding assistant (Claude Code) directed by me. What that
means concretely, and how AI output was verified:

## Division of labor

- **Mine:** system design (pipeline stages, orchestrator/backend seams, eval layer design),
  all acceptance criteria, the decision log in `docs/design.md`, golden-set case selection,
  data-source vetting (public-domain status of every corpus file), and every merge decision.
- **AI-assisted:** module scaffolding from my earlier notebook implementation, boilerplate
  (CLI parsing, report rendering), test drafting, and data-collection scripts.

## Verification, not trust

Nothing shipped on the assistant's say-so:

- Every module is covered by deterministic tests (`tests/`) that I reviewed case-by-case.
- The eval harness scores the pipeline against frozen, human-verifiable references — the same
  harness that gates my own changes gates AI-written changes.
- The harness caught a real AI-era failure mode during development: the inherited image
  preprocessing silently destroyed OCR on degraded scans (ADR-2). It was found by the CER
  layer, not by reading the code.
- Corpus provenance is auditable without trusting anyone: `data/manifest.csv` lists source
  URLs (all US-government works) with SHA-256 checksums.

## Why say this at all

Evaluating AI output is the core skill this project practices — the eval harness *is* the
point. Being explicit about the AI-assisted workflow keeps the same standard for the
project's own code that the pipeline applies to model answers: verify against ground truth,
never vibes.
