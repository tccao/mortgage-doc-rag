#!/usr/bin/env bash
# Reference baselines + ablation grid over the same golden set.
#
# The committed baselines answer "did I break anything?". These answer "does the
# complexity earn its place?" — a question no regression baseline can address.
#
# Every arm writes evals/report-<tag>.md and evals/baselines/<tag>.json, so arms
# never overwrite the shipped baseline and can be diffed against each other.
#
#   ./scripts/run_ablations.sh retrieval    # no LLM: retrieval-layer arms only
#   ./scripts/run_ablations.sh answer       # full arms, needs an LLM backend
#
# Retrieval-only arms are CPU-cheap. Answer arms cost one generation pass each,
# so run them when a GPU session is already open.
set -euo pipefail

MODE="${1:-retrieval}"
RUN="uv run python evals/run_eval.py"

if [[ "$MODE" == "retrieval" ]]; then
  LAYERS="--layers retrieval"
else
  LAYERS="--layers retrieval,answer"
fi

run_arm() {
  local tag="$1"; shift
  echo ""
  echo "===== arm: ${tag} ====="
  # shellcheck disable=SC2086
  $RUN $LAYERS --tag "$tag" --save-baseline "$@" || echo "  (arm ${tag} failed — continuing)"
}

# --- reference baselines: is the pipeline better than something simpler? ---
run_arm dense-default
run_arm bm25 --retriever bm25

if [[ "$MODE" == "answer" ]]; then
  # No context at all. Any case that still passes was answered from the model's
  # weights, not the documents — the corpus is public CFPB/IRS forms, so this is
  # the check that separates retrieval from memorization.
  run_arm no-retrieval --retriever none
fi

# --- ablations: which components pay rent? ---
run_arm no-reranker      --set use_reranker=false
run_arm no-overlap       --set chunk_overlap=0
run_arm chunk-250        --set chunk_size=250
run_arm chunk-1000       --set chunk_size=1000
run_arm semantic-chunks  --set use_semantic_chunking=true
run_arm topk-10          --set top_k=10

echo ""
echo "Done. Compare arms:"
echo "  grep -H 'hit@k\\|pass rate' evals/report-*.md"
