#!/usr/bin/env bash
#SBATCH --job-name=geo-alg-ti-llms-interp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
# The mechanistic interventions on Qwen3-32B (experiments/representation, experiments/components).
# They read the stored behavioral runs under $RUNS_ROOT (the trials they re-score) and write
# their run directories beside them. One job per STAGE, in order (submission: REPRODUCE.md):
#   fit        item representations, the rank direction and the probe by layer
#   intervene  erase / invert / write along the rank direction at every hidden state (after fit)
#   score      attention mass of every head onto the context relations
#   ablate     mean-ablate the declared top and random head sets (after score)
# DIM is countries, cities or usstates; MODEL defaults to qwen3-32b.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}"
source .env; source "$VENV/bin/activate"
: "${STAGE:?fit|intervene|score|ablate}"; DIM="${DIM:-countries}"; MODEL="${MODEL:-qwen3-32b}"
export GEO_ALG_TI_LLMS_PUBLISHED_RUNS="$RUNS_ROOT"
case "$STAGE" in
  fit)       geo-alg-ti-llms represent fit       --dim "$DIM" --model "$MODEL" --out "$RUNS_ROOT/rank_intervention/$DIM/$MODEL" ;;
  intervene) geo-alg-ti-llms represent intervene --dim "$DIM" --model "$MODEL" --out "$RUNS_ROOT/rank_intervention/$DIM/$MODEL" ;;
  score)     geo-alg-ti-llms components score    --dim "$DIM" --model "$MODEL" --out "$RUNS_ROOT/head_ablation/$DIM/$MODEL" ;;
  ablate)    geo-alg-ti-llms components ablate   --dim "$DIM" --model "$MODEL" --out "$RUNS_ROOT/head_ablation/$DIM/$MODEL" ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
