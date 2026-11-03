#!/usr/bin/env bash
#SBATCH --job-name=geo-alg-ti-llms
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm_logs/%x-%j.out
# One (experiment, item set, test, model) run. The launcher (submit.sh) sets EXPERIMENT, RUNNER
# (direct|ladder), ITEM_SET, TEST, MODEL and OUT, and passes --mem/--time per model; GEN_BATCH
# (the ladder batch, 16 by default) may be set in the environment.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}"
source .env
source "$VENV/bin/activate"
mkdir -p "$OUT"
echo "host $(hostname)  $(date -Iseconds)  $EXPERIMENT $ITEM_SET $TEST $MODEL -> $OUT"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
case "$RUNNER" in
  direct) geo-alg-ti-llms run-direct --experiment "$EXPERIMENT" --item-set "$ITEM_SET" --test "$TEST" --model "$MODEL" --out "$OUT" ;;
  ladder) geo-alg-ti-llms run-ladder --experiment "$EXPERIMENT" --item-set "$ITEM_SET" --test "$TEST" --model "$MODEL" --out "$OUT" \
            --gen-batch-size "${GEN_BATCH:-16}" ;;
  *) echo "unknown RUNNER $RUNNER" >&2; exit 2 ;;
esac
echo "done $(date -Iseconds)"
