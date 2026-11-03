#!/usr/bin/env bash
# Launch every (item set, test) job an experiment declares, for a list of models, as separate
# SLURM jobs writing the published results layout under $RUNS_ROOT:
#   <RUNS_ROOT>/<nxhop | cot_sweep>/final_<set key>/<test's stored name>/<model>/*.jsonl
# so that `geo-alg-ti-llms export` reads a re-run exactly as it reads the published runs.
#
#   scripts/slurm/submit.sh <experiment> [model ...]        (default: every model in configs/models)
#   TESTS="iwl budget_iwl" scripts/slurm/submit.sh symbolic_distance qwen3-32b
#   DRY_RUN=1 scripts/slurm/submit.sh content_effects          # print the sbatch lines only
# Memory and wall time are sized by model size; the paper's ladders (one 1,024-token greedy
# trace per trial, batches of 16) took up to 16 h for a 32B model on one 96 GB GPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .env
EXPERIMENT="${1:?experiment name}"; shift
if [[ $# -gt 0 ]]; then MODELS=("$@"); else MODELS=($(ls configs/models | sed 's/\.yaml$//')); fi
mkdir -p slurm_logs
size_of() {
  case "$1" in
    *70b*)                MEM=160G; T_LADDER=24:00:00; T_DIRECT=03:00:00 ;;
    *32b*)                MEM=40G;  T_LADDER=16:00:00; T_DIRECT=02:00:00 ;;
    *14b*|*phi-4*|*20b*)  MEM=34G;  T_LADDER=08:00:00; T_DIRECT=01:00:00 ;;
    *)                    MEM=22G;  T_LADDER=06:00:00; T_DIRECT=01:00:00 ;;
  esac
}
"$VENV/bin/geo-alg-ti-llms" jobs --experiment "$EXPERIMENT" | while IFS=$'\t' read -r runner set_key test stored; do
  if [[ -n "${TESTS:-}" ]] && ! grep -qw "$test" <<< "$TESTS"; then continue; fi
  tree=nxhop; [[ $runner == ladder ]] && tree=cot_sweep
  for model in "${MODELS[@]}"; do
    size_of "$model"; t=$T_DIRECT; [[ $runner == ladder ]] && t=$T_LADDER
    gpus=$(awk '/^n_gpus:/{print $2}' "configs/models/$model.yaml"); gpus="${gpus:-1}"
    out="$RUNS_ROOT/$tree/final_$set_key/$stored/$model"
    cmd=(sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" ${SLURM_QOS:+--qos=$SLURM_QOS}
         ${SLURM_CONSTRAINT:+--constraint=$SLURM_CONSTRAINT} --gres=gpu:$gpus --mem="$MEM" --time="$t"
         --job-name="$EXPERIMENT-$test-$model"
         --export=ALL,EXPERIMENT="$EXPERIMENT",RUNNER="$runner",ITEM_SET="$set_key",TEST="$test",MODEL="$model",OUT="$out"
         scripts/slurm/job.sh)
    if [[ -n "${DRY_RUN:-}" ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
  done
done
