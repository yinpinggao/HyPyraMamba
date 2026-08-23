#!/usr/bin/env bash
# Parallel launcher for the DGS sensitivity sweep: 15 workers, 3 per GPU on
# GPUs 0-4. Each worker still goes through tools/run_dgs_sensitivity.sh
# --worker so task definition stays in one place. After all workers finish,
# the summary checker runs automatically.
set -uo pipefail

REPO_DIR="/data2/gyp/HyPyraMamba/HypraMamba"
cd "${REPO_DIR}" || exit 1
mkdir -p logs/dgs_sensitivity

GPUS=(0 1 2 3 4)
WORKERS=15
export EXP_PREFIX=RUNS_DGS_SENS_10RUN_20260823
export EVALUATE_TEST=true
export SEEDS=0,1,2,3,4,5,6,7,8,9

echo "[$(date '+%F %T')] Launching ${WORKERS} workers on GPUs ${GPUS[*]} (2 per GPU)"
pids=()
for ((w = 0; w < WORKERS; w++)); do
  gpu=${GPUS[$((w % ${#GPUS[@]}))]}
  nohup bash tools/run_dgs_sensitivity.sh --worker "${gpu}" "${w}" "${WORKERS}" \
    > "logs/dgs_sensitivity/queue_w${w}_gpu${gpu}.log" 2>&1 &
  pids+=($!)
  echo "  worker ${w} -> GPU ${gpu}, pid ${pids[-1]}"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then fail=1; fi
done

echo "[$(date '+%F %T')] All workers finished (fail=${fail}). Running summary check."
conda run --no-capture-output -n gyp_hsi_env python tools/summarize_dgs_sensitivity.py \
  --prefix "${EXP_PREFIX}" \
  --seeds "${SEEDS}" \
  --evaluate-test "${EVALUATE_TEST}" \
  --output "${EXP_PREFIX}_summary.csv"
summary_rc=$?
echo "[$(date '+%F %T')] Summary check exit code: ${summary_rc}"
exit ${summary_rc}
