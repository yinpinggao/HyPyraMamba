#!/usr/bin/env bash
# Context-restricted (patch-equivalent) protocol for reviewer comment 2.
#
# The scene is partitioned into non-overlapping PxP blocks that are processed
# independently: training propagates only the blocks that contain labeled
# training pixels, and inference predicts every pixel from the block that
# contains it.  Everything else (splits, epochs, optimizer, loss, update
# schedule) is identical to the main setting, so the only variable is the
# spatial context available to each pixel.
#
# Usage:
#   GPU_IDS=0,1,2,3,4 bash tools/run_context_protocol.sh
#   PS="16 32" DATASETS="qingyun tangdaowan" bash tools/run_context_protocol.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXP_PREFIX="${EXP_PREFIX:-RUNS_CONTEXT_PROTOCOL}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
SEEDS="${SEEDS:-0,1,2}"
PS="${PS:-16 32 64 128}"
DATASETS="${DATASETS:-tangdaowan qingyun longkou}"
CONDA_ENV="${CONDA_ENV:-gyp_hsi_env}"

dataset_spec() {
  # tag -> "dataset_index|train_samples|val_samples|use_fixed_split"
  case "$1" in
    longkou)    echo "4|30|10|false" ;;
    qingyun)    echo "11|100|30|true" ;;
    tangdaowan) echo "12|100|30|true" ;;
    *) return 1 ;;
  esac
}

# Heaviest dataset first so the long jobs start earliest.
TASKS=()
for dataset_tag in ${DATASETS}; do
  if ! dataset_spec "${dataset_tag}" > /dev/null; then
    echo "Unknown dataset tag: ${dataset_tag}" >&2
    exit 2
  fi
  for p in ${PS}; do
    TASKS+=("${dataset_tag}|${p}")
  done
done

run_worker() {
  local physical_gpu="$1"
  local worker_index="$2"
  local worker_count="$3"
  local queue_failed=0
  local task_index

  cd "${REPO_DIR}" || exit 1
  mkdir -p logs/context_protocol

  for task_index in "${!TASKS[@]}"; do
    if (( task_index % worker_count != worker_index )); then
      continue
    fi

    local dataset_tag p
    IFS='|' read -r dataset_tag p <<< "${TASKS[task_index]}"

    local dataset_index train_samples val_samples use_fixed_split
    IFS='|' read -r dataset_index train_samples val_samples use_fixed_split \
      <<< "$(dataset_spec "${dataset_tag}")"

    # Batch equally sized tiles so peak activation stays at or below the
    # 512x512 main-line setting while removing the per-tile launch overhead.
    local tile_batch=$(( 1024 / p ))
    (( tile_batch < 1 )) && tile_batch=1

    local task_name="P${p}_${dataset_tag}"
    local task_log="logs/context_protocol/${task_name}_gpu${physical_gpu}.log"

    local split_args=()
    if [[ "${use_fixed_split}" == "true" ]]; then
      split_args=(--split_dir ./splits/quh_100_30_seed0-9)
    fi

    echo "[$(date '+%F %T')] START ${task_name} (tile_batch=${tile_batch}) on physical GPU ${physical_gpu}"
    if ! CUDA_VISIBLE_DEVICES="${physical_gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
      python -u train.py \
      --dataset_index "${dataset_index}" \
      --data_set_path ./data \
      --exp_name "${EXP_PREFIX}_P${p}" \
      "${split_args[@]}" \
      --train_samples "${train_samples}" \
      --val_samples "${val_samples}" \
      --seed_list "${SEEDS}" \
      --max_epoch 200 \
      --optimizer adam \
      --scheduler none \
      --lr 0.0003 \
      --weight_decay 0.00001 \
      --label_smoothing 0.05 \
      --class_weight_mode balanced \
      --checkpoint_metric oa \
      --checkpoint_tie_break secondary \
      --evaluate_test true \
      --pca_components 30 \
      --gaussian_sigma 1.0 \
      --stretch_low 2.0 \
      --stretch_high 98.0 \
      --hidden_dim 128 \
      --token_num 4 \
      --group_num 4 \
      --pool_size 2 \
      --high_res_skip none \
      --cls_head_dim 128 \
      --prca_num_scales 3 \
      --prca_num_layers 2 \
      --prca_num_heads 4 \
      --pyramid_dilation 3 \
      --spectral_diff_stage latent \
      --spectral_diff_alpha 0.5 \
      --spectral_fusion_scale 1.0 \
      --tile_size "${p}" \
      --tile_overlap 0 \
      --tile_update_groups 2 \
      --tile_batch_size "${tile_batch}" \
      --record_computecost false \
      --save_vis false \
      > "${task_log}" 2>&1; then
      queue_failed=1
      echo "[$(date '+%F %T')] FAILED ${task_name}; see ${task_log}"
    else
      echo "[$(date '+%F %T')] DONE ${task_name}"
    fi
  done

  return "${queue_failed}"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker "$2" "$3" "$4"
  exit $?
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU index." >&2
  exit 2
fi

cd "${REPO_DIR}" || exit 1
mkdir -p logs/context_protocol

echo "Experiment prefix: ${EXP_PREFIX}_P<size>"
echo "Datasets: ${DATASETS}"
echo "Block sizes: ${PS}"
echo "Seeds: ${SEEDS}"
echo "Tasks: ${#TASKS[@]}"
echo "Physical GPUs: ${GPU_ARRAY[*]}"

pids=()
for worker_index in "${!GPU_ARRAY[@]}"; do
  bash "${BASH_SOURCE[0]}" --worker "${GPU_ARRAY[worker_index]}" \
    "${worker_index}" "${#GPU_ARRAY[@]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done

if (( failed )); then
  echo "[$(date '+%F %T')] Some tasks failed; check logs/context_protocol/."
  exit 1
fi
echo "[$(date '+%F %T')] All context-protocol tasks finished."
