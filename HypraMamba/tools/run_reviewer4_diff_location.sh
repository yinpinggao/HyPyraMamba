#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXP_NAME="${EXP_NAME:-RUNS_REVIEWER4_DIFF_LOCATION_20260820}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"

# B is intentionally omitted: the paper-reported latent-difference result is reused.
TASKS=(
  "A|longkou|4|30|10|none|true"
  "A|qingyun|11|100|30|none|true"
  "A|tangdaowan|12|100|30|none|true"
  "C|longkou|4|30|10|raw|true"
  "C|qingyun|11|100|30|raw|true"
  "C|tangdaowan|12|100|30|raw|true"
  "D|longkou|4|30|10|raw|false"
  "D|qingyun|11|100|30|raw|false"
  "D|tangdaowan|12|100|30|raw|false"
  "E|longkou|4|30|10|none|false"
  "E|qingyun|11|100|30|none|false"
  "E|tangdaowan|12|100|30|none|false"
)

run_worker() {
  local physical_gpu="$1"
  local worker_index="$2"
  local worker_count="$3"
  local queue_failed=0
  local task_index

  cd "${REPO_DIR}" || exit 1
  mkdir -p logs/reviewer4_diff_location

  for task_index in "${!TASKS[@]}"; do
    if (( task_index % worker_count != worker_index )); then
      continue
    fi

    local variant dataset_tag dataset_index train_samples val_samples diff_stage use_pca
    IFS='|' read -r variant dataset_tag dataset_index train_samples val_samples diff_stage use_pca \
      <<< "${TASKS[task_index]}"
    local task_name="${variant}_${dataset_tag}"
    local task_log="logs/reviewer4_diff_location/${task_name}_gpu${physical_gpu}.log"

    echo "[$(date '+%F %T')] START ${task_name} on physical GPU ${physical_gpu}"
    if ! conda run --no-capture-output -n gyp_hsi_env python -u train.py \
      --dataset_index "${dataset_index}" \
      --data_set_path ./data \
      --split_dir ./splits/quh_100_30_seed0-9 \
      --exp_name "${EXP_NAME}" \
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
      --use_pca "${use_pca}" \
      --gaussian_sigma 1.0 \
      --stretch_low 2.0 \
      --stretch_high 98.0 \
      --hidden_dim 128 \
      --token_num 4 \
      --group_num 4 \
      --pool_size 2 \
      --high_res_skip none \
      --prca_num_scales 3 \
      --prca_num_layers 2 \
      --prca_num_heads 4 \
      --pyramid_dilation 3 \
      --spectral_diff_stage "${diff_stage}" \
      --spectral_diff_alpha 0.5 \
      --spectral_fusion_scale 1.0 \
      --tile_size 512 \
      --tile_overlap 32 \
      --tile_update_groups 2 \
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
mkdir -p logs/reviewer4_diff_location

echo "Experiment: ${EXP_NAME}"
echo "Tasks: ${#TASKS[@]} (A/C/D/E x LongKou/Qingyun/Tangdaowan; B omitted)"
echo "Physical GPUs: ${GPU_ARRAY[*]}"
echo "Each GPU runs one sequential background queue."

worker_count="${#GPU_ARRAY[@]}"
if [[ "${1:-}" == "--plan" ]]; then
  for task_index in "${!TASKS[@]}"; do
    worker_index=$((task_index % worker_count))
    IFS='|' read -r variant dataset_tag dataset_index train_samples val_samples diff_stage use_pca \
      <<< "${TASKS[task_index]}"
    echo "GPU ${GPU_ARRAY[worker_index]} <- ${variant}_${dataset_tag} "\
"(dataset_index=${dataset_index}, train/val=${train_samples}/${val_samples}, "\
"stage=${diff_stage}, use_pca=${use_pca})"
  done
  exit 0
fi

for worker_index in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[worker_index]}"
  queue_log="logs/reviewer4_diff_location/queue_gpu${gpu}.log"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" bash "${BASH_SOURCE[0]}" \
    --worker "${gpu}" "${worker_index}" "${worker_count}" \
    > "${queue_log}" 2>&1 &
  echo "GPU ${gpu}: PID $! log=${queue_log}"
done

echo "All queues launched. Monitor with:"
echo "  tail -f logs/reviewer4_diff_location/queue_gpu*.log"
