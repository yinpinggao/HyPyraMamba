#!/usr/bin/env bash
# DGS-Mamba sensitivity study: spectral difference coefficient alpha and
# spectral group number G (token_num). Each task runs the complete seed list
# (10 runs by default) on one dataset. The paper setting alpha=0.5/G=4 is
# included as an explicit control rather than silently reusing an old result.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXP_PREFIX="${EXP_PREFIX:-RUNS_DGS_SENS_20260823}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4}"
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
DATASETS="${DATASETS:-longkou qingyun tangdaowan}"
CONDA_ENV="${CONDA_ENV:-gyp_hsi_env}"
EVALUATE_TEST="${EVALUATE_TEST:-true}"

# variant_tag|alpha|token_num
VARIANTS=(
  "a0p00|0.0|4"
  "a0p25|0.25|4"
  "a0p50|0.5|4"
  "a0p75|0.75|4"
  "a1p00|1.0|4"
  "g2|0.5|2"
  "g8|0.5|8"
  "g16|0.5|16"
)

dataset_spec() {
  case "$1" in
    longkou)    echo "4|30|10"   ;;
    qingyun)    echo "11|100|30" ;;
    tangdaowan) echo "12|100|30" ;;
    *) echo "unknown dataset tag: $1" >&2; return 1 ;;
  esac
}

TASKS=()
for variant in "${VARIANTS[@]}"; do
  for ds in ${DATASETS}; do
    TASKS+=("${variant}|${ds}")
  done
done

run_worker() {
  local physical_gpu="$1" worker_index="$2" worker_count="$3"
  local queue_failed=0 task_index

  cd "${REPO_DIR}" || exit 1
  mkdir -p logs/dgs_sensitivity

  for task_index in "${!TASKS[@]}"; do
    if (( task_index % worker_count != worker_index )); then continue; fi

    local variant_tag alpha token_num dataset_tag
    IFS='|' read -r variant_tag alpha token_num dataset_tag <<< "${TASKS[task_index]}"

    local dataset_index train_samples val_samples
    IFS='|' read -r dataset_index train_samples val_samples <<< "$(dataset_spec "${dataset_tag}")"

    local task_name="${variant_tag}_${dataset_tag}"
    local task_log="logs/dgs_sensitivity/${task_name}_gpu${physical_gpu}.log"

    echo "[$(date '+%F %T')] START ${task_name} (alpha=${alpha} G=${token_num}) on GPU ${physical_gpu}"
    if ! CUDA_VISIBLE_DEVICES="${physical_gpu}" conda run --no-capture-output -n "${CONDA_ENV}" \
      python -u train.py \
      --dataset_index "${dataset_index}" \
      --data_set_path ./data \
      --split_dir ./splits/quh_100_30_seed0-9 \
      --exp_name "${EXP_PREFIX}_${variant_tag}" \
      --train_samples "${train_samples}" \
      --val_samples "${val_samples}" \
      --seed_list "${SEEDS}" \
      --max_epoch 200 \
      --optimizer adam --scheduler none \
      --lr 0.0003 --weight_decay 0.00001 \
      --label_smoothing 0.05 --class_weight_mode balanced \
      --checkpoint_metric oa --checkpoint_tie_break secondary \
      --evaluate_test "${EVALUATE_TEST}" \
      --pca_components 30 --use_pca true \
      --gaussian_sigma 1.0 --stretch_low 2.0 --stretch_high 98.0 \
      --hidden_dim 128 --group_num 4 --pool_size 2 --high_res_skip none \
      --cls_head_dim 128 \
      --prca_num_scales 3 --prca_num_layers 2 --prca_num_heads 4 --pyramid_dilation 3 \
      --token_num "${token_num}" \
      --spectral_diff_stage latent --spectral_diff_alpha "${alpha}" \
      --spectral_fusion_scale 1.0 \
      --tile_size 512 --tile_overlap 32 --tile_update_groups 2 \
      --record_computecost false --save_vis false \
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
  run_worker "$2" "$3" "$4"; exit $?
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
worker_count="${#GPU_ARRAY[@]}"
if (( worker_count == 0 )); then
  echo "GPU_IDS must contain at least one GPU index." >&2
  exit 2
fi
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index in GPU_IDS: ${gpu}" >&2
    exit 2
  fi
done

if [[ "${1:-}" == "--plan" ]]; then
  echo "Total tasks: ${#TASKS[@]} (${#VARIANTS[@]} variants x $(echo ${DATASETS} | wc -w) datasets), seeds=${SEEDS}, evaluate_test=${EVALUATE_TEST}"
  for task_index in "${!TASKS[@]}"; do
    printf '  gpu=%s  %s\n' "${GPU_ARRAY[$((task_index % worker_count))]}" "${TASKS[task_index]}"
  done
  exit 0
fi

cd "${REPO_DIR}" || exit 1
mkdir -p logs/dgs_sensitivity
if compgen -G "${EXP_PREFIX}_*" > /dev/null; then
  echo "Refusing to mix with an existing sweep: ${EXP_PREFIX}_*" >&2
  echo "Choose a new EXP_PREFIX or archive the existing directories first." >&2
  exit 2
fi
echo "Experiment prefix: ${EXP_PREFIX}"
echo "Tasks: ${#TASKS[@]} (${#VARIANTS[@]} variants x $(echo ${DATASETS} | wc -w) datasets)"
echo "Seeds: ${SEEDS}"
echo "evaluate_test: ${EVALUATE_TEST}"
echo "GPUs: ${GPU_ARRAY[*]}"
for idx in "${!GPU_ARRAY[@]}"; do
  nohup bash "${BASH_SOURCE[0]}" --worker "${GPU_ARRAY[$idx]}" "${idx}" "${worker_count}" \
    > "logs/dgs_sensitivity/queue_gpu${GPU_ARRAY[$idx]}.log" 2>&1 &
  echo "  queue on GPU ${GPU_ARRAY[$idx]} -> pid $!"
done
wait

echo "[$(date '+%F %T')] All task queues finished. Running summary check."
conda run --no-capture-output -n "${CONDA_ENV}" python tools/summarize_dgs_sensitivity.py \
  --prefix "${EXP_PREFIX}" \
  --seeds "${SEEDS}" \
  --evaluate-test "${EVALUATE_TEST}" \
  --output "${EXP_PREFIX}_summary.csv"
