#!/usr/bin/env bash
# 低标注量敏感性实验 K in {5,10,30,50,100}
#   LongKou      : 新跑 5,10,50,100 (K=30  沿用论文 Table II)
#   QUH-Qingyun  : 新跑 5,10,30,50  (K=100 沿用论文 Table III)
#   QUH-Tangdaowan: 新跑 5,10,30,50 (K=100 沿用论文 Table IV)
# 除 train_samples/val_samples 外，全部超参与主实验一致；val = min(K,30)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPU_IDS="${GPU_IDS:-1,2,3,4}"
SEEDS="${SEEDS:-0,1,2}"

# 任务顺序按 轻/中/重 交替，使 4 worker 的 modulo 分派天然均衡
# K|val|tag|dataset_index|need_fixed_split
TASKS=(
  "5|5|longkou|4|false"      "5|5|qingyun|11|true"      "5|5|tangdaowan|12|true"
  "10|10|longkou|4|false"    "10|10|qingyun|11|true"    "10|10|tangdaowan|12|true"
  "50|30|longkou|4|false"    "30|30|qingyun|11|true"    "30|30|tangdaowan|12|true"
  "100|30|longkou|4|false"   "50|30|qingyun|11|true"    "50|30|tangdaowan|12|true"
)
# QUH 需要预先生成的划分档位
QUH_LEVELS=("5 5" "10 10" "30 30" "50 30")

cd "${REPO_DIR}" || exit 1
mkdir -p logs/sample_level

if [[ "${1:-}" != "--worker" ]]; then
  for kv in "${QUH_LEVELS[@]}"; do
    set -- $kv
    out="./splits/quh_$1_$2_seed0-2"
    if [[ ! -d "${out}" ]]; then
      echo "[split] ${out}"
      conda run --no-capture-output -n gyp_hsi_env python tools/export_quh_splits.py \
        --out_dir "${out}" --train_samples "$1" --val_samples "$2" --seed_list "${SEEDS}"
    fi
  done
fi

run_worker() {
  local gpu="$1" idx="$2" total="$3" rc=0 i
  cd "${REPO_DIR}" || exit 1
  for i in "${!TASKS[@]}"; do
    (( i % total != idx )) && continue
    local K V tag dsidx fixed split_arg log
    IFS='|' read -r K V tag dsidx fixed <<< "${TASKS[i]}"
    log="logs/sample_level/K${K}_${tag}.log"
    if [[ "${fixed}" == "true" ]]; then
      split_arg="./splits/quh_${K}_${V}_seed0-2"
    else
      split_arg="./splits/quh_100_30_seed0-9"   # LongKou 走随机采样，不读此文件
    fi
    echo "[$(date '+%F %T')] START K=${K} ${tag} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" conda run --no-capture-output -n gyp_hsi_env python -u train.py \
      --dataset_index "${dsidx}" \
      --data_set_path ./data \
      --split_dir "${split_arg}" \
      --exp_name "RUNS_SAMPLE_LEVEL_K${K}" \
      --train_samples "${K}" --val_samples "${V}" \
      --seed_list "${SEEDS}" \
      --max_epoch 200 \
      --tile_size 512 --tile_overlap 32 --tile_update_groups 2 \
      --optimizer adam --scheduler none --lr 0.0003 --weight_decay 1e-5 \
      --label_smoothing 0.05 --class_weight_mode balanced \
      --checkpoint_metric oa --checkpoint_tie_break secondary \
      --evaluate_test true \
      --pca_components 30 --use_pca true \
      --gaussian_sigma 1.0 --stretch_low 2.0 --stretch_high 98.0 \
      --hidden_dim 128 --token_num 4 --group_num 4 --pool_size 2 \
      --high_res_skip none --cls_head_dim 128 \
      --prca_num_scales 3 --prca_num_layers 2 --prca_num_heads 4 \
      --pyramid_dilation 3 --spectral_diff_alpha 0.5 --spectral_fusion_scale 1.0 \
      --record_computecost false --save_vis false \
      > "${log}" 2>&1 || { rc=1; echo "[$(date '+%F %T')] FAILED K=${K} ${tag} -> ${log}"; }
    echo "[$(date '+%F %T')] DONE  K=${K} ${tag}"
  done
  return "${rc}"
}

if [[ "${1:-}" == "--worker" ]]; then run_worker "$2" "$3" "$4"; exit $?; fi

IFS=',' read -r -a G <<< "${GPU_IDS}"
echo "tasks=${#TASKS[@]} gpus=${G[*]} seeds=${SEEDS}"
for i in "${!G[@]}"; do
  nohup bash "$0" --worker "${G[i]}" "${i}" "${#G[@]}" \
    > "logs/sample_level/queue_gpu${G[i]}_w${i}.log" 2>&1 &
done
wait
echo "ALL DONE"
