#!/usr/bin/env bash
# Train PyS2CF-Mamba (ImprovedMambaHSI full) on Kaggle tree-species-hsi-2026.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs submissions

GPU="${GPU:-0}"
SEEDS="${SEEDS:-0}"
MAX_EPOCH="${MAX_EPOCH:-200}"
EXP_NAME="${EXP_NAME:-RUNS_TreeSpeciesHSI}"

export CUDA_VISIBLE_DEVICES="${GPU}"

python -u train.py \
  --dataset_index 13 \
  --data_set_path ./data \
  --exp_name "${EXP_NAME}" \
  --train_samples 10 \
  --val_samples 0 \
  --seed_list "${SEEDS}" \
  --max_epoch "${MAX_EPOCH}" \
  --tile_size 512 \
  --tile_overlap 32 \
  --tile_update_groups 2 \
  --optimizer adam \
  --scheduler none \
  --lr 0.0003 \
  --weight_decay 1e-5 \
  --label_smoothing 0.05 \
  --class_weight_mode balanced \
  --spectral_diff_alpha 0.5 \
  --checkpoint_metric oa \
  --checkpoint_tie_break secondary \
  --evaluate_test true \
  --save_vis false \
  "$@"
