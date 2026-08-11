#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_bida_tree_species.sh scene1 0
#        bash scripts/run_bida_tree_species.sh scene2 0
SCENE="${1:?scene1 or scene2 is required}"
GPU="${2:-0}"
if [[ "${SCENE}" != "scene1" && "${SCENE}" != "scene2" ]]; then
  echo "SCENE must be scene1 or scene2" >&2
  exit 2
fi

ROOT="/data2/gyp/HyPyraMamba/tree-hsi-2026"
BIDA_DIR="${ROOT}/third_party/IEEE_TCSVT_BiDA"
DATA_DIR="/data2/gyp/HyPyraMamba/HypraMamba/data/TreeSpeciesHSI"
PYTHON="/home/guest/anaconda3/envs/gyp_hsi_env/bin/python"

cd "${BIDA_DIR}"
exec "${PYTHON}" main.py \
  --model BiDA \
  --source_name TreeSpeciesHSI \
  --target_name "TreeSpeciesHSI_${SCENE}" \
  --dataset_dir "${DATA_DIR}" \
  --cache_dir "${ROOT}/data/cache/bida" \
  --output_dir "${ROOT}/outputs/bida_v2" \
  --device "${GPU}" \
  --epoch 200 \
  --bs 128 \
  --val_bs 1024 \
  --lr 1e-2 \
  --patch_size 13 \
  --depth 3 \
  --num_tokens 4 \
  --mmd_start_epoch 100 \
  --mmd_start_step 512 \
  --adapt_warmup_steps 320 \
  --samples_per_epoch 4096 \
  --quick_val_per_class 2048 \
  --full_val_interval 10 \
  --min_active_classes 8 \
  --max_prediction_fraction 0.90 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --lambda1 1e-1 \
  --lambda2 1e0 \
  --norm normband \
  --seed 2100
