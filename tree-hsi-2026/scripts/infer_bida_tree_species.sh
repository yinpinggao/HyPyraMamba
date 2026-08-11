#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT="/data2/gyp/HyPyraMamba/tree-hsi-2026"
BIDA_DIR="${ROOT}/third_party/IEEE_TCSVT_BiDA"
DATA_DIR="/data2/gyp/HyPyraMamba/HypraMamba/data/TreeSpeciesHSI"
PYTHON="/home/guest/anaconda3/envs/gyp_hsi_env/bin/python"

cd "${BIDA_DIR}"
exec "${PYTHON}" infer_tree_species.py \
  --dataset_dir "${DATA_DIR}" \
  --cache_dir "${ROOT}/data/cache/bida" \
  --prediction_dir "${ROOT}/outputs/bida_v2/predictions" \
  --scene1_checkpoint "${ROOT}/outputs/bida_v2/BiDA/TreeSpeciesHSItoTreeSpeciesHSI_scene1/model_ts_best.pth" \
  --scene2_checkpoint "${ROOT}/outputs/bida_v2/BiDA/TreeSpeciesHSItoTreeSpeciesHSI_scene2/model_ts_best.pth" \
  --output "${ROOT}/submissions/bida_v2_scene1_scene2.csv" \
  --device "${GPU}" \
  --batch_size 1024
