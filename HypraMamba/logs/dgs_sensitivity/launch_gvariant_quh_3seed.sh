#!/usr/bin/env bash
# Fill-in run: g2/g8/g16 x QUH-Qingyun/QUH-Tangdaowan, seeds 0,1,2 only
# (the alpha sweep tasks already have >=3 seeds; these 6 tasks had 0 when
# the main sweep was stopped). Results go into the same
# RUNS_DGS_SENS_10RUN_20260823_* directories so summarize_dgs_partial.py
# picks them up. Hyperparameters identical to tools/run_dgs_sensitivity.sh.
set -uo pipefail

REPO_DIR="/data2/gyp/HyPyraMamba/HypraMamba"
cd "${REPO_DIR}" || exit 1
mkdir -p logs/dgs_sensitivity

EXP_PREFIX=RUNS_DGS_SENS_10RUN_20260823
SEEDS=0,1,2

# task: variant|token_num|dataset_index|gpu
TASKS=(
  "g2|2|11|0"
  "g2|2|12|1"
  "g8|8|11|2"
  "g8|8|12|3"
  "g16|16|11|4"
  "g16|16|12|0"
)

echo "[$(date '+%F %T')] Launching ${#TASKS[@]} fill-in tasks (seeds=${SEEDS})"
pids=()
for task in "${TASKS[@]}"; do
  IFS='|' read -r tag token_num dataset_index gpu <<< "${task}"
  ds_name=$([[ "${dataset_index}" == "11" ]] && echo qingyun || echo tangdaowan)
  log="logs/dgs_sensitivity/${tag}_${ds_name}_3seed_gpu${gpu}.log"
  echo "  ${tag}_${ds_name} -> GPU ${gpu}, log ${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup conda run --no-capture-output -n gyp_hsi_env \
    python -u train.py \
    --dataset_index "${dataset_index}" \
    --data_set_path ./data \
    --split_dir ./splits/quh_100_30_seed0-9 \
    --exp_name "${EXP_PREFIX}_${tag}" \
    --train_samples 100 --val_samples 30 \
    --seed_list "${SEEDS}" \
    --max_epoch 200 \
    --optimizer adam --scheduler none \
    --lr 0.0003 --weight_decay 0.00001 \
    --label_smoothing 0.05 --class_weight_mode balanced \
    --checkpoint_metric oa --checkpoint_tie_break secondary \
    --evaluate_test true \
    --pca_components 30 --use_pca true \
    --gaussian_sigma 1.0 --stretch_low 2.0 --stretch_high 98.0 \
    --hidden_dim 128 --group_num 4 --pool_size 2 --high_res_skip none \
    --cls_head_dim 128 \
    --prca_num_scales 3 --prca_num_layers 2 --prca_num_heads 4 --pyramid_dilation 3 \
    --token_num "${token_num}" \
    --spectral_diff_stage latent --spectral_diff_alpha 0.5 \
    --spectral_fusion_scale 1.0 \
    --tile_size 512 --tile_overlap 32 --tile_update_groups 2 \
    --record_computecost false --save_vis false \
    > "${log}" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then fail=1; fi
done
echo "[$(date '+%F %T')] All fill-in tasks finished (fail=${fail}). Running partial summary."
conda run --no-capture-output -n gyp_hsi_env python tools/summarize_dgs_partial.py
