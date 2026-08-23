#!/usr/bin/env bash
# Fill-in run, fully parallel: g2/g8/g16 x QUH-Qingyun/QUH-Tangdaowan x
# seeds 0,1,2 = 18 single-seed processes spread over GPUs 0-4
# (4/4/4/3/3). train.py names run dirs as run<pos-in-seed_list>_seed<seed>,
# so single-seed processes never collide. Results land in the same
# RUNS_DGS_SENS_10RUN_20260823_* dirs; summarize_dgs_partial.py picks them up.
# Note: mean_result.txt in the dataset dir only reflects that process's own
# seed and is ignored by the partial summarizer.
set -uo pipefail

REPO_DIR="/data2/gyp/HyPyraMamba/HypraMamba"
cd "${REPO_DIR}" || exit 1
mkdir -p logs/dgs_sensitivity

EXP_PREFIX=RUNS_DGS_SENS_10RUN_20260823
GPU_MAP=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2)

# build task list: tag|token_num|dataset_index|seed
TASKS=()
for spec in "g2|2" "g8|8" "g16|16"; do
  IFS='|' read -r tag token_num <<< "${spec}"
  for dataset_index in 11 12; do
    for seed in 0 1 2; do
      TASKS+=("${tag}|${token_num}|${dataset_index}|${seed}")
    done
  done
done

echo "[$(date '+%F %T')] Launching ${#TASKS[@]} single-seed processes"
pids=()
for i in "${!TASKS[@]}"; do
  IFS='|' read -r tag token_num dataset_index seed <<< "${TASKS[$i]}"
  gpu=${GPU_MAP[$i]}
  ds_name=$([[ "${dataset_index}" == "11" ]] && echo qingyun || echo tangdaowan)
  log="logs/dgs_sensitivity/${tag}_${ds_name}_seed${seed}_gpu${gpu}.log"
  echo "  ${tag}_${ds_name} seed${seed} -> GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup conda run --no-capture-output -n gyp_hsi_env \
    python -u train.py \
    --dataset_index "${dataset_index}" \
    --data_set_path ./data \
    --split_dir ./splits/quh_100_30_seed0-9 \
    --exp_name "${EXP_PREFIX}_${tag}" \
    --train_samples 100 --val_samples 30 \
    --seed_list "${seed}" \
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
echo "[$(date '+%F %T')] All 18 fill-in processes finished (fail=${fail}). Running partial summary."
conda run --no-capture-output -n gyp_hsi_env python tools/summarize_dgs_partial.py
