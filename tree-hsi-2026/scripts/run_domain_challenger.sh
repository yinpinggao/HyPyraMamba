#!/usr/bin/env bash
set -euo pipefail

cd /data2/gyp/HyPyraMamba/tree-hsi-2026
python_bin="${HSI_PYTHON_BIN:-/home/guest/anaconda3/envs/gyp_hsi_env/bin/python}"
phase="${1:-prepare}"
data_dir="${HSI_DATA_DIR:-/data2/gyp/HyPyraMamba/HypraMamba/data/TreeSpeciesHSI}"
cache_dir="${HSI_DOMAIN_CACHE_DIR:-data/cache/hypersigma_source_scenealign}"
fold_dir="${HSI_DOMAIN_FOLD_DIR:-outputs/domain_validation/component512_buffer64}"

case "$phase" in
  prepare)
    "$python_bin" verify_pipeline.py
    "$python_bin" scripts/prepare_domain_validation.py \
      --data-dir "$data_dir" --output-dir "$fold_dir" \
      --folds 5 --macro-block 512 --buffer 64 --seed 2026
    "$python_bin" src/preprocess_hypersigma_domain.py \
      --data-dir "$data_dir" --output-dir "$cache_dir" \
      --target-align quantile
    "$python_bin" scripts/make_domain_challenger_configs.py \
      --fold-dir "$fold_dir" --cache-dir "$cache_dir"
    ;;
  adapt)
    gpu_ids="${HSI_DOMAIN_GPUS:-0,1}"
    weight_dir="outputs/m5_domain_source_scenealign"
    mkdir -p "$weight_dir"
    for branch in spatial spectral; do
      if [[ "$branch" == "spatial" ]]; then
        suffix=domain_pca30
        source_ckpt=weights/hypersigma/spat-vit-base-ultra-checkpoint-1599.pth
        output="$weight_dir/spatial_mae.pt"
      else
        suffix=domain_native98
        source_ckpt=weights/hypersigma/spec-vit-base-ultra-checkpoint-1599.pth
        output="$weight_dir/spectral_mae.pt"
      fi
      CUDA_VISIBLE_DEVICES="$gpu_ids" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
        src/adapt_hypersigma_mae.py --branch "$branch" \
        --checkpoint "$source_ckpt" --output "$output" \
        --cache-path "$cache_dir/train_${suffix}_float16.npy" \
        --cache-path "$cache_dir/scene1_${suffix}_float16.npy" \
        --cache-path "$cache_dir/scene2_${suffix}_float16.npy"
    done
    "$python_bin" scripts/make_domain_challenger_configs.py \
      --fold-dir "$fold_dir" --cache-dir "$cache_dir"
    ;;
  fold)
    fold="${HSI_DOMAIN_FOLD:?Set HSI_DOMAIN_FOLD=0..4}"
    gpu_ids="${HSI_DOMAIN_GPUS:-0,1}"
    branch="${HSI_DOMAIN_BRANCH:-challenger}"
    config="configs/domain_${branch}/fold${fold}.yaml"
    CUDA_VISIBLE_DEVICES="$gpu_ids" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
      src/train_hypersigma_dense.py --config "$config"
    ;;
  all-folds)
    gpu_ids="${HSI_DOMAIN_GPUS:-0,1}"
    for fold in 0 1 2 3 4; do
      HSI_DOMAIN_FOLD="$fold" HSI_DOMAIN_GPUS="$gpu_ids" \
        bash scripts/run_domain_challenger.sh fold
    done
    ;;
  summarize)
    branch="${HSI_DOMAIN_BRANCH:-challenger}"
    if [[ "$branch" == "control" ]]; then
      "$python_bin" scripts/summarize_domain_folds.py \
        --run-prefix m5_component_control_fold --summary-only \
        --output-summary outputs/domain_validation/control_summary.json
    else
      "$python_bin" scripts/summarize_domain_folds.py \
        --cache-dir "$cache_dir"
    fi
    ;;
  refit)
    gpu_ids="${HSI_DOMAIN_GPUS:-0,1}"
    CUDA_VISIBLE_DEVICES="$gpu_ids" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
      src/train_hypersigma_dense.py --config configs/m5_domain_refit_all.yaml
    ;;
  infer)
    gpu_ids="${HSI_DOMAIN_GPUS:-0,1,2,3,4}"
    gpu_count="${HSI_DOMAIN_GPU_COUNT:-}"
    if [[ -z "$gpu_count" ]]; then
      gpu_count=1
      rest="$gpu_ids"
      while [[ "$rest" == *,* ]]; do
        rest="${rest#*,}"
        gpu_count=$((gpu_count + 1))
      done
    fi
    CUDA_VISIBLE_DEVICES="$gpu_ids" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "$python_bin" -m torch.distributed.run --standalone \
      --nproc_per_node="$gpu_count" src/infer_hypersigma_dense.py \
      --config configs/m5_domain_refit_all.yaml \
      --checkpoint outputs/m5_domain_refit_all/seed0/best.pt \
      --output-dir outputs/m5_domain_refit_all/seed0
    ;;
  csv)
    "$python_bin" src/submit.py \
      --sample "$data_dir/sample_submission.csv" \
      --scene-info "$data_dir/scene_info.csv" \
      --probs-dir outputs/m5_domain_refit_all/seed0 \
      --output submissions/hypersigma_domain_challenger.csv
    ;;
  *)
    echo "Usage: $0 {prepare|adapt|fold|all-folds|summarize|refit|infer|csv}" >&2
    exit 2
    ;;
esac
