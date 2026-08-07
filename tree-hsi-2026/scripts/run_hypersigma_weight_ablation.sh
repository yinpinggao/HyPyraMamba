#!/usr/bin/env bash
set -euo pipefail

cd /data2/gyp/HyPyraMamba/tree-hsi-2026

python_bin="${HSI_PYTHON_BIN:-/home/guest/anaconda3/envs/gyp_hsi_env/bin/python}"
noweight_gpus="${HSI_NOWEIGHT_GPUS:-0,1}"
mildweight_gpus="${HSI_MILDWEIGHT_GPUS:-4,5}"
log_dir="outputs/hypersigma_weight_ablation_logs"
mkdir -p "$log_dir"

if [[ ! -x "$python_bin" ]]; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi

for required in \
  outputs/m5_domain_adapt/spatial_mae.pt \
  outputs/m5_domain_adapt/spectral_mae.pt \
  data/cache/train_hsmax_pca30_float16.npy \
  data/cache/train_hsmax_native98_float16.npy; do
  if [[ ! -f "$required" ]]; then
    echo "Required artifact missing: $required" >&2
    exit 1
  fi
done

run_branch() {
  local branch="$1"
  local gpu_ids="$2"
  local refit90_config="$3"
  local refit_all_config="$4"
  local branch_log="$log_dir/${branch}.log"

  echo "[$(date -Is)] ${branch}: refit90 on GPUs ${gpu_ids}"
  CUDA_VISIBLE_DEVICES="$gpu_ids" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -m torch.distributed.run \
      --standalone --nproc_per_node=2 \
      src/train_hypersigma_dense.py --config "$refit90_config" \
      2>&1 | tee -a "$branch_log"

  echo "[$(date -Is)] ${branch}: full refit on GPUs ${gpu_ids}"
  CUDA_VISIBLE_DEVICES="$gpu_ids" \
  OMP_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" -m torch.distributed.run \
      --standalone --nproc_per_node=2 \
      src/train_hypersigma_dense.py --config "$refit_all_config" \
      2>&1 | tee -a "$branch_log"

  echo "[$(date -Is)] ${branch}: complete"
}

run_branch \
  noweight "$noweight_gpus" \
  configs/m5_max_refit90_noweight.yaml \
  configs/m5_max_refit_all_noweight.yaml &
pid_noweight=$!

run_branch \
  mildweight "$mildweight_gpus" \
  configs/m5_max_refit90_mildweight.yaml \
  configs/m5_max_refit_all_mildweight.yaml &
pid_mildweight=$!

status=0
wait "$pid_noweight" || status=1
wait "$pid_mildweight" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "At least one branch failed. Inspect $log_dir/*.log" >&2
  exit "$status"
fi

echo "Both HyperSIGMA weight-ablation branches completed."
echo "Validation metrics:"
echo "  outputs/m5_max_refit90_noweight/seed0/full_val_metrics.json"
echo "  outputs/m5_max_refit90_mildweight/seed0/full_val_metrics.json"
echo "Final checkpoints:"
echo "  outputs/m5_max_refit_all_noweight/seed0/best.pt"
echo "  outputs/m5_max_refit_all_mildweight/seed0/best.pt"
