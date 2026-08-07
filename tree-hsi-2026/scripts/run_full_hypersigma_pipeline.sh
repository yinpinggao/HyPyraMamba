#!/usr/bin/env bash
set -euo pipefail
cd /data2/gyp/HyPyraMamba/tree-hsi-2026
log=outputs/m5_pipeline_supervisor.log
exec > >(tee -a "$log") 2>&1
echo "supervisor_start $(date -Is)"
while [[ ! -f outputs/m5_max_strict/seed0/best.pt || ! -f outputs/m5_max_strict/seed0/full_val_metrics.json ]]; do
  sleep 30
done
echo "strict_ready $(date -Is)"
CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n gyp_hsi_env python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/train_hypersigma_dense.py --config configs/m5_max_refit90.yaml
while [[ ! -f outputs/m5_max_refit90/seed0/best.pt ]]; do sleep 30; done
echo "refit90_ready $(date -Is)"
CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n gyp_hsi_env python -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/train_hypersigma_dense.py --config configs/m5_max_refit_all.yaml
while [[ ! -f outputs/m5_max_refit_all/seed0/best.pt ]]; do sleep 30; done
echo "refit_all_ready $(date -Is)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run -n gyp_hsi_env python -m torch.distributed.run --standalone --nproc_per_node=6 \
  src/infer_hypersigma_dense.py --config configs/m5_max_refit_all.yaml \
  --checkpoint outputs/m5_max_refit_all/seed0/best.pt \
  --output-dir outputs/m5_max_refit_all/seed0 \
  --tta identity,hflip,vflip,rot90,rot180,rot270
conda run -n gyp_hsi_env python src/submit.py \
  --sample ../HypraMamba/data/TreeSpeciesHSI/sample_submission.csv \
  --scene-info ../HypraMamba/data/TreeSpeciesHSI/scene_info.csv \
  --probs-dir outputs/m5_max_refit_all/seed0 \
  --output submissions/hypersigma_dense_max_refit_all.csv
kaggle competitions submit -c tree-species-hsi-2026 \
  -f submissions/hypersigma_dense_max_refit_all.csv \
  -m "HyperSIGMA dense dual-view MAE-domain-adapt full-refit"
kaggle competitions submissions -c tree-species-hsi-2026
echo "supervisor_done $(date -Is)"
