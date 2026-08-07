# Tree Species HSI 2026 — Current Champion

## Authoritative best model

- Model: HyperSIGMA official dense dual-branch adapter.
- Upstream commit: `07e9ea24e3072fcb5c3a92a2bcb8185e43b295b9`.
- Kaggle public OA: `0.16714` (public leaderboard uses only about 3% of the hidden test labels; do not tune against it).
- Final checkpoint: `outputs/m5_max_refit_all_noweight/seed0/best.pt`.
- Final inference probabilities:
  - `outputs/m5_max_refit_all_noweight/seed0/probs_scene1.npy`
  - `outputs/m5_max_refit_all_noweight/seed0/probs_scene2.npy`
- Submitted CSV: `submissions/hypersigma_dense_noweight.csv`.

## Selection evidence

- Selection checkpoint: `outputs/m5_max_refit90_noweight/seed0/best.pt`.
- Full spatial-holdout metrics: `outputs/m5_max_refit90_noweight/seed0/full_val_metrics.json`.
- Local spatial-holdout metrics:
  - OA: `0.7963601558`
  - AA: `0.6751261095`
  - Kappa: `0.7688122199`
  - selected epoch: `40`
- The competition selects by OA first, so this no-class-weight branch supersedes the previous square-root-weight and mild-weight branches.

## Training method

1. Use the official HyperSIGMA dense Spatial MAE + Spectral MAE architecture through `src/models/hypersigma.py`; do not replace it with a local reimplementation.
2. Spatial branch input: transductive robust-normalized PCA30 cache.
3. Spectral branch input: robust-normalized continuous native 98-band cache.
4. Initialize from the retained domain-adapted MAE weights:
   - `outputs/m5_domain_adapt/spatial_mae.pt`
   - `outputs/m5_domain_adapt/spectral_mae.pt`
5. Model selection uses `configs/m5_max_refit90_noweight.yaml`:
   - official train plus 90% of official val for supervision;
   - fixed 128-pixel spatial blocks hold out 10% of val;
   - seed 0, 128x128 tiles, 2048 sampled tiles per epoch;
   - BF16, stochastic augmentation, label smoothing 0.05;
   - no class weights;
   - early selection by spatial-holdout OA.
6. Final training uses `configs/m5_max_refit_all_noweight.yaml`:
   - all 690,999 labeled pixels from train+val;
   - no class weights;
   - automatically inherits the selected 40 epochs;
   - final refit checkpoint is the authoritative model above.
7. Test inference uses six GPUs, 32-pixel tile overlap, and six-view TTA (`identity,hflip,vflip,rot90,rot180,rot270`) via `src/infer_hypersigma_dense.py`.
8. `src/submit.py` is the only CSV writer. It must preserve sample IDs exactly and enforce labels 1–17.

## Reproduction commands

Train the no-weight selection and full-refit branch by using the no-weight half of:

```bash
cd /data2/gyp/HyPyraMamba/tree-hsi-2026
HSI_NOWEIGHT_GPUS=0,1 HSI_MILDWEIGHT_GPUS=4,5 \
  bash scripts/run_hypersigma_weight_ablation.sh
```

For a no-weight-only rerun, execute the two configs sequentially with two homogeneous GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 src/train_hypersigma_dense.py \
  --config configs/m5_max_refit90_noweight.yaml

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/guest/anaconda3/envs/gyp_hsi_env/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 src/train_hypersigma_dense.py \
  --config configs/m5_max_refit_all_noweight.yaml
```

## Artifact policy

- Preserve the authoritative paths listed above, the domain-adaptation weights, source code, configs, raw data, and preprocessing caches.
- Old checkpoints, old test probabilities, aborted experiments, TensorBoard event files, and superseded submission CSVs are not part of the champion lineage and may be deleted.
- Local spatial validation determines model decisions. Public leaderboard scores are recorded only as external evidence.
