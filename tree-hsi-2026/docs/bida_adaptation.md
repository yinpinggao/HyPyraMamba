# BiDA adaptation for Tree Species HSI 2026

The author repository is pinned under
`third_party/IEEE_TCSVT_BiDA` at commit
`66a92b542ffa1df3c8b628af3d8bd001b799f1f6`.

## Verified data contract

- Source HDF5 key/shape: `data`, `(98, 4040, 2444)` -> HWC `(4040, 2444, 98)`.
- Train/validation keys: `train_label`, `val_label`; both are `(4040, 2444)`
  `uint8` maps with values `0..17`.
- Scene 1 key/shape: `image`, `(98, 4507, 3104)` -> HWC `(3104, 4507, 98)`.
- Scene 2 key/shape: `image`, `(98, 2181, 3409)` -> HWC `(3409, 2181, 98)`.
- The internal BiDA convention is `-1` for ignored background and `0..16` for
  the 17 foreground classes. Submission converts predictions back to `1..17`.
- The sample columns are exactly `id,label`; it contains 21,424,757 data rows.

## Adaptation changes

- `utils/dataset.py`: verified v7.3/HDF5 loading, blockwise L2 normalization,
  reusable `.npy` memory-mapped caches, official train/validation label loading,
  all-pixel unlabeled target sampling, and edge-safe patch extraction.
- `models/BiDA.py`: dynamic 98-band/17-class construction and an efficient
  target-only inference path.
- `main.py`: official train/validation split, separate target-scene runs,
  randomly sampled unlabeled target patches, and configurable output/cache paths.
- `train_pipeline.py`: per-epoch source/adaptation losses and validation OA/AA,
  source-validation checkpoint selection, readable JSON/CSV artifacts, and the
  corrected non-trivial second MMD comparison.
- `infer_tree_species.py`: two-checkpoint dense inference and streaming,
  sample-ordered submission generation with full ID/count/range checks.
- `loss/make_loss.py`: avoids constructing the unused CUDA center loss for the
  default softmax-only configuration.

## Training

Run the scenes independently; do not concatenate their target pixels:

```bash
cd /data2/gyp/HyPyraMamba/tree-hsi-2026
bash scripts/run_bida_tree_species.sh scene1 0
bash scripts/run_bida_tree_species.sh scene2 0
```

The revised anti-collapse protocol uses 4,096 balanced source samples and 4,096
random target samples per epoch, producing 32 optimizer steps instead of two.
Domain-adaptation losses ramp up over the first 320 steps and MMD begins at step
512. Every epoch evaluates a fixed balanced subset of 34,816 validation pixels;
the complete 690,829-pixel validation map is evaluated every 10 epochs and is
the only source of checkpoint-selection metrics. A checkpoint is rejected when
fewer than eight classes are predicted or one class occupies more than 90% of
the predictions.

The first run creates float32 normalized memory-map caches. Training writes to:

```text
outputs/bida_v2/BiDA/TreeSpeciesHSItoTreeSpeciesHSI_scene1/
outputs/bida_v2/BiDA/TreeSpeciesHSItoTreeSpeciesHSI_scene2/
```

Each directory contains `model_ts_best.pth`, `best_metrics.json`,
`best_confusion_matrix.csv`, `history.csv`, and `results_best.mat`.

## Inference and submission

```bash
cd /data2/gyp/HyPyraMamba/tree-hsi-2026
bash scripts/infer_bida_tree_species.sh 0
```

The final file is `submissions/bida_v2_scene1_scene2.csv`. The writer validates
all 21,424,757 sample IDs in their original order and enforces labels `1..17`.
