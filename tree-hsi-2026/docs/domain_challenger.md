# Cross-region HyperSIGMA challenger

The retained champion remains the no-class-weight HyperSIGMA lineage described
in `AGENTS.md`.  This challenger does not overwrite its checkpoint, caches,
probabilities, or submission.

## Why this branch exists

The original 128-pixel random holdout cut through labelled connected regions:
the audit found 73.55% of holdout pixels in a same-class connected component
that still had supervision, and 90.43% within 64 pixels of a same-class
supervised pixel.  The competition explicitly evaluates cross-region
generalization, so this branch uses whole connected components and a 64-pixel
buffer.

## Artifacts

- `outputs/domain_validation/component512_buffer64/fold{0..4}.npz`: component
  folds with `supervision`, `validation`, and `protected` masks.
- `data/cache/hypersigma_source_scenealign/`: source-fitted robust statistics,
  source-only PCA30, and scene-specific quantile-aligned native98/PCA30 caches.
- `configs/domain_control/fold{0..4}.yaml`: same new folds using the original
  champion preprocessing, for a fair control.
- `configs/domain_challenger/fold{0..4}.yaml`: same folds using source-only
  PCA and scene-specific alignment.

## Reproduction

Prepare both fold sets and caches:

```bash
bash scripts/run_domain_challenger.sh prepare
```

The source-only PCA basis changes the spatial channel semantics, so the old
joint-PCA MAE adapter must not be reused.  Adapt fresh candidate MAE weights
after GPUs are available:

```bash
HSI_DOMAIN_GPUS=0,1 bash scripts/run_domain_challenger.sh adapt
```

Run a control fold or challenger fold after homogeneous GPUs are available:

```bash
HSI_DOMAIN_BRANCH=control HSI_DOMAIN_FOLD=0 HSI_DOMAIN_GPUS=0,1 \
  bash scripts/run_domain_challenger.sh fold

HSI_DOMAIN_BRANCH=challenger HSI_DOMAIN_FOLD=0 HSI_DOMAIN_GPUS=0,1 \
  bash scripts/run_domain_challenger.sh fold
```

Repeat folds 0–4, then summarize.  Only after the challenger has stable
cross-fold evidence should `summarize`, `refit`, `infer`, and `csv` be used.
No Kaggle upload is performed by this workflow.
