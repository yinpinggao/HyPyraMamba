# Tree Species HSI 2026 experiment report

## Current conclusion

The mandatory P0 alignment gate passed. M1 did not reach the original OA 0.70
gate, but the user explicitly cancelled that gate and authorized parallel
continuation through M2-M5 plus one Kaggle submission per model. Public scores
are recorded only and are not used for tuning.

## P0 data alignment

- Validation truth-loop: `690829 / 690829 = 100%`.
- Train/validation labeled-pixel overlap: `0`.
- Test storage layouts were verified as `(98, W, H)` and converted explicitly
  to `(H, W, 98)`.
- Sample scene boundary was verified at global row `13,989,728`.
- Sample schema is actually lowercase `id,label`; the submission writer derives
  column names from the sample and asserts exact ID order, label range, and row
  count after reading its own output back.
- Full evidence and the ID mapping formula are in `data_alignment_report.md`.

## Model source policy

Following the user correction, formal model architectures are loaded from the
authors' open-source repositories rather than locally reimplemented. All five
repositories and pinned commits are recorded in `third_party/SOURCES.md`.

For M1, the adapter parses and instantiates the exact `HybridSN` class from the
author's released `HybridSN.ipynb`. Local code only adapts the input/output and
converts the released linear layers to an exactly equivalent convolutional
evaluation path for dense validation. The dense and patch paths matched to a
maximum absolute difference of `5.96e-08` in the regression check.

## M1 HybridSN seed 0

| item | value |
|---|---:|
| upstream commit | `01b3ba919d1b3d8d15b132ff3e804ea0488ca08f` |
| official train samples | 170, exactly 10 per class |
| preprocessing | train-mask-only band standardization, PCA-30 with whitening |
| patch size | 25 x 25 |
| best epoch | 141 |
| validation OA | 0.555075 |
| validation AA | 0.682714 |
| validation Kappa | 0.514531 |
| evaluated validation pixels | 690,829 |
| parameters | 5,122,305 |
| training time | 609.2 seconds |
| best validation inference time | 3.89 seconds |
| original gate result | cancelled by explicit user instruction |

The weakest validation classes at the best checkpoint were class 1
(`0.2134`), class 12 (`0.2680`), class 10 (`0.4298`), and class 2 (`0.4781`).
Training loss had already fallen close to zero while validation OA peaked at
0.5551, which is consistent with severe overfitting under 10 labeled pixels per
class. No hyperparameter was changed after observing validation results.

## Failed launch attempts

Two launch-wrapper failures occurred before the successful rank processes began:

1. A user-level Python 3.10 `torchrun` executable was selected outside the
   conda environment and could not import NumPy. No training step ran.
2. The `conda run` wrapper exited during the initial NCCL barrier. No training
   step ran.

The successful attempt used the environment's absolute Python executable. Its
launcher was later reaped by the execution host, so the two already-connected
rank processes logged TCPStore heartbeat warnings while continuing valid NCCL
training. Both ranks completed epoch 161, wrote the checkpoint and summary, and
exited. The warning-heavy log and both pre-training failure logs are retained.

## Upstream weights

- HyperSIGMA Spatial and Spectral ViT-B official checkpoints were downloaded
  from `WHU-Sigma/HyperSIGMA` and SHA-256 hashed; see `third_party/SOURCES.md`.
- The author HyperSIGMA dual-branch ViT-B adapter loaded 299 compatible
  pretrained tensors, including bicubic spatial position interpolation and the
  documented 98-band patch-projection adaptation. CUDA smoke produced
  `[1,17]` from `[1,98,33,33]` with gradient checkpointing enabled upstream.
- M1-M4 author repositories do not provide a checkpoint compatible with this
  98-band, 17-class competition; their author code snapshots are present.

## Continuation status

M1 and M2 have completed all three seeds. M2 results were:

| seed | best epoch | OA | AA | Kappa |
|---:|---:|---:|---:|---:|
| 0 | 26 | 0.635148 | 0.718266 | 0.600170 |
| 1 | 60 | 0.685251 | 0.719308 | 0.651129 |
| 2 | 40 | 0.678825 | 0.720383 | 0.645567 |

Seed 1 was selected strictly by local validation OA. Six-view TTA probabilities
were written for both test scenes, `submissions/m2_seed1.csv` passed the full
streaming readback assertions, and Kaggle submission `55007617` completed with
public score `0.02691`. That score is record-only and caused no mapping search,
hyperparameter change, or model-selection decision.

The first parallel M3 launch was audited and found to have `patience=30`, which
violated the required shared value 20. Seed 1 had already crossed that stopping
boundary, so none of those checkpoints are eligible for selection or
submission. They were preserved (not deleted) under
`outputs/m3_invalid_patience30/` with logs in
`logs/invalid_patience30/`. `configs/m3.yaml` was corrected to 20 and all three
formal seeds were restarted concurrently on GPU pairs 2-3, 4-5, and 0-1.
M4 uses the pinned author MambaHSI and the user-selected 512-pixel tile with
32-pixel overlap; a strict-deterministic 512 training-step smoke passed. The
98-band no-PCA caches use the same 170-pixel train-mask statistics as M1 and do
not access validation/test statistics. Formal M4 seed 0 is running as an
independent four-rank DDP group on GPUs 0-3 concurrently with the lightweight
M3 jobs; the two 40GB GPUs are not part of this DDP group.

The first formal M4 epoch under FP16 autocast produced a finite validation map
but `NaN` training loss. That run was stopped immediately and preserved under
`outputs/m4_invalid_fp16_nan/` and `logs/invalid_fp16_nan/`. Holding all other
settings fixed, BF16 completed five consecutive strict-deterministic
forward/backward/optimizer steps on real 512 tiles with finite decreasing
losses (`2.8625 -> 2.7448`) and 5.72 GiB peak allocation. The formal seed 0 run
was therefore restarted with the single recorded change `amp_dtype=bfloat16`.

M5 head-only warmup completed a strict-deterministic backward smoke. Full
backbone fine-tuning reaches the author deformable-attention
`grid_sampler_2d_backward_cuda` kernel, for which PyTorch provides no
deterministic implementation. Formal M5 is paused pending the user's explicit
choice between strict determinism with a frozen backbone and allowing that
single upstream non-deterministic backward operation after epoch 5.
