# Repository Guidelines

## Project Structure
`train.py` is the only training entry. The model lives in `model/MambaHSI.py`. Data loading, loss, evaluation, logging, and visualization are under `utils/`. Raw `.mat` datasets are read from `data/<DatasetName>/`. Training artifacts go to `RUNS/<model>/<dataset>/run*_seed*/`. Runtime logs go to `logs/`.

## Runtime Environment
Run this repo in the `gyp_hsi_env` conda environment.

The usual command shape is:

```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python train.py --dataset_index <id> > logs/<name>.log 2>&1 &
```

At the moment there is no confirmed long-running `train.py` process from this repo. The current `logs/` directory mainly contains recent experiment logs for `LongKou`, `Salinas`, and `indian`, including variants such as:

- `*_current.log`
- `*_x1_x2.log`
- `*_single_pca_aux01.log`
- `*_dual_raw.log`
- `*_lw_selector.log`

The repo also contains accumulated outputs under `RUNS/`, `RUNS_new/`, and `RUNS_raw/`.

## Current QUH Main Baseline
The strongest and most stable QUH configuration is currently:

- Result directory: `RUNS_QUH_100_30_WEIGHTED_ACCUM_GROUP2`
- Model output folder: `MambaHSI_competitive_diff_alpha0p5`
- Protocol: QUH `100 train / 30 val / rest test` with fixed splits under `splits/quh_100_30_seed0-9`
- Datasets: `dataset_index=10` for `QUH-Pingan`, `11` for `QUH-Qingyun`, and `12` for `QUH-Tangdaowan`
- Preprocessing: spatial/spectral `Gaussian sigma=1.0 -> PCA 30 -> ImageStretching(2,98)`, without PCA whitening
- Tile training: `tile_size=512`, `tile_overlap=32`, `tile_update_groups=2`
- Optimizer: `Adam`, `lr=0.0003`, `weight_decay=1e-5`, `scheduler=none`, `max_epoch=200`
- Loss: `CrossEntropyLoss`, `label_smoothing=0.05`, `class_weight_mode=balanced`
- Model knobs: `hidden_dim=128`, `token_num=4`, `group_num=4`, `pool_size=2`, `high_res_skip=none`, `prca_num_scales=3`, `prca_num_layers=2`, `prca_num_heads=4`, `pyramid_dilation=3`, `spectral_diff_alpha=0.5`
- Checkpoint selection: validation OA unless an experiment explicitly changes `--checkpoint_metric`; new runs use `--checkpoint_tie_break secondary`, so equal OA is resolved by validation mIoU and then the earlier epoch

The verified 10-seed means for this baseline are:

- `QUH-Pingan`: `OA 94.97 ± 0.50 / AA 95.43 ± 0.73 / Kpp 92.61 ± 0.73 / mIOU_test 80.57 ± 1.86`
- `QUH-Qingyun`: `OA 90.70 ± 1.06 / AA 92.25 ± 0.85 / Kpp 87.78 ± 1.38 / mIOU_test 78.20 ± 1.42`
- `QUH-Tangdaowan`: `OA 95.92 ± 0.45 / AA 97.78 ± 0.20 / Kpp 95.39 ± 0.50 / mIOU_test 88.07 ± 2.07`

Use this baseline as the default when the user asks to run the current best QUH setup:

```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python -u train.py \
  --dataset_index <10|11|12> \
  --data_set_path ./data \
  --split_dir ./splits/quh_100_30_seed0-9 \
  --exp_name RUNS_QUH_100_30_WEIGHTED_ACCUM_GROUP2 \
  --train_samples 100 \
  --val_samples 30 \
  --seed_list 0,1,2,3,4,5,6,7,8,9 \
  --tile_size 512 \
  --tile_overlap 32 \
  --tile_update_groups 2 \
  --optimizer adam \
  --scheduler none \
  --lr 0.0003 \
  --weight_decay 1e-5 \
  --label_smoothing 0.05 \
  --class_weight_mode balanced \
  --checkpoint_tie_break secondary \
  > logs/<name>.log 2>&1 &
```

Do not replace this baseline with HS4M-style recipe changes, PCA whitening, `token_num=8`, core-only tile loss, high-res shallow fusion, or multiple stacked tricks unless a full three-dataset 10-seed comparison shows a clear gain with acceptable runtime. Follow the rule from `经验教训.md`: keep `WEIGHTED_ACCUM_GROUP2` as the main line, change one primary variable at a time, and record OA, AA, Kappa, mIoU, and training time.

## Current Model Architecture
`train.py` currently builds `ImprovedMambaHSI` with PCA-preprocessed 30-band input, `hidden_dim=128`, `token_num=4`, `group_num=4`, dual spatial/spectral branches, and competitive channel fusion.

Input preprocessing in `train.py`: Gaussian smoothing (`sigma=1`), PCA to 30 bands, per-band 2%-98% percentile stretching, and `ToTensor()`. The whole HSI image is then passed as one dense tensor `[1, 30, H, W]`; training does not use patch dataloaders.

The network uses:

1. `patch_embedding`: `1x1 Conv(30 -> 128) + GroupNorm(4, 128) + SiLU`.
2. Dual branch encoder in `ImprovedBothMamba`:
   - Spatial branch: `LightSpatialPrior -> PyramidRefinedChannelAttention -> Mamba` over flattened spatial tokens `[B, H*W, C]`.
   - `LightSpatialPrior`: depthwise `3x3` local feature extraction, a single-channel spatial gate, `1x1 Conv + GroupNorm + SiLU`, then an internal residual.
   - `PyramidRefinedChannelAttention`: default `num_scales=3`, `num_layers=2`, `num_heads=4`; each scale applies `PyramidAttention`, lower scales use average pooling and are upsampled back before concatenation and `1x1` projection.
   - `PyramidAttention` computes QKV with `1x1 Conv` plus depthwise dilated `3x3` convolution. In training the CLI default is `--pyramid_dilation 3`, so the active default is a single dilation value `3`; the class default `(2, 3)` only applies when `ImprovedMambaHSI` is instantiated without the training script argument.
   - Spectral branch: first-order spectral-difference enhancement, optional zero padding to `token_num * ceil(C / token_num)`, reshape to `[B*H*W, token_num, ceil(C/token_num)]`, then Mamba over each pixel's short spectral token sequence.
3. `CompetitiveFusion`: per-channel global softmax competition between spatial and spectral branch logits from `AdaptiveAvgPool2d(1) + Linear(C, C)`. If `wo_competitive` is selected, fusion falls back to `0.5 * (spa + spe)`.
   - `spectral_fusion_scale=1.0` preserves the historical fusion exactly. Values below `1.0` keep both branches and the competitive gate, but interpolate the fused feature toward the spatial branch.
4. Residuals are currently stacked: `LightSpatialPrior` has an internal residual; spatial and spectral branches keep their inner residuals when `use_residual=True`; `ImprovedBothMamba` also applies the outer residual according to `outer_residual_mode` (`standard`, `no_outer`, or `scaled`).
5. `AvgPool2d(2)` after the dual-branch block, then the classification head: `1x1 Conv(hidden_dim -> 128) + GroupNorm + SiLU + 1x1 Conv(128 -> num_classes)`.

Supported ablations in `model/MambaHSI.py` are:

- `full`: spatial branch + spectral branch + competitive fusion.
- `wo_lpps`: disables the spatial branch.
- `wo_dgs`: disables the spectral branch.
- `wo_lsp`: disables `LightSpatialPrior`.
- `wo_prca`: disables `PyramidRefinedChannelAttention`.
- `wo_competitive`: keeps both branches but replaces competitive fusion with a simple average.

## Current Stability Rules
- `SpeMamba` treats each spatial location as one spectral token sequence and restores the tensor back to `[B, C, H, W]` after Mamba.
- `GroupNorm` group counts and `PyramidAttention` head counts are not auto-resolved in the current source. Keep `hidden_dim`, active `channels`, and padded spectral channel counts divisible by `group_num`, and keep PRCA channel width divisible by `num_heads=4`.
- The default training configuration (`hidden_dim=128`, `token_num=4`, `group_num=4`, `num_heads=4`) satisfies those divisibility assumptions. Non-divisible experimental settings can still fail at `GroupNorm` or `einops.rearrange`.
- `GradScaler` is initialized independently for every seed. Do not move it back to module scope, because scaler state must not leak across seed runs.

## Tile Mode Notes
- Per-epoch validation only propagates the tiles that contain validation pixels (`val_tile_slices`). Tiles without validation pixels cannot change any validation metric, so this is exact, not an approximation. Test evaluation still sweeps every tile. When `--save_vis true` triggers a periodic map, that epoch falls back to the full tile list.
- `--tile_batch_size` (default `1`) batches equally sized tiles into one forward pass. Because the model has no cross-sample operations and the per-tile loss is a mean over that tile's labeled pixels, weighting a batch by its labeled-pixel share reproduces the tile-by-tile gradient exactly; only AMP reduction order differs. Keep the default for main-line runs and raise it for small-tile experiments, where launch overhead dominates.
- Context-restricted protocol (reviewer comment on spatial context): `tools/run_context_protocol.sh` runs `--tile_size P --tile_overlap 0` for `P` in `16 32 64 128` on LongKou/Qingyun/Tangdaowan with seeds `0,1,2`, writing to `RUNS_CONTEXT_PROTOCOL_P<size>`. Training only propagates blocks containing training pixels, so no test-region data is seen at training time, making `P=16` a patch-equivalent inductive protocol. Everything except `tile_size`/`tile_overlap`/`tile_batch_size` matches the main configuration, so the number of optimizer steps per epoch is unchanged and spatial context is the only variable.

## Validation-Only Tuning
Hyperparameter search must not evaluate the test set. Use `--evaluate_test false`; this writes `validation_result_tr100_val30.txt` per seed and `mean_validation_result.txt` per dataset, and deliberately does not write `result_tr100_val30.txt` or `mean_result.txt`.

For equal primary validation scores, use `--checkpoint_tie_break secondary`:

- Qingyun: `checkpoint_metric=oa`, then validation mIoU, then the earlier epoch.
- Tangdaowan: `checkpoint_metric=miou`, then validation OA, then the earlier epoch.

The first evidence-backed tuning screen keeps the dual-branch architecture and competitive fusion, and tests only `spectral_fusion_scale=1.0,0.75,0.5,0.25`. Generate the detached validation-only workflow with:

```bash
python tools/generate_quh_fusion_screen_commands.py \
  --gpus 3,4,5 \
  --seeds 0,1,2,3,4 \
  --output tools/run_quh_fusion_screen.sh
```

After it finishes, summarize validation metrics only with:

```bash
python tools/summarize_quh_validation_screen.py \
  --output RUNS_QUH_VALSCREEN_fusion_summary.csv
```

The fixed QUH training split contains exactly 100 samples per class, so `class_weight_mode=balanced` produces all-one base weights and is mathematically equivalent to `none`; only explicit `class_weight_multipliers` change the loss. The validation split also contains exactly 30 samples per class, so validation OA and AA are equivalent and Kappa is monotonic with OA. Tune only OA versus mIoU checkpoint selection.

## Training, Testing, and Review
Default QUH training follows the `RUNS_QUH_100_30_WEIGHTED_ACCUM_GROUP2` configuration above. For older non-QUH datasets, check the CLI defaults and the intended protocol before launching; do not assume QUH `100/30` or legacy `30/10` applies universally. The loss is cross-entropy. `indian` can use balanced class weights when `--class_weight_mode auto` or `balanced` is selected. The best checkpoint is selected by validation OA by default, then tested and written to `result_*.txt` and `mean_result.txt`.

Logits are upsampled back to label resolution with `bilinear` interpolation. Training loss uses `align_corners=False` through `head_loss`; validation and test explicitly use `align_corners=True` in `train.py`.

Keep Python style consistent with the repo: 4-space indentation, snake_case for functions, CamelCase for `nn.Module` classes. Recent commits use short experiment-focused subjects such as `CCAF-V2` or `实验一：去掉DynamicConvBlock`.
