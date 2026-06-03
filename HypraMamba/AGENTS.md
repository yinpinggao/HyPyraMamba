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

## Training, Testing, and Review
Default training here is `Adam(lr=3e-4, weight_decay=1e-5)`, 200 epochs, `train_samples=30`, `val_samples=10`, and 10 seeds. The loss is cross-entropy. `indian` can use balanced class weights when `--class_weight_mode auto` or `balanced` is selected. The best checkpoint is selected by validation OA, then tested and written to `result_*.txt` and `mean_result.txt`.

Logits are upsampled back to label resolution with `bilinear` interpolation. Training loss uses `align_corners=False` through `head_loss`; validation and test explicitly use `align_corners=True` in `train.py`.

Keep Python style consistent with the repo: 4-space indentation, snake_case for functions, CamelCase for `nn.Module` classes. Recent commits use short experiment-focused subjects such as `CCAF-V2` or `实验一：去掉DynamicConvBlock`.
