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
`train.py` currently builds `ImprovedMambaHSI` with PCA-preprocessed 30-band input, `hidden_dim=128`, dual spatial/spectral branches, and competitive channel fusion.

Input preprocessing in `train.py`: Gaussian smoothing (`sigma=1`) and PCA to 30 bands. The network then uses:

1. `patch_embedding`: `1x1 Conv(30 -> 128) + GroupNorm + SiLU`.
2. Dual branch encoder in `ImprovedBothMamba`:
   - Spatial branch: `LightSpatialPrior -> PyramidRefinedChannelAttention -> Mamba` over flattened spatial tokens.
   - Spectral branch: grouped spectral tokens per pixel -> Mamba.
   - `token_num` is now a spectral-branch hyperparameter only. Spatial PRCA always works on the real `channels` width and no longer derives channels from `token_num`.
3. `CompetitiveFusion`: per-channel softmax competition between spatial and spectral branch logits.
4. `ImprovedBothMamba` owns the only outer residual inside the dual-branch block. Inner branch residuals are disabled when branches are instantiated inside this block, so single-branch ablations no longer apply a double residual.
5. `AvgPool2d(2)` after fusion, then the classification head: `1x1 Conv(hidden_dim -> 128 -> num_classes)`.

## Current Stability Rules
- `SpeMamba` treats each spatial location as one spectral token sequence and restores the tensor back to `[B, C, H, W]` after Mamba.
- `GroupNorm` group counts and `PyramidAttention` head counts are auto-resolved to safe divisors of the active channel width. Non-positive `channels`, `token_num`, `group_num`, or `num_heads` still raise `ValueError`.
- Non-divisible configs such as `hidden_dim=130`, `channels=130`, or `token_num=5` should now run instead of crashing on shape or normalization divisibility.

## Training, Testing, and Review
Default training here is `Adam(lr=3e-4, weight_decay=1e-5)`, 200 epochs, `train_samples=30`, `val_samples=10`, and 10 seeds. The loss is cross-entropy. `indian` can use balanced class weights when `--class_weight_mode auto` or `balanced` is selected. The best checkpoint is selected by validation OA, then tested and written to `result_*.txt` and `mean_result.txt`.

Logits are upsampled back to label resolution with `bilinear` interpolation and `align_corners=False` in training, validation, and test code paths.

Keep Python style consistent with the repo: 4-space indentation, snake_case for functions, CamelCase for `nn.Module` classes. Recent commits use short experiment-focused subjects such as `CCAF-V2` or `实验一：去掉DynamicConvBlock`.
