# Repository Guidelines

## Project Structure
`train.py` is the only training entry. The model lives in `model/MambaHSI.py`. Data loading, loss, evaluation, logging, and visualization are under `utils/`. Raw `.mat` datasets are read from `data/<DatasetName>/`. Training artifacts go to `RUNS/<model>/<dataset>/run*_seed*/`. Runtime logs go to `logs/`.

## Current Run Matrix
The active long-running jobs are:

| GPU | dataset_index | dataset | log |
| --- | --- | --- | --- |
| 0 | 4 | LongKou | `logs/longkou_ccaf_v2.log` |
| 0 | 5 | Salinas | `logs/salinas_ccaf_v2.log` |
| 1 | 6 | indian | `logs/indian_ccaf_v2.log` |
| 1 | 8 | XuZhou | `logs/xuzhou_ccaf_v2.log` |

The command shape is:

```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python train.py --dataset_index <id> --fusion_mode ccaf_v2 > logs/<name>.log 2>&1 &
```

## Current Model Architecture
These commands build `ImprovedMambaHSI` with `hidden_dim=128`, `mamba_type='both'`, and `fusion_mode='ccaf_v2'`.

Input preprocessing in `train.py`: Gaussian smoothing (`sigma=1`) and PCA to 30 bands. The network then uses:

1. `patch_embedding`: `1x1 Conv(30 -> 128) + GroupNorm + SiLU`.
2. Dual branch encoder in `ImprovedBothMamba`:
   - Spatial branch: `LightSpatialPrior -> PyramidRefinedChannelAttention -> Mamba` over flattened spatial tokens.
   - Spectral branch: spectral first-difference injection -> `PyramidRefinedChannelAttention -> Mamba` over 4 spectral tokens per pixel.
3. `CrossBranchBridge`: channel gates let the spectral branch reweight spatial channels, and vice versa.
4. `ConflictSuppressedCCAF` (`ccaf_v2`): first does per-channel competitive fusion, then adds a gated consensus term, and suppresses that consensus on high-conflict channels. The fused output is added back to the input residual.
5. `AvgPool2d(2)` after fusion, then:
   - classification head: `1x1 Conv(128 -> 128 -> num_classes)`
   - reconstruction head: `1x1 Conv(128 -> 64 -> 30)` for the auxiliary loss.

`spatial_mode` is left at `auto`, but the current `ImprovedSpaMamba` implementation stores this flag and does not branch on it. So these four runs share the same spatial block code path.

## Training, Testing, and Review
Default training here is `Adam(lr=3e-4, weight_decay=1e-5)`, 200 epochs, `train_samples=30`, `val_samples=10`, and 3 seeds. The loss is cross-entropy plus `0.05 * recon_loss`; `recon_loss` defaults to `SmoothL1`. `indian` also auto-enables balanced class weights and label smoothing. The best checkpoint is selected by validation OA, then tested and written to `result_*.txt` and `mean_result.txt`.

Keep Python style consistent with the repo: 4-space indentation, snake_case for functions, CamelCase for `nn.Module` classes. Recent commits use short experiment-focused subjects such as `CCAF-V2` or `实验一：去掉DynamicConvBlock`.
