# Repository Guidelines

## Project Structure
`train.py` is the only training entry. The model lives in `model/MambaHSI.py`. Data loading, loss, evaluation, logging, and visualization are under `utils/`. Raw `.mat` datasets are read from `data/<DatasetName>/`. Training artifacts go to `RUNS/<model>/<dataset>/run*_seed*/`. Runtime logs go to `logs/`.

## Current Run Matrix
The active long-running jobs are:

| GPU | dataset_index | dataset | log |
| --- | --- | --- | --- |
| 0 | 4 | LongKou | `logs/longkou_competitive.log` |
| 0 | 5 | Salinas | `logs/salinas_competitive.log` |
| 1 | 6 | indian | `logs/indian_competitive.log` |
| 1 | 8 | XuZhou | `logs/xuzhou_competitive.log` |

The command shape is:

```bash
CUDA_VISIBLE_DEVICES=<gpu> nohup python train.py --dataset_index <id> > logs/<name>.log 2>&1 &
```

## Current Model Architecture
These commands build `ImprovedMambaHSI` with `hidden_dim=128`, dual spatial/spectral branches, and competitive channel fusion.

Input preprocessing in `train.py`: Gaussian smoothing (`sigma=1`) and PCA to 30 bands. The network then uses:

1. `patch_embedding`: `1x1 Conv(30 -> 128) + GroupNorm + SiLU`.
2. Dual branch encoder in `ImprovedBothMamba`:
   - Spatial branch: `LightSpatialPrior -> PyramidRefinedChannelAttention -> Mamba` over flattened spatial tokens.
   - Spectral branch: grouped spectral tokens per pixel -> Mamba, with no spectral PRCA.
3. `CompetitiveFusion`: per-channel softmax competition between spatial and spectral branch logits. The older gated additions are no longer present. The fused output is added back to the input residual.
4. `AvgPool2d(2)` after fusion, then the classification head: `1x1 Conv(128 -> 128 -> num_classes)`.

## Training, Testing, and Review
Default training here is `Adam(lr=3e-4, weight_decay=1e-5)`, 200 epochs, `train_samples=30`, `val_samples=10`, and 10 seeds. The loss is cross-entropy. `indian` can use balanced class weights when `--class_weight_mode auto` or `balanced` is selected. The best checkpoint is selected by validation OA, then tested and written to `result_*.txt` and `mean_result.txt`.

Keep Python style consistent with the repo: 4-space indentation, snake_case for functions, CamelCase for `nn.Module` classes. Recent commits use short experiment-focused subjects such as `CCAF-V2` or `实验一：去掉DynamicConvBlock`.
