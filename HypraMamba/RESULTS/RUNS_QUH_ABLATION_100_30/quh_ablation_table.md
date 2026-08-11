# QUH Ablation Table

Source: `HypraMamba/RUNS_QUH_ABLATION_100_30/quh_ablation_summary.csv`

Protocol: QUH fixed split, train/val/test = 100/30/rest, 10 seeds per configuration and dataset. Values are mean percentages.

| Configuration | Pingan OA | Pingan AA | Pingan Kappa | Qingyun OA | Qingyun AA | Qingyun Kappa | Tangdaowan OA | Tangdaowan AA | Tangdaowan Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o LPPS-Mamba | 89.85 | 87.69 | 85.19 | 85.64 | 86.37 | 81.22 | 86.38 | 92.47 | 84.81 |
| w/o LSP (in LPPS-Mamba) | 94.34 | 94.83 | 91.70 | 90.18 | 91.30 | 87.10 | 95.40 | 97.53 | 94.79 |
| w/o PRCA (in LPPS-Mamba) | 93.04 | 92.06 | 89.79 | 88.78 | 89.33 | 85.27 | 94.94 | 97.06 | 94.28 |
| w/o DGS-Mamba | 94.83 | 95.57 | 92.42 | 91.41 | 92.82 | 88.70 | 96.11 | 97.95 | 95.60 |
| w/o Diff. (in DGS-Mamba) | 94.98 | 95.54 | 92.63 | 90.73 | 92.15 | 87.83 | 95.79 | 97.67 | 95.23 |
| w/o channel-wise competitive fusion module | 94.84 | 95.58 | 92.42 | 91.21 | 92.80 | 88.45 | 95.42 | 97.64 | 94.82 |
| PyS2CF-Mamba | 94.74 | 95.32 | 92.27 | 90.93 | 92.34 | 88.09 | 95.79 | 97.80 | 95.24 |

