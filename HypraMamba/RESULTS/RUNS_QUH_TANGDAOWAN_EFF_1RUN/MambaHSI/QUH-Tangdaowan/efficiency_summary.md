# QUH-Tangdaowan Efficiency Summary

Source run: `RUNS_QUH_TANGDAOWAN_EFF_1RUN/MambaHSI/QUH-Tangdaowan`

| Dataset | Run | Train time(s) | Test time(s) | Paras(M) | FLOPs(G/pixel) |
|---|---:|---:|---:|---:|---:|
| QUH-Tangdaowan | seed0 | 6804.66 | 7.7352 | 2.996295 | 0.001499 |

Notes:
- Raw calflops output: `393.05 GFLOPS` for input shape `(1, 30, 512, 512)`.
- Per-pixel FLOPs: `393.05 / (512 * 512) = 0.0014993668 G/pixel`.
- `mean_result.txt` records testing time in milliseconds in this checkout, so `7735.17 ms = 7.73517 s`.
