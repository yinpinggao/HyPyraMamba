# Tree Species HSI 2026: Data Alignment Report

## P0 result

- Validation row-major truth-loop: **690829/690829 = 100.000000%**.
- Status: **PASS**. No training is started by this audit.
- Official train/val labeled-pixel overlap: **0**.
- Actual sample columns: `id`, `label`. The pipeline reads these names from the sample instead of hard-coding a conflicting schema.
- The former writer/inverse-writer round trip was not independent: the same wrong transpose could cancel itself and report 100%. The current audit uses fixed sentinel IDs and a directly defined row-major validation vector.
- Controlled submissions made from identical cached predictions also support this interpretation: canonical HxW + C-order scored `0.11278`, while applying the metadata transpose again scored `0.04828`. These scores are diagnostic evidence only, not a hyperparameter-tuning signal.

## Real file layouts

| file | variable | observed Python shape | dtype | loader |
|---|---|---:|---|---|
| `data_hsi.mat` | `data` | `(98, 4040, 2444)` | `uint16` | v7.3 / HDF5 |
| `train_label.mat` | `train_label` | `(4040, 2444)` | `uint8` | classic MAT |
| `val_label.mat` | `val_label` | `(4040, 2444)` | `uint8` | classic MAT |
| `test_scene1.mat` | `image` | `(98, 4507, 3104)` | `uint16` | v7.3 / HDF5 |
| `test_scene2.mat` | `image` | `(98, 2181, 3409)` | `uint16` | v7.3 / HDF5 |

`h5py.is_hdf5(path)` is checked before loading. The labeled cube is stored as `(98, 4040, 2444)` and maps to label `(H, W)=(4040, 2444)` via `transpose(1, 2, 0)`. Test cubes are stored as `(98, W, H)` and map to HWC via `transpose(2, 1, 0)`.

## Submission scene segmentation

| scene | H | W | pixels | raw MAT axis correction | CSV order |
|---|---:|---:|---:|---|---|
| scene1 | 3104 | 4507 | 13989728 | True | `C` |
| scene2 | 3409 | 2181 | 7435029 | True | `C` |

The sample was streamed end-to-end. Its only scene transitions are:

- row 0: scene1_00000000
- row 13989728: scene2_00000000

## ID to `(scene, row, col)` mapping

`scene_info.transpose=True` describes the raw HDF5 spatial layout: the observed test cube `(98, W, H)` is transposed once to canonical `(H, W, 98)`. It is **not** a second transpose instruction for CSV generation. Once predictions have shape `(H, W)`, sample IDs enumerate them directly in row-major C order:

```text
local_index = integer suffix of sample ID
row = local_index // W
col = local_index % W
ID = f"{scene}_{local_index:08d}"
```

- `scene1` probes: 0->(0, 0), 1->(0, 1), 4506->(0, 4506), 4507->(1, 0), 13989727->(3103, 4506)
- `scene2` probes: 0->(0, 0), 1->(0, 1), 2180->(0, 2180), 2181->(1, 0), 7435028->(3408, 2180)

Equivalently, `flat = prediction_hw.reshape(-1, order="C")`. All reshape/flatten calls in the implementation pass an explicit `order`.

## Label offset

Network predictions are treated as `0..16`; `model_output_to_labels(..., zero_based=True)` adds exactly one and rejects anything outside `1..17`. The truth-loop exercised this conversion for every labeled validation pixel.

## Class distributions

Background (`0`) is excluded below. Official train contains exactly 10 pixels per class.

| class | train pixels | val pixels |
|---:|---:|---:|
| 1 | 10 | 68425 |
| 2 | 10 | 38586 |
| 3 | 10 | 60956 |
| 4 | 10 | 10066 |
| 5 | 10 | 12191 |
| 6 | 10 | 18013 |
| 7 | 10 | 3470 |
| 8 | 10 | 4418 |
| 9 | 10 | 53578 |
| 10 | 10 | 86429 |
| 11 | 10 | 110949 |
| 12 | 10 | 125414 |
| 13 | 10 | 6230 |
| 14 | 10 | 12279 |
| 15 | 10 | 26427 |
| 16 | 10 | 46368 |
| 17 | 10 | 7030 |

- Train labeled total: 170
- Validation labeled total: 690829
- Train background: 9873590
- Validation background: 9182931

## Submission contract

`src/submit.py` is the sole CSV writer. After writing it reads the CSV back and asserts:

- output IDs exactly equal sample IDs and preserve their order;
- every prediction is between 1 and 17;
- output row count exactly equals sample row count.
