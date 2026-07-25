#!/usr/bin/env python3
"""Export prediction maps from validation-selected QUH checkpoints."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from torch.cuda.amp import autocast
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.MambaHSI import ImprovedMambaHSI as MambaHSI
import utils.data_load_operate as data_load_operate
from utils.HSICommonUtils import ImageStretching
from utils.Loss import resize
from utils.visual_predict import visualize_predict
from utils.checkpoint_selection import checkpoint_rank


DATASETS = {
    "QUH-Pingan": 10,
    "QUH-Qingyun": 11,
    "QUH-Tangdaowan": 12,
}

MODEL_KWARGS = {
    "hidden_dim": 128,
    "token_num": 4,
    "group_num": 4,
    "use_residual": True,
    "pyramid_dilation": "3",
    "ablation": "full",
    "outer_residual_mode": "standard",
    "outer_residual_alpha": 1.0,
    "spectral_diff_alpha": 0.5,
    "pool_size": 2,
    "high_res_skip": "none",
    "cls_head_dim": 128,
    "prca_num_scales": 3,
    "prca_num_layers": 2,
    "prca_num_heads": 4,
    "lsp_reduction": 4,
    "spa_mamba_d_state": 16,
    "spa_mamba_d_conv": 4,
    "spa_mamba_expand": 2,
    "spe_mamba_d_state": 16,
    "spe_mamba_d_conv": 4,
    "spe_mamba_expand": 2,
}

TEST_OA_RE = re.compile(r"^OA=([0-9eE+\-.]+)", re.MULTILINE)
RUN_RESULT_RE = re.compile(r"exp_idx=(\d+)\s+seed=(\d+)")
VALIDATION_METRICS = {"oa", "aa", "miou", "kappa"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_root",
        default="RUNS_QUH_100_30_WEIGHTED_ACCUM_GROUP2/MambaHSI_competitive_diff_alpha0p5",
    )
    parser.add_argument("--data_set_path", default="./data")
    parser.add_argument("--output_dir", default="PRED_VIS_BEST_QUH_VAL_SELECTED")
    parser.add_argument(
        "--datasets",
        default="QUH-Pingan,QUH-Qingyun,QUH-Tangdaowan",
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--val_samples", type=int, default=30)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument("--pca_components", type=int, default=30)
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument(
        "--gaussian_spectral_sigma",
        type=float,
        default=None,
        help=(
            "Gaussian sigma for the spectral axis. The default reuses "
            "--gaussian_sigma for legacy checkpoints."
        ),
    )
    parser.add_argument("--spectral_fusion_scale", type=float, default=1.0)
    parser.add_argument("--stretch_low", type=float, default=2.0)
    parser.add_argument("--stretch_high", type=float, default=98.0)
    parser.add_argument(
        "--selection_metric",
        default="auto",
        choices=["auto", "oa", "aa", "miou", "kappa"],
        help=(
            "Validation metric used to compare saved seed checkpoints. "
            "'auto' uses the checkpoint metric recorded by train.py."
        ),
    )
    args = parser.parse_args()
    if args.gaussian_sigma < 0:
        parser.error("--gaussian_sigma must be non-negative")
    if args.gaussian_spectral_sigma is not None and args.gaussian_spectral_sigma < 0:
        parser.error("--gaussian_spectral_sigma must be non-negative")
    if not 0.0 < args.spectral_fusion_scale <= 1.0:
        parser.error("--spectral_fusion_scale must be within (0, 1]")
    return args


def read_test_oa(result_path, expected_exp_idx, expected_seed):
    matches = []
    current_run = None
    for line in result_path.read_text(errors="replace").splitlines():
        run_match = RUN_RESULT_RE.search(line)
        if run_match is not None:
            current_run = (int(run_match.group(1)), int(run_match.group(2)))
            continue
        oa_match = TEST_OA_RE.match(line)
        if oa_match is not None and current_run == (expected_exp_idx, expected_seed):
            matches.append(oa_match.group(1))
    if not matches:
        raise ValueError(
            "Could not parse test OA for exp_idx={} seed={} from {}".format(
                expected_exp_idx,
                expected_seed,
                result_path,
            )
        )
    # train.py appends single-run results. The last block matches the checkpoint
    # left on disk if an experiment directory was reused.
    return float(matches[-1]), len(matches)


def parse_evaluation_line(line):
    if "Evaluate " not in line:
        return None

    payload = line.split("Evaluate ", 1)[1]
    parts = payload.split("|")
    try:
        row = {"epoch": int(parts[0])}
    except ValueError:
        return None

    key_map = {
        "oa": "oa",
        "aa": "aa",
        "miou": "miou",
        "kappa": "kappa",
        "score": "checkpoint_score",
    }
    for item in parts[1:]:
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "checkpoint_metric":
            row["checkpoint_metric"] = value.lower()
        elif key == "checkpoint_tie_break":
            row["checkpoint_tie_break"] = value.lower()
        elif key in key_map:
            row[key_map[key]] = float(value)
    return row


def parse_validation_checkpoints(log_path):
    if not log_path.exists():
        raise FileNotFoundError(
            "Training log required for validation-based selection does not exist: {}".format(log_path)
        )

    completed_runs = {}
    evaluation_rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        row = parse_evaluation_line(line)
        if row is not None:
            if evaluation_rows and row["epoch"] <= evaluation_rows[-1]["epoch"]:
                # Drop an unfinished attempt if the same log later restarts
                # from epoch 0 (or otherwise moves backwards).
                evaluation_rows = []
            evaluation_rows.append(row)

        run_match = RUN_RESULT_RE.search(line)
        if run_match is None:
            continue
        if not evaluation_rows:
            raise ValueError(
                "Found a completed run marker without validation rows in {}".format(log_path)
            )

        exp_idx = int(run_match.group(1))
        seed = int(run_match.group(2))
        run_name = "run{}_seed{}".format(exp_idx, seed)
        explicit_metrics = {
            item["checkpoint_metric"]
            for item in evaluation_rows
            if "checkpoint_metric" in item
        }
        if len(explicit_metrics) > 1:
            raise ValueError(
                "Run {} contains multiple checkpoint metrics: {}".format(
                    run_name,
                    sorted(explicit_metrics),
                )
            )
        checkpoint_metric = next(iter(explicit_metrics), "oa")
        checkpoint_metric_source = "logged" if explicit_metrics else "legacy_assumed_oa"
        if checkpoint_metric not in VALIDATION_METRICS:
            raise ValueError(
                "Unsupported checkpoint metric '{}' in {}".format(checkpoint_metric, log_path)
            )

        explicit_tie_breaks = {
            item["checkpoint_tie_break"]
            for item in evaluation_rows
            if "checkpoint_tie_break" in item
        }
        if len(explicit_tie_breaks) > 1:
            raise ValueError(
                "Run {} contains multiple checkpoint tie-break rules: {}".format(
                    run_name,
                    sorted(explicit_tie_breaks),
                )
            )
        checkpoint_tie_break = next(iter(explicit_tie_breaks), "latest")
        checkpoint_tie_break_source = (
            "logged" if explicit_tie_breaks else "legacy_assumed_latest"
        )

        scored_rows = []
        for item in evaluation_rows:
            score = item.get("checkpoint_score", item.get(checkpoint_metric))
            if score is None:
                raise ValueError(
                    "Validation metric '{}' is missing for {} in {}".format(
                        checkpoint_metric,
                        run_name,
                        log_path,
                    )
                )
            if checkpoint_tie_break == "latest":
                rank = (score, item["epoch"])
            elif checkpoint_tie_break == "earliest":
                rank = (score, -item["epoch"])
            else:
                metrics = {
                    metric: item[metric]
                    for metric in VALIDATION_METRICS
                    if metric in item
                }
                if len(metrics) != len(VALIDATION_METRICS):
                    raise ValueError(
                        "Incomplete validation metrics for {} epoch {} in {}".format(
                            run_name,
                            item["epoch"],
                            log_path,
                        )
                    )
                rank = checkpoint_rank(
                    metrics,
                    checkpoint_metric,
                    checkpoint_tie_break,
                    item["epoch"],
                )
            scored_rows.append((
                rank,
                score,
                item["epoch"],
                item,
            ))

        _, checkpoint_score, checkpoint_epoch, checkpoint_row = max(
            scored_rows,
            key=lambda item: item[0],
        )
        completed_runs[run_name] = {
            "run_name": run_name,
            "exp_idx": exp_idx,
            "seed": seed,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metric_source": checkpoint_metric_source,
            "checkpoint_tie_break": checkpoint_tie_break,
            "checkpoint_tie_break_source": checkpoint_tie_break_source,
            "checkpoint_score": checkpoint_score,
            "checkpoint_epoch": checkpoint_epoch,
            "validation_metrics": {
                metric: checkpoint_row[metric]
                for metric in VALIDATION_METRICS
                if metric in checkpoint_row
            },
        }
        evaluation_rows = []

    if not completed_runs:
        raise ValueError("No completed validation runs found in {}".format(log_path))
    return completed_runs


def select_best_run(dataset_dir, train_samples, val_samples, selection_metric):
    log_path = dataset_dir / "train_tr{}_val{}.log".format(train_samples, val_samples)
    run_records = parse_validation_checkpoints(log_path)
    candidates = []
    for run_name, record in sorted(run_records.items()):
        run_dir = dataset_dir / run_name
        result_path = run_dir / "result_tr{}_val{}.txt".format(train_samples, val_samples)
        checkpoint_path = run_dir / "best_tr{}_val{}.pth".format(train_samples, val_samples)
        if not result_path.exists() or not checkpoint_path.exists():
            continue
        candidates.append({
            **record,
            "result_path": result_path,
            "checkpoint_path": checkpoint_path,
        })

    if not candidates:
        raise FileNotFoundError(
            "No completed runs with validation logs, results, and checkpoints under {}".format(dataset_dir)
        )

    if selection_metric == "auto":
        checkpoint_metrics = {item["checkpoint_metric"] for item in candidates}
        if len(checkpoint_metrics) != 1:
            raise ValueError(
                "Cannot use --selection_metric auto because runs use different checkpoint metrics: {}".format(
                    sorted(checkpoint_metrics)
                )
            )
        resolved_metric = next(iter(checkpoint_metrics))
    else:
        resolved_metric = selection_metric

    for item in candidates:
        if resolved_metric not in item["validation_metrics"]:
            raise ValueError(
                "Validation metric '{}' is unavailable for {} in {}".format(
                    resolved_metric,
                    item["run_name"],
                    log_path,
                )
            )
        item["selection_metric"] = resolved_metric
        item["validation_score"] = item["validation_metrics"][resolved_metric]

    candidates.sort(
        key=lambda item: (-item["validation_score"], item["seed"], item["run_name"])
    )
    selected = candidates[0]
    test_oa, result_blocks = read_test_oa(
        selected["result_path"],
        selected["exp_idx"],
        selected["seed"],
    )
    selected["test_oa"] = test_oa
    selected["result_blocks"] = result_blocks
    selected["log_path"] = log_path
    return selected


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_tile_slices(height, width, tile_size, overlap):
    return [
        (y0, min(y0 + tile_size, height), x0, min(x0 + tile_size, width))
        for y0 in tile_starts(height, tile_size, overlap)
        for x0 in tile_starts(width, tile_size, overlap)
    ]


def predict_tiled(model, x_cpu, tile_slices, class_count, output_size, device):
    height, width = output_size
    logit_sum = np.zeros((class_count, height, width), dtype=np.float32)
    logit_count = np.zeros((height, width), dtype=np.float32)

    for y0, y1, x0, x1 in tile_slices:
        input_tile = x_cpu[:, :, y0:y1, x0:x1].to(device)
        with autocast(enabled=device.type == "cuda"):
            output_tile = model(input_tile)
            logits_tile = resize(
                input=output_tile,
                size=(y1 - y0, x1 - x0),
                mode="bilinear",
                align_corners=True,
            )

        logit_sum[:, y0:y1, x0:x1] += logits_tile.squeeze(0).float().cpu().numpy()
        logit_count[y0:y1, x0:x1] += 1.0
        del input_tile, output_tile, logits_tile

    logit_sum /= np.maximum(logit_count[None, :, :], 1.0)
    return np.expand_dims(np.argmax(logit_sum, axis=0), axis=0)


def export_dataset(args, dataset_name, device):
    baseline_root = Path(args.baseline_root)
    dataset_dir = baseline_root / dataset_name
    selected = select_best_run(
        dataset_dir,
        args.train_samples,
        args.val_samples,
        args.selection_metric,
    )
    run_name = selected["run_name"]
    result_path = selected["result_path"]
    checkpoint_path = selected["checkpoint_path"]

    data, gt = data_load_operate.load_data(dataset_name, args.data_set_path)
    gaussian_spectral_sigma = (
        args.gaussian_sigma
        if args.gaussian_spectral_sigma is None
        else args.gaussian_spectral_sigma
    )
    data_filtered = gaussian_filter(
        data,
        sigma=(args.gaussian_sigma, args.gaussian_sigma, gaussian_spectral_sigma),
    )
    data_reshaped = data_filtered.reshape(-1, data_filtered.shape[2])
    data_pca = PCA(n_components=args.pca_components).fit_transform(data_reshaped)
    data_pca = data_pca.reshape(data_filtered.shape[0], data_filtered.shape[1], -1)
    img = ImageStretching(data_pca, low=args.stretch_low, high=args.stretch_high)

    height, width, channels = data_pca.shape
    class_count = int(max(np.unique(gt)))
    x = transforms.ToTensor()(np.asarray(img, dtype=np.float32)).unsqueeze(0).float()
    tile_slices = generate_tile_slices(height, width, args.tile_size, args.tile_overlap)

    model_kwargs = dict(MODEL_KWARGS)
    model_kwargs["spectral_fusion_scale"] = args.spectral_fusion_scale
    model = MambaHSI(
        in_channels=channels,
        num_classes=class_count,
        **model_kwargs,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        predict = predict_tiled(model, x, tile_slices, class_count, (height, width), device)

    output_dir = Path(args.output_dir) / dataset_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "pred_full.png"
    mask_path = output_dir / "pred_mask.png"
    gt_path = output_dir / "gt.png"
    npy_path = output_dir / "pred_label_zero_based.npy"

    visualize_predict(gt, predict, str(full_path), str(gt_path), only_vis_label=False)
    visualize_predict(gt, predict, str(mask_path), str(gt_path), only_vis_label=True)
    np.save(npy_path, np.reshape(predict, (height, width)).astype(np.int16))

    return {
        "dataset": dataset_name,
        "selected_run": run_name,
        "selection_metric": selected["selection_metric"],
        "validation_score": selected["validation_score"],
        "checkpoint_metric": selected["checkpoint_metric"],
        "checkpoint_metric_source": selected["checkpoint_metric_source"],
        "checkpoint_tie_break": selected["checkpoint_tie_break"],
        "checkpoint_tie_break_source": selected["checkpoint_tie_break_source"],
        "checkpoint_score": selected["checkpoint_score"],
        "checkpoint_epoch": selected["checkpoint_epoch"],
        "validation_metrics_at_checkpoint": selected["validation_metrics"],
        "test_oa": selected["test_oa"],
        "test_oa_used_for_selection": False,
        "result_blocks": selected["result_blocks"],
        "training_log": str(selected["log_path"]),
        "checkpoint": str(checkpoint_path),
        "result": str(result_path),
        "pred_full": str(full_path),
        "pred_mask": str(mask_path),
        "gt": str(gt_path),
        "pred_label_zero_based": str(npy_path),
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "gaussian_spatial_sigma": args.gaussian_sigma,
        "gaussian_spectral_sigma": gaussian_spectral_sigma,
        "spectral_fusion_scale": args.spectral_fusion_scale,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(selected_datasets) - set(DATASETS))
    if unknown:
        raise ValueError("Unknown datasets: {}".format(unknown))

    manifest = {
        "baseline_root": args.baseline_root,
        "device": str(device),
        "selection_rule": (
            "highest requested validation metric among completed saved checkpoints; "
            "test OA is reported only after selection"
        ),
        "selection_metric_requested": args.selection_metric,
        "exports": [],
    }
    for dataset_name in selected_datasets:
        item = export_dataset(args, dataset_name, device)
        manifest["exports"].append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Wrote {}".format(manifest_path))


if __name__ == "__main__":
    main()
