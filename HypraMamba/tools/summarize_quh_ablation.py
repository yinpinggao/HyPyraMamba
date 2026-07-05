#!/usr/bin/env python3
"""Summarize QUH ablation results into a CSV table."""

import argparse
import csv
import glob
import os
import re

import numpy as np


DATASETS = [
    ("QUH-Pingan", 10),
    ("QUH-Qingyun", 11),
    ("QUH-Tangdaowan", 12),
]

ABLATIONS = [
    ("wo_lpps", "w/o LPPS-Mamba"),
    ("wo_lsp", "w/o LSP (in LPPS-Mamba)"),
    ("wo_prca", "w/o PRCA (in LPPS-Mamba)"),
    ("wo_dgs", "w/o DGS-Mamba"),
    ("wo_diff", "w/o Diff. (in DGS-Mamba)"),
    ("wo_competitive", "w/o channel-wise competitive fusion module"),
    ("full", "PyS2CF-Mamba"),
]

METRIC_PATTERNS = {
    "OA": re.compile(r"^OA=([0-9eE+\-.]+)", re.MULTILINE),
    "AA": re.compile(r"^AA=([0-9eE+\-.]+)", re.MULTILINE),
    "Kpp": re.compile(r"^kpp=([0-9eE+\-.]+)", re.MULTILINE),
    "mIOU": re.compile(r"^mIOU_test:?([0-9eE+\-.]+)", re.MULTILINE),
    "Train_Time": re.compile(r"^Train time\(s\)=([0-9eE+\-.]+)", re.MULTILINE),
    "Test_Time": re.compile(r"^Test time\(s\)=([0-9eE+\-.]+)", re.MULTILINE),
}

MEAN_PATTERNS = {
    "OA": re.compile(r"^OA=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)", re.MULTILINE),
    "AA": re.compile(r"^AA=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)", re.MULTILINE),
    "Kpp": re.compile(r"^Kpp=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)", re.MULTILINE),
    "mIOU": re.compile(r"^mIOU_test=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)", re.MULTILINE),
}


def model_name(base_model_name, ablation):
    if ablation == "full":
        return base_model_name
    return "{}_{}".format(base_model_name, ablation)


def percent(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    if np.nanmax(np.abs(values)) <= 1.5:
        return values * 100.0
    return values


def extract_last(pattern, text):
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_seed_results(result_dir, train_samples, val_samples):
    pattern = os.path.join(
        result_dir,
        "run*_seed*",
        "result_tr{}_val{}.txt".format(train_samples, val_samples),
    )
    files = sorted(glob.glob(pattern))
    values = {key: [] for key in METRIC_PATTERNS}

    for path in files:
        with open(path, "r", errors="ignore") as f:
            text = f.read()
        for key, metric_pattern in METRIC_PATTERNS.items():
            value = extract_last(metric_pattern, text)
            if value is not None:
                values[key].append(value)

    if not values["OA"]:
        return None

    summary = {"seeds": len(values["OA"])}
    for key, metric_values in values.items():
        metric_values = np.asarray(metric_values, dtype=np.float64)
        if key in ("OA", "AA", "Kpp", "mIOU"):
            metric_values = percent(metric_values)
        if metric_values.size == 0:
            summary["{}_mean".format(key)] = ""
            summary["{}_std".format(key)] = ""
        else:
            summary["{}_mean".format(key)] = float(np.mean(metric_values))
            summary["{}_std".format(key)] = float(np.std(metric_values))
    return summary


def parse_mean_result(mean_result_path):
    if not os.path.exists(mean_result_path):
        return None
    with open(mean_result_path, "r", errors="ignore") as f:
        text = f.read()

    summary = {"seeds": ""}
    found = False
    for key, pattern in MEAN_PATTERNS.items():
        match = pattern.search(text)
        if match:
            summary["{}_mean".format(key)] = float(match.group(1))
            summary["{}_std".format(key)] = float(match.group(2))
            found = True
        else:
            summary["{}_mean".format(key)] = ""
            summary["{}_std".format(key)] = ""
    return summary if found else None


def format_value(value):
    if value == "":
        return ""
    return "{:.4f}".format(float(value))


def build_rows(args):
    rows = []
    for dataset_name, dataset_index in DATASETS:
        for ablation, table_row in ABLATIONS:
            curr_model_name = model_name(args.base_model_name, ablation)
            result_dir = os.path.join(args.work_dir, args.exp_name, curr_model_name, dataset_name)
            summary = parse_seed_results(result_dir, args.train_samples, args.val_samples)
            status = "ok"
            if summary is None:
                summary = parse_mean_result(os.path.join(result_dir, "mean_result.txt"))
                status = "mean_result_only" if summary is not None else "missing"
            if summary is None:
                summary = {}

            row = {
                "dataset": dataset_name,
                "dataset_index": dataset_index,
                "ablation": ablation,
                "table_row": table_row,
                "model_dir": curr_model_name,
                "status": status,
                "seeds": summary.get("seeds", ""),
                "result_dir": result_dir,
            }
            for key in ["OA", "AA", "Kpp", "mIOU", "Train_Time", "Test_Time"]:
                row["{}_mean".format(key)] = format_value(summary.get("{}_mean".format(key), ""))
                row["{}_std".format(key)] = format_value(summary.get("{}_std".format(key), ""))
            rows.append(row)
    return rows


def get_parser():
    parser = argparse.ArgumentParser(description="Summarize QUH ablation outputs.")
    parser.add_argument("--work-dir", default=".")
    parser.add_argument("--exp_name", default="RUNS_QUH_ABLATION_100_30")
    parser.add_argument("--base-model-name", default="MambaHSI_competitive_diff_alpha0p5")
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--val_samples", type=int, default=30)
    parser.add_argument("--output", default="RUNS_QUH_ABLATION_100_30/quh_ablation_summary.csv")
    return parser


def main():
    args = get_parser().parse_args()
    rows = build_rows(args)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "dataset",
        "dataset_index",
        "ablation",
        "table_row",
        "model_dir",
        "status",
        "seeds",
        "OA_mean",
        "OA_std",
        "AA_mean",
        "AA_std",
        "Kpp_mean",
        "Kpp_std",
        "mIOU_mean",
        "mIOU_std",
        "Train_Time_mean",
        "Train_Time_std",
        "Test_Time_mean",
        "Test_Time_std",
        "result_dir",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
