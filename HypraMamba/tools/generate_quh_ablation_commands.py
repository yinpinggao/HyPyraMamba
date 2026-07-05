#!/usr/bin/env python3
"""Generate QUH ablation launch scripts without starting training."""

import argparse
import os
import shlex


DATASETS = [
    ("pingan", "QUH-Pingan", 10),
    ("qingyun", "QUH-Qingyun", 11),
    ("tangdaowan", "QUH-Tangdaowan", 12),
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

COMMON_ARGS = [
    ("--data_set_path", "./data"),
    ("--split_dir", "./splits/quh_100_30_seed0-9"),
    ("--train_samples", "100"),
    ("--val_samples", "30"),
    ("--seed_list", "0,1,2,3,4,5,6,7,8,9"),
    ("--max_epoch", "200"),
    ("--tile_size", "512"),
    ("--tile_overlap", "32"),
    ("--tile_update_groups", "2"),
    ("--optimizer", "adam"),
    ("--scheduler", "none"),
    ("--lr", "0.0003"),
    ("--weight_decay", "1e-5"),
    ("--label_smoothing", "0.05"),
    ("--class_weight_mode", "balanced"),
    ("--checkpoint_metric", "oa"),
    ("--pca_components", "30"),
    ("--gaussian_sigma", "1.0"),
    ("--stretch_low", "2.0"),
    ("--stretch_high", "98.0"),
    ("--hidden_dim", "128"),
    ("--token_num", "4"),
    ("--group_num", "4"),
    ("--pool_size", "2"),
    ("--high_res_skip", "none"),
    ("--cls_head_dim", "128"),
    ("--prca_num_scales", "3"),
    ("--prca_num_layers", "2"),
    ("--prca_num_heads", "4"),
    ("--pyramid_dilation", "3"),
    ("--spectral_diff_alpha", "0.5"),
]


def parse_csv(value):
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def parse_ablations(value):
    requested = parse_csv(value)
    if requested == ["all"]:
        return ABLATIONS

    known = {name: label for name, label in ABLATIONS}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown ablation(s): {}; choices are {}".format(
                ",".join(unknown),
                ",".join(["all"] + list(known.keys())),
            )
        )
    return [(name, known[name]) for name in requested]


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def train_command(args, gpu, dataset_index, ablation):
    cmd = [
        args.python,
        "-u",
        "train.py",
        "--dataset_index",
        str(dataset_index),
        "--exp_name",
        args.exp_name,
        "--ablation",
        ablation,
    ]
    for key, value in COMMON_ARGS:
        cmd.extend([key, value])
    return "CUDA_VISIBLE_DEVICES={} {}".format(shlex.quote(str(gpu)), shell_join(cmd))


def append_launch_group(lines, args, ablation, dataset_chunk, gpus):
    lines.append("pids=()")
    for gpu, (dataset_key, dataset_name, dataset_index) in zip(gpus, dataset_chunk):
        log_path = "logs/quh_ablation_{}_{}.log".format(dataset_key, ablation)
        lines.append(
            "echo \"[launch] {} {} gpu={} log={}\"".format(
                ablation,
                dataset_name,
                gpu,
                log_path,
            )
        )
        lines.append("{} > {} 2>&1 &".format(
            train_command(args, gpu, dataset_index, ablation),
            shlex.quote(log_path),
        ))
        lines.append("pids+=(\"$!\")")

    lines.extend([
        "status=0",
        "for pid in \"${pids[@]}\"; do",
        "  if ! wait \"$pid\"; then",
        "    status=1",
        "  fi",
        "done",
        "if [ \"$status\" -ne 0 ]; then",
        "  echo \"[failed] at least one job in this group failed\" >&2",
        "  exit \"$status\"",
        "fi",
    ])


def build_script(args):
    gpus = args.gpus
    ablations = args.ablations

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "REPO_DIR=${REPO_DIR:-" + shlex.quote(args.repo_dir) + "}",
        "cd \"$REPO_DIR\"",
        "mkdir -p logs",
        "",
        "echo \"[info] repo=$REPO_DIR\"",
        "echo \"[info] exp_name={}\"".format(args.exp_name),
        "echo \"[info] gpus={}\"".format(",".join(gpus)),
        "",
    ]

    for ablation, label in ablations:
        lines.append("echo \"[start] {} ({})\"".format(label, ablation))
        for start in range(0, len(DATASETS), len(gpus)):
            dataset_chunk = DATASETS[start:start + len(gpus)]
            append_launch_group(lines, args, ablation, dataset_chunk, gpus)
        lines.append("echo \"[done] {} ({})\"".format(label, ablation))
        lines.append("")

    lines.extend([
        "echo \"[summary] writing {}\"".format(args.summary_output),
        shell_join([
            args.python,
            "tools/summarize_quh_ablation.py",
            "--exp_name",
            args.exp_name,
            "--output",
            args.summary_output,
        ]),
        "echo \"[done] all QUH ablations finished\"",
        "",
    ])
    return "\n".join(lines)


def get_parser():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(
        description="Generate a QUH ablation shell script. It does not start training."
    )
    parser.add_argument("--repo-dir", default=repo_dir)
    parser.add_argument("--python", default="python")
    parser.add_argument("--gpus", type=parse_csv, default=["0", "1", "2"])
    parser.add_argument("--exp_name", default="RUNS_QUH_ABLATION_100_30")
    parser.add_argument("--ablations", type=parse_ablations, default=ABLATIONS)
    parser.add_argument(
        "--summary-output",
        default=None,
    )
    parser.add_argument("--output", default=None, help="Optional path for the generated shell script.")
    return parser


def main():
    args = get_parser().parse_args()
    if len(args.gpus) == 0:
        raise SystemExit("--gpus must contain at least one GPU id")
    if args.summary_output is None:
        args.summary_output = os.path.join(args.exp_name, "quh_ablation_summary.csv")

    script_text = build_script(args)
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(script_text)
        os.chmod(args.output, 0o755)
        print(args.output)
    else:
        print(script_text)


if __name__ == "__main__":
    main()
