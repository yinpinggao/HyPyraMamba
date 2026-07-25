#!/usr/bin/env python3
"""Generate a validation-only QUH spectral-fusion screening script."""

import argparse
import os
import shlex


DATASETS = (
    {
        'key': 'qingyun',
        'index': 11,
        'checkpoint_metric': 'oa',
        'extra_args': ('--class_weight_multipliers', '6:1.10'),
    },
    {
        'key': 'tangdaowan',
        'index': 12,
        'checkpoint_metric': 'miou',
        'extra_args': (),
    },
)


def parse_csv(value):
    items = [item.strip() for item in value.split(',') if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError('expected a comma-separated list')
    return items


def format_float_for_name(value):
    return '{:g}'.format(value).replace('-', 'm').replace('.', 'p')


def shell_join(parts):
    return ' '.join(shlex.quote(str(part)) for part in parts)


def build_train_command(args, dataset, scale):
    tag = format_float_for_name(scale)
    exp_name = 'RUNS_QUH_VALSCREEN_{}_SPE_SCALE{}'.format(
        dataset['key'].upper(),
        tag.upper(),
    )
    command = [
        args.python,
        '-u',
        'train.py',
        '--dataset_index', str(dataset['index']),
        '--data_set_path', './data',
        '--split_dir', './splits/quh_100_30_seed0-9',
        '--exp_name', exp_name,
        '--train_samples', '100',
        '--val_samples', '30',
        '--seed_list', args.seeds,
        '--max_epoch', '200',
        '--tile_size', '512',
        '--tile_overlap', '32',
        '--tile_update_groups', '2',
        '--optimizer', 'adam',
        '--scheduler', 'none',
        '--lr', '0.0003',
        '--weight_decay', '1e-5',
        '--label_smoothing', '0.05',
        '--class_weight_mode', 'balanced',
        '--checkpoint_metric', dataset['checkpoint_metric'],
        '--checkpoint_tie_break', 'secondary',
        '--evaluate_test', 'false',
        '--gaussian_sigma', '1.0',
        '--gaussian_spectral_sigma', '1.0',
        '--spectral_diff_alpha', '0.5',
        '--spectral_fusion_scale', str(scale),
    ]
    command.extend(dataset['extra_args'])
    log_path = 'logs/quh_valscreen_{}_spe_scale{}.log'.format(dataset['key'], tag)
    return exp_name, log_path, command


def append_batch(lines, args, dataset, scales, gpu_batch):
    lines.append('pids=()')
    for gpu, scale in zip(gpu_batch, scales):
        exp_name, log_path, command = build_train_command(args, dataset, scale)
        lines.append(
            'echo "[launch] dataset={} scale={} gpu={} exp={}"'.format(
                dataset['key'], scale, gpu, exp_name
            )
        )
        lines.append(
            'CUDA_VISIBLE_DEVICES={} nohup {} > {} 2>&1 < /dev/null &'.format(
                shlex.quote(str(gpu)),
                shell_join(command),
                shlex.quote(log_path),
            )
        )
        lines.append('pids+=("$!")')
    lines.extend([
        'status=0',
        'for pid in "${pids[@]}"; do',
        '  if ! wait "$pid"; then status=1; fi',
        'done',
        'if [ "$status" -ne 0 ]; then',
        '  echo "[failed] at least one validation-screen job failed" >&2',
        '  exit "$status"',
        'fi',
    ])


def build_script(args):
    lines = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        '',
        'REPO_DIR=${REPO_DIR:-' + shlex.quote(args.repo_dir) + '}',
        'cd "$REPO_DIR"',
        'mkdir -p logs',
        '',
        'echo "[info] validation-only screen; test set evaluation is disabled"',
        'echo "[info] seeds={}"'.format(args.seeds),
        'echo "[info] gpus={}"'.format(','.join(args.gpus)),
        '',
    ]

    for dataset in DATASETS:
        lines.append('echo "[dataset] {}"'.format(dataset['key']))
        for start in range(0, len(args.scales), len(args.gpus)):
            scales = args.scales[start:start + len(args.gpus)]
            gpu_batch = args.gpus[:len(scales)]
            append_batch(lines, args, dataset, scales, gpu_batch)
        lines.append('')

    lines.extend([
        'echo "[done] validation-only fusion screen completed"',
        'echo "[next] compare mean_validation_result.txt files; do not inspect test metrics"',
        '',
    ])
    return '\n'.join(lines)


def get_parser():
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parser = argparse.ArgumentParser(
        description='Generate, but do not execute, a validation-only fusion-scale screen.'
    )
    parser.add_argument('--repo-dir', default=repo_dir)
    parser.add_argument(
        '--python',
        default='/home/guest/anaconda3/envs/gyp_hsi_env/bin/python',
    )
    parser.add_argument('--gpus', type=parse_csv, default=['3', '4', '5'])
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument(
        '--scales',
        type=lambda value: [float(item) for item in parse_csv(value)],
        default=[1.0, 0.75, 0.5, 0.25],
    )
    parser.add_argument('--output', default=None)
    return parser


def main():
    args = get_parser().parse_args()
    if any(not 0.0 < scale <= 1.0 for scale in args.scales):
        raise SystemExit('--scales values must be within (0, 1]')

    script = build_script(args)
    if args.output is None:
        print(script)
        return

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script)
    os.chmod(output_path, 0o755)
    print(output_path)


if __name__ == '__main__':
    main()
