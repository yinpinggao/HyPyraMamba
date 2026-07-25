#!/usr/bin/env python3
"""Summarize validation-only QUH tuning runs without reading test results."""

import argparse
import ast
import csv
import re
import sys
from pathlib import Path


METRIC_RE = re.compile(
    r'^Validation (OA|AA|Kpp|mIOU)=([0-9.eE+-]+)\+-([0-9.eE+-]+)$',
    re.MULTILINE,
)


def read_logged_config(log_path):
    for line in log_path.read_text(errors='replace').splitlines():
        start = line.find("{'net_name':")
        if start < 0:
            continue
        return ast.literal_eval(line[start:])
    raise ValueError('Could not find logged training config in {}'.format(log_path))


def parse_summary(path):
    text = path.read_text(errors='replace')
    if 'test_evaluated=false' not in text:
        raise ValueError('{} is not a validation-only summary'.format(path))
    metrics = {
        name.lower(): (float(mean), float(std))
        for name, mean, std in METRIC_RE.findall(text)
    }
    required = {'oa', 'aa', 'kpp', 'miou'}
    if required.difference(metrics):
        raise ValueError('Missing validation metrics in {}'.format(path))

    dataset_dir = path.parent
    log_candidates = sorted(dataset_dir.glob('train_tr*_val*.log'))
    if len(log_candidates) != 1:
        raise ValueError('Expected one training log under {}'.format(dataset_dir))
    config = read_logged_config(log_candidates[0])
    metric_name = config['checkpoint_metric']
    ranking_metric = 'miou' if metric_name == 'miou' else 'oa'
    return {
        'dataset': dataset_dir.name,
        'experiment': path.parents[2].name,
        'model_dir': dataset_dir.parent.name,
        'seeds': ','.join(str(seed) for seed in config['seed_list']),
        'checkpoint_metric': metric_name,
        'checkpoint_tie_break': config['checkpoint_tie_break'],
        'spectral_fusion_scale': config.get('spectral_fusion_scale', 1.0),
        'gaussian_spatial_sigma': config['gaussian_sigma'],
        'gaussian_spectral_sigma': config.get(
            'gaussian_spectral_sigma',
            config['gaussian_sigma'],
        ),
        'spectral_diff_alpha': config['spectral_diff_alpha'],
        'val_oa_mean': metrics['oa'][0],
        'val_oa_std': metrics['oa'][1],
        'val_miou_mean': metrics['miou'][0],
        'val_miou_std': metrics['miou'][1],
        'ranking_score': metrics[ranking_metric][0],
        'test_evaluated': False,
        'summary_path': str(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--pattern', default='RUNS_QUH_VALSCREEN_*/**/mean_validation_result.txt')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.root.glob(args.pattern))
    if not paths:
        raise SystemExit('No validation summaries matched {}'.format(args.pattern))
    rows = [parse_summary(path) for path in paths]
    rows.sort(key=lambda row: (row['dataset'], -row['ranking_score'], row['experiment']))

    fieldnames = list(rows[0])
    output = args.output.open('w', newline='') if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            output.close()


if __name__ == '__main__':
    main()
