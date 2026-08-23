#!/usr/bin/env python3
"""Collect and validate the 10-run DGS-Mamba sensitivity sweep."""

import argparse
import csv
import glob
import os
import re

MEAN_PATTERNS = {
    'OA': re.compile(r'^OA=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'AA': re.compile(r'^AA=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'Kpp': re.compile(r'^Kpp=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'mIOU': re.compile(r'^mIOU_test=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'Train_time_s': re.compile(r'^Average training time\(s\)=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'Test_time_s': re.compile(r'^Average testing time\(s\)=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
}
VALIDATION_PATTERNS = {
    'OA': re.compile(r'^Validation OA=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'AA': re.compile(r'^Validation AA=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'Kpp': re.compile(r'^Validation Kpp=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'mIOU': re.compile(r'^Validation mIOU=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
    'Train_time_s': re.compile(r'^Average training time\(s\)=([0-9eE+\-.]+)\+-([0-9eE+\-.]+)', re.MULTILINE),
}
VARIANTS = [
    ('a0p00', 'alpha=0.00 (G=4)'),
    ('a0p25', 'alpha=0.25 (G=4)'),
    ('a0p50', 'alpha=0.50 (G=4) [paper control]'),
    ('a0p75', 'alpha=0.75 (G=4)'),
    ('a1p00', 'alpha=1.00 (G=4)'),
    ('g2', 'G=2 (alpha=0.5)'),
    ('g8', 'G=8 (alpha=0.5)'),
    ('g16', 'G=16 (alpha=0.5)'),
]
DATASETS = ['LongKou', 'QUH-Qingyun', 'QUH-Tangdaowan']


def read_mean(path, evaluate_test):
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    patterns = MEAN_PATTERNS if evaluate_test else VALIDATION_PATTERNS
    row = {}
    for name, pattern in patterns.items():
        match = pattern.search(text)
        row[name] = '{}+-{}'.format(match.group(1), match.group(2)) if match else ''
    return row


def count_seed_outputs(dataset_dir, seeds, evaluate_test):
    prefix = 'result_' if evaluate_test else 'validation_result_'
    completed = 0
    missing = []
    for seed in seeds:
        pattern = os.path.join(dataset_dir, 'run*_seed{}'.format(seed), '{}*.txt'.format(prefix))
        if glob.glob(pattern):
            completed += 1
        else:
            missing.append(seed)
    return completed, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='RUNS_DGS_SENS_20260823')
    parser.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--evaluate-test', default='true', choices=['true', 'false'])
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    seeds = [item.strip() for item in args.seeds.split(',') if item.strip()]
    evaluate_test = args.evaluate_test == 'true'
    if args.output is None:
        args.output = '{}_summary.csv'.format(args.prefix)

    rows = []
    incomplete = False
    summary_name = 'mean_result.txt' if evaluate_test else 'mean_validation_result.txt'
    for tag, label in VARIANTS:
        for dataset in DATASETS:
            pattern = os.path.join('{}_{}'.format(args.prefix, tag), '*', dataset, summary_name)
            matches = sorted(glob.glob(pattern))
            if not matches:
                incomplete = True
                rows.append({'variant': label, 'dataset': dataset, 'seed_files': '0/{}'.format(len(seeds)), 'missing_seeds': ','.join(seeds), 'OA': 'MISSING'})
                continue
            if len(matches) > 1:
                print('warning: {} matched {} folders, using {}'.format(pattern, len(matches), matches[0]))
            summary_path = matches[0]
            dataset_dir = os.path.dirname(summary_path)
            completed, missing = count_seed_outputs(dataset_dir, seeds, evaluate_test)
            incomplete |= bool(missing)
            row = read_mean(summary_path, evaluate_test)
            row.update({'variant': label, 'dataset': dataset, 'seed_files': '{}/{}'.format(completed, len(seeds)), 'missing_seeds': ','.join(missing)})
            rows.append(row)

    fieldnames = ['variant', 'dataset', 'seed_files', 'missing_seeds', 'OA', 'AA', 'Kpp', 'mIOU', 'Train_time_s', 'Test_time_s']
    with open(args.output, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print('{:36s} {:16s} seeds={:7s} OA={:16s} AA={:16s} Kpp={}'.format(row['variant'], row['dataset'], row.get('seed_files', ''), row.get('OA', ''), row.get('AA', ''), row.get('Kpp', '')))
    print('\nwrote {}'.format(args.output))
    if incomplete:
        raise SystemExit('DGS sensitivity sweep is incomplete; inspect MISSING rows and queue logs.')


if __name__ == '__main__':
    main()
