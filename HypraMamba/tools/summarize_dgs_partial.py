#!/usr/bin/env python3
"""Partial summary for the stopped DGS sensitivity sweep.

Unlike tools/summarize_dgs_sensitivity.py (which validates 10/10 seeds via
mean_result.txt), this script parses every per-seed result_*.txt that
actually exists and averages over however many seeds completed (n>=3).
Rows with fewer than 3 seeds are reported with n but flagged; rows with
0 seeds are reported as n/a.
"""

import glob
import os
import re
import statistics

PREFIX = 'RUNS_DGS_SENS_10RUN_20260823'
VARIANTS = ['a0p00', 'a0p25', 'a0p50', 'a0p75', 'a1p00', 'g2', 'g8', 'g16']
DATASETS = ['LongKou', 'QUH-Qingyun', 'QUH-Tangdaowan']

PATTERNS = {
    'OA': re.compile(r'^OA=([0-9eE+\-.]+)$', re.MULTILINE),
    'AA': re.compile(r'^AA=([0-9eE+\-.]+)$', re.MULTILINE),
    'Kpp': re.compile(r'^kpp=([0-9eE+\-.]+)$', re.MULTILINE),
    'mIOU': re.compile(r'^mIOU_test:([0-9eE+\-.]+)$', re.MULTILINE),
    'Train_s': re.compile(r'^Train time\(s\)=([0-9eE+\-.]+)$', re.MULTILINE),
    'Test_s': re.compile(r'^Test time\(s\)=([0-9eE+\-.]+)$', re.MULTILINE),
}
SEED_RE = re.compile(r'run\d+_seed(\d+)')


def parse_seed_file(path):
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    row = {}
    for name, pattern in PATTERNS.items():
        match = pattern.search(text)
        row[name] = float(match.group(1)) if match else None
    return row


def fmt(values, scale=1.0):
    vals = [v * scale for v in values if v is not None]
    if not vals:
        return 'n/a'
    mean = statistics.mean(vals)
    if len(vals) > 1:
        return '{:.2f}+-{:.2f}'.format(mean, statistics.stdev(vals))
    return '{:.2f}'.format(mean)


def main():
    rows = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            seed_rows = {}
            for path in sorted(glob.glob(os.path.join(
                    '{}_{}'.format(PREFIX, variant), '*', dataset,
                    'run*_seed*', 'result_*.txt'))):
                seed = int(SEED_RE.search(path).group(1))
                seed_rows[seed] = parse_seed_file(path)
            n = len(seed_rows)
            entry = {'variant': variant, 'dataset': dataset, 'n': n,
                     'seeds': ','.join(str(s) for s in sorted(seed_rows))}
            for name in PATTERNS:
                scale = 100.0 if name in ('OA', 'AA', 'Kpp', 'mIOU') else 1.0
                entry[name] = fmt([r[name] for r in seed_rows.values()], scale)
            rows.append(entry)

    header = ['variant', 'dataset', 'n', 'seeds', 'OA', 'AA', 'Kpp', 'mIOU', 'Train_s', 'Test_s']
    print(('{:<8s} {:<12s} {:>3s} ' + '{:>14s}' * 6).format(
        'variant', 'dataset', 'n', 'OA(%)', 'AA(%)', 'Kpp(%)', 'mIOU(%)', 'Train(s)', 'Test(s)'))
    for row in rows:
        flag = '' if row['n'] >= 10 else (' (partial)' if row['n'] >= 3 else ' (INSUFFICIENT)')
        print(('{:<8s} {:<12s} {:>3d} ' + '{:>14s}' * 6 + '{}').format(
            row['variant'], row['dataset'], row['n'], row['OA'], row['AA'],
            row['Kpp'], row['mIOU'], row['Train_s'], row['Test_s'], flag))

    import csv
    out = '{}_partial_summary.csv'.format(PREFIX)
    with open(out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print('\nwrote {}'.format(out))


if __name__ == '__main__':
    main()
