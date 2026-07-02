import argparse
import re
from pathlib import Path

import numpy as np


def parse_vector(text, key):
    match = re.search(r'\n{}:\[([^\]]+)\]'.format(re.escape(key)), text, re.S)
    if match is None:
        raise ValueError('Missing {} vector.'.format(key))
    return np.fromstring(match.group(1).replace('\n', ' '), sep=' ')


def parse_result_file(path):
    text = path.read_text()
    seed_match = re.search(r'seed=(\d+)', text)
    oa_match = re.search(r'\nOA=([0-9.eE+-]+)', text)
    return {
        'path': path,
        'seed': int(seed_match.group(1)) if seed_match else -1,
        'oa': float(oa_match.group(1)) if oa_match else float('nan'),
        'acc': parse_vector(text, 'Acc_test'),
        'iou': parse_vector(text, 'IOU_test'),
    }


def summarize_dataset(dataset_dir, top_k):
    rows = [
        parse_result_file(path)
        for path in sorted(dataset_dir.glob('run*_seed*/result_tr100_val30.txt'))
    ]
    if not rows:
        return []

    recall = np.stack([row['acc'] for row in rows]).mean(axis=0)
    recall_std = np.stack([row['acc'] for row in rows]).std(axis=0)
    iou = np.stack([row['iou'] for row in rows]).mean(axis=0)
    iou_std = np.stack([row['iou'] for row in rows]).std(axis=0)

    precision = 1.0 / (
        1.0 / np.maximum(iou, 1e-9)
        - 1.0 / np.maximum(recall, 1e-9)
        + 1.0
    )
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-9)

    records = []
    for class_idx in np.argsort(iou)[:top_k]:
        if recall[class_idx] < 0.93 and precision[class_idx] >= 0.90:
            diagnosis = 'recall-weak'
        elif precision[class_idx] < 0.80 and recall[class_idx] >= 0.93:
            diagnosis = 'precision-weak'
        elif recall[class_idx] < 0.93 and precision[class_idx] < 0.90:
            diagnosis = 'mixed-weak'
        else:
            diagnosis = 'secondary'
        records.append({
            'dataset': dataset_dir.name,
            'class_id': class_idx + 1,
            'recall': recall[class_idx],
            'recall_std': recall_std[class_idx],
            'precision_est': precision[class_idx],
            'iou': iou[class_idx],
            'iou_std': iou_std[class_idx],
            'f1_est': f1[class_idx],
            'diagnosis': diagnosis,
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_root',
        type=Path,
        default=Path('RUNS_QUH_100_30_WEIGHTED_ACCUM_GROUP2')
        / 'MambaHSI_competitive_diff_alpha0p5',
    )
    parser.add_argument('--top_k', type=int, default=8)
    args = parser.parse_args()

    print('dataset,class_id,recall,recall_std,precision_est,iou,iou_std,f1_est,diagnosis')
    for dataset_dir in sorted(args.result_root.glob('QUH-*')):
        for record in summarize_dataset(dataset_dir, args.top_k):
            print(
                '{dataset},{class_id},{recall:.4f},{recall_std:.4f},'
                '{precision_est:.4f},{iou:.4f},{iou_std:.4f},{f1_est:.4f},{diagnosis}'.format(
                    **record
                )
            )


if __name__ == '__main__':
    main()
