import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import data_load_operate


DATASETS = [
    ("QUH-Pingan", 10),
    ("QUH-Qingyun", 11),
    ("QUH-Tangdaowan", 12),
]


def parse_int_list(value):
    if isinstance(value, list):
        return value
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def setup_numpy_seed(seed):
    np.random.seed(seed)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def label_masks(height, width, gt_reshape, train_idx, val_idx, test_idx):
    def make_mask(indices):
        mask = np.full(gt_reshape.shape, -1, dtype=np.int16)
        mask[indices] = gt_reshape[indices].astype(np.int16) - 1
        return mask.reshape(height, width)

    return make_mask(train_idx), make_mask(val_idx), make_mask(test_idx)


def export_one_split(out_dir, dataset_name, dataset_index, seed, gt, train_samples, val_samples, save_masks):
    height, width = gt.shape
    gt_reshape = gt.reshape(-1)
    class_count = int(np.max(gt))

    setup_numpy_seed(seed)
    train_idx, val_idx, test_idx, all_idx = data_load_operate.sampling(
        [0.1, 0.01],
        [train_samples, val_samples],
        gt_reshape,
        class_count,
        1,
    )

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    all_idx = np.asarray(all_idx, dtype=np.int64)

    dataset_dir = out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    gt_npz_path = dataset_dir / "gt.npz"
    gt_mat_path = dataset_dir / "gt.mat"
    if not gt_npz_path.exists():
        np.savez(gt_npz_path, gt=gt.astype(np.int16))
    if not gt_mat_path.exists():
        sio.savemat(gt_mat_path, {"gt": gt.astype(np.int16)})

    stem = f"seed{seed}_tr{train_samples}_val{val_samples}"
    npz_path = dataset_dir / f"{stem}.npz"
    mat_path = dataset_dir / f"{stem}.mat"

    metadata = {
        "dataset": dataset_name,
        "dataset_index": dataset_index,
        "seed": seed,
        "height": height,
        "width": width,
        "class_count": class_count,
        "train_samples_per_class": train_samples,
        "val_samples_per_class": val_samples,
        "index_base": 0,
        "index_order": "row_major_flattened",
        "mask_label_encoding": "0_to_K_minus_1_valid_labels_and_minus_1_ignore",
    }

    npz_payload = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "all_idx": all_idx,
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }
    mat_payload = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "all_idx": all_idx,
        "dataset_index": np.asarray([[dataset_index]], dtype=np.int32),
        "seed": np.asarray([[seed]], dtype=np.int32),
        "class_count": np.asarray([[class_count]], dtype=np.int32),
        "train_samples_per_class": np.asarray([[train_samples]], dtype=np.int32),
        "val_samples_per_class": np.asarray([[val_samples]], dtype=np.int32),
    }

    class_train_counts = [
        int(np.sum(gt_reshape[train_idx] == cls + 1)) for cls in range(class_count)
    ]
    class_val_counts = [
        int(np.sum(gt_reshape[val_idx] == cls + 1)) for cls in range(class_count)
    ]
    class_test_counts = [
        int(np.sum(gt_reshape[test_idx] == cls + 1)) for cls in range(class_count)
    ]

    if save_masks:
        train_mask, val_mask, test_mask = label_masks(
            height,
            width,
            gt_reshape,
            train_idx,
            val_idx,
            test_idx,
        )
        npz_payload.update(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        mat_payload.update(
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

    np.savez(npz_path, **npz_payload)
    sio.savemat(mat_path, mat_payload)

    return {
        **metadata,
        "train_count": int(train_idx.size),
        "val_count": int(val_idx.size),
        "test_count": int(test_idx.size),
        "class_train_counts": class_train_counts,
        "class_val_counts": class_val_counts,
        "class_test_counts": class_test_counts,
        "npz": str(npz_path),
        "mat": str(mat_path),
        "contains_masks": bool(save_masks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set_path", type=str, default="./data")
    parser.add_argument("--out_dir", type=str, default="./splits/quh_100_30_seed0-9")
    parser.add_argument("--train_samples", type=int, default=100)
    parser.add_argument("--val_samples", type=int, default=30)
    parser.add_argument("--seed_list", type=parse_int_list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--save_masks", type=str2bool, default=False)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for dataset_name, dataset_index in DATASETS:
        _, gt = data_load_operate.load_data(dataset_name, args.data_set_path)
        for seed in args.seed_list:
            summary.append(
                export_one_split(
                    out_dir=out_dir,
                    dataset_name=dataset_name,
                    dataset_index=dataset_index,
                    seed=seed,
                    gt=gt,
                    train_samples=args.train_samples,
                    val_samples=args.val_samples,
                    save_masks=args.save_masks,
                )
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Exported {len(summary)} splits to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
