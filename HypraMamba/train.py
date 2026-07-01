import argparse
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

import utils.data_load_operate as data_load_operate
from model.MambaHSI import ImprovedMambaHSI as MambaHSI
from utils.HSICommonUtils import ImageStretching
from utils.Loss import resize
from utils.evaluation import Evaluator
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict


DATASET_NAMES = [
    "UP",
    "HanChuan",
    "HongHu",
    "Houston",
    "LongKou",
    "Salinas",
    "indian",
    "Botswana",
    "XuZhou",
    "Pavia",
    "QUH-Pingan",
    "QUH-Qingyun",
    "QUH-Tangdaowan",
]
QUH_DATASETS = {"QUH-Pingan", "QUH-Qingyun", "QUH-Tangdaowan"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_index", type=int, default=8)
    parser.add_argument("--data_set_path", type=str, default="./data")
    parser.add_argument("--work_dir", type=str, default="./")
    parser.add_argument("--exp_name", type=str, default="RUNS")

    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--class_weight_mode", type=str, default="balanced", choices=["none", "balanced"])

    parser.add_argument("--train_samples", type=int, default=30)
    parser.add_argument("--val_samples", type=int, default=10)
    parser.add_argument("--seed_list", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--split_dir", type=str, default="./splits/quh_100_30_seed0-9")
    parser.add_argument("--use_fixed_splits", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--pca_components", type=int, default=30)
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument("--stretch_low", type=float, default=2.0)
    parser.add_argument("--stretch_high", type=float, default=98.0)

    parser.add_argument("--tile_size", type=int, default=0)
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument("--tile_update_groups", type=int, default=1)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--token_num", type=int, default=4)
    parser.add_argument("--group_num", type=int, default=4)
    parser.add_argument("--cls_head_dim", type=int, default=128)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--high_res_skip", type=str, default="none", choices=["none"])
    parser.add_argument("--spectral_diff_alpha", type=float, default=0.5)
    parser.add_argument("--ablation", type=str, default="full", choices=["full", "both", "spa", "spe"])

    parser.add_argument("--record_computecost", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_vis_interval", type=int, default=50)
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seed_list(seed_list):
    seeds = [int(seed.strip()) for seed in seed_list.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("--seed_list must contain at least one seed")
    return seeds


def validate_args(args, dataset_name):
    if args.tile_update_groups <= 0:
        raise ValueError("--tile_update_groups must be positive")
    if args.tile_size < 0:
        raise ValueError("--tile_size must be non-negative")
    if args.tile_size > 0 and args.tile_overlap >= args.tile_size:
        raise ValueError("--tile_overlap must be smaller than --tile_size")
    if args.pool_size < 1:
        raise ValueError("--pool_size must be >= 1")
    if args.hidden_dim % args.group_num != 0:
        raise ValueError("--hidden_dim must be divisible by --group_num")
    if args.cls_head_dim % args.group_num != 0:
        raise ValueError("--cls_head_dim must be divisible by --group_num")
    if dataset_name in QUH_DATASETS and args.tile_size == 0:
        args.tile_size = 512


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    visualize_predict(
        gt_vis,
        pred_vis,
        save_single_predict_path.replace(".png", "_mask.png"),
        save_single_gt_path,
        only_vis_label=True,
    )


def preprocess_image(data, args, logger):
    logger.info(
        "Preprocess: gaussian_sigma={} pca_components={} stretch={}-{}".format(
            args.gaussian_sigma, args.pca_components, args.stretch_low, args.stretch_high
        )
    )
    data = data.astype(np.float32, copy=False)
    data_filtered = gaussian_filter(data, sigma=args.gaussian_sigma) if args.gaussian_sigma > 0 else data

    if args.pca_components <= 0:
        data_pca = data_filtered
    else:
        if args.pca_components > data_filtered.shape[2]:
            raise ValueError("--pca_components cannot exceed input channel count")
        data_reshaped = data_filtered.reshape(-1, data_filtered.shape[2])
        pca = PCA(n_components=args.pca_components)
        data_pca = pca.fit_transform(data_reshaped).reshape(data_filtered.shape[0], data_filtered.shape[1], -1)
    return ImageStretching(data_pca, low=args.stretch_low, high=args.stretch_high)


def build_tiles(height, width, tile_size, tile_overlap):
    stride = tile_size - tile_overlap

    def starts(length):
        if length <= tile_size:
            return [0]
        last_start = length - tile_size
        values = list(range(0, last_start + 1, stride))
        if values[-1] != last_start:
            values.append(last_start)
        return values

    tiles = []
    for y0 in starts(height):
        for x0 in starts(width):
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            tiles.append((y0, y1, x0, x1))
    return tiles


def build_label_mask(gt_reshape, indices, height, width):
    label_map = np.full(height * width, -1, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    label_map[indices] = gt_reshape[indices].astype(np.int64) - 1
    return torch.from_numpy(label_map.reshape(height, width)).long()


def load_fixed_split(split_dir, dataset_name, seed, train_samples, val_samples, height, width):
    split_path = Path(split_dir) / dataset_name / "seed{}_tr{}_val{}.npz".format(seed, train_samples, val_samples)
    if not split_path.exists():
        raise FileNotFoundError(
            "Fixed split not found: {}. Use --use_fixed_splits false to fall back to random sampling.".format(split_path)
        )
    split = np.load(split_path, allow_pickle=True)
    train_idx = split["train_idx"].astype(np.int64)
    val_idx = split["val_idx"].astype(np.int64)
    test_idx = split["test_idx"].astype(np.int64)
    all_idx = split["all_idx"].astype(np.int64)
    max_index = height * width
    for name, values in [("train_idx", train_idx), ("val_idx", val_idx), ("test_idx", test_idx)]:
        if len(values) and (values.min() < 0 or values.max() >= max_index):
            raise ValueError("{} in {} is out of image bounds".format(name, split_path))
    return train_idx, val_idx, test_idx, all_idx, str(split_path)


def get_indices(args, dataset_name, seed, gt_reshape, class_count, height, width):
    if dataset_name in QUH_DATASETS and args.use_fixed_splits:
        return load_fixed_split(
            args.split_dir,
            dataset_name,
            seed,
            args.train_samples,
            args.val_samples,
            height,
            width,
        )
    ratio_list = [0.1, 0.01]
    num_list = [args.train_samples, args.val_samples]
    train_idx, val_idx, test_idx, all_idx = data_load_operate.sampling(ratio_list, num_list, gt_reshape, class_count, 1)
    return train_idx, val_idx, test_idx, all_idx, "random_sampling"


def compute_class_weights(train_label, class_count, mode, device):
    if mode == "none":
        return None
    train_np = train_label.numpy()
    valid = train_np[train_np >= 0]
    counts = np.bincount(valid, minlength=class_count).astype(np.float32)
    weights = np.ones(class_count, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = valid.size / (class_count * counts[nonzero])
    weights[~nonzero] = 0.0
    return torch.from_numpy(weights).float().to(device)


def build_model(args, channels, class_count):
    mamba_type = {"full": "both", "both": "both", "spa": "spa", "spe": "spe"}[args.ablation]
    return MambaHSI(
        in_channels=channels,
        num_classes=class_count,
        hidden_dim=args.hidden_dim,
        token_num=args.token_num,
        group_num=args.group_num,
        pool_size=args.pool_size,
        cls_head_dim=args.cls_head_dim,
        mamba_type=mamba_type,
    )


def build_optimizer(args, net):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_epoch, 1))
    return None


def loss_sum_and_count(loss_func, logits, label):
    seg_logits = resize(input=logits, size=label.shape[1:], mode="bilinear", align_corners=True, warning=False)
    loss_map = loss_func(seg_logits, label)
    valid_count = int((label != -1).sum().item())
    if valid_count == 0:
        return None, 0
    return loss_map.sum(), valid_count


def train_one_epoch_full(net, x_device, train_label, loss_func, optimizer, scaler, device):
    net.train()
    y_train = train_label.unsqueeze(0).to(device)
    optimizer.zero_grad()
    with autocast(enabled=device.type == "cuda"):
        logits = net(x_device)
        loss_sum, valid_count = loss_sum_and_count(loss_func, logits, y_train)
        loss = loss_sum / valid_count
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().cpu().item()), valid_count


def _build_balanced_tile_groups(tile_infos, group_count):
    group_count = max(1, min(group_count, len(tile_infos)))
    groups = [[] for _ in range(group_count)]
    group_weights = [0 for _ in range(group_count)]
    for tile_info in sorted(tile_infos, key=lambda item: item["labeled_count"], reverse=True):
        group_index = int(np.argmin(group_weights))
        groups[group_index].append(tile_info)
        group_weights[group_index] += tile_info["labeled_count"]
    return [group for group in groups if group]


def get_labeled_tile_infos(tiles, train_label):
    tile_infos = []
    for tile in tiles:
        y0, y1, x0, x1 = tile
        labeled_count = int((train_label[y0:y1, x0:x1] != -1).sum().item())
        if labeled_count > 0:
            tile_infos.append({"tile": tile, "labeled_count": labeled_count})
    return tile_infos


def train_one_epoch_tiled(net, x_cpu, train_label, tiles, loss_func, optimizer, scaler, args, device):
    net.train()
    tile_infos = get_labeled_tile_infos(tiles, train_label)
    if not tile_infos:
        raise RuntimeError("No training-labeled pixels found in any tile")
    groups = _build_balanced_tile_groups(tile_infos, args.tile_update_groups)

    epoch_loss_sum = 0.0
    epoch_valid_count = 0
    for group in groups:
        optimizer.zero_grad(set_to_none=True)
        group_valid_count = sum(tile_info["labeled_count"] for tile_info in group)
        for tile_info in group:
            y0, y1, x0, x1 = tile_info["tile"]
            x_tile = x_cpu[:, :, y0:y1, x0:x1].to(device)
            y_tile = train_label[y0:y1, x0:x1].unsqueeze(0).to(device)
            with autocast(enabled=device.type == "cuda"):
                logits = net(x_tile)
                tile_loss_sum, tile_valid_count = loss_sum_and_count(loss_func, logits, y_tile)
            if tile_valid_count == 0:
                continue
            tile_loss = tile_loss_sum / group_valid_count
            scaler.scale(tile_loss).backward()
            epoch_loss_sum += float(tile_loss_sum.detach().cpu().item())
            epoch_valid_count += tile_valid_count

        if group_valid_count == 0:
            continue
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()

    return epoch_loss_sum / max(epoch_valid_count, 1), len(tile_infos), len(groups)


def predict_full(net, x_device, output_size):
    with autocast(enabled=x_device.device.type == "cuda"):
        output = net(x_device)
    seg_logits = resize(input=output, size=output_size, mode="bilinear", align_corners=True, warning=False)
    return torch.argmax(seg_logits, dim=1).detach().cpu().numpy()[0]


def predict_tiled(net, x_cpu, tiles, class_count, height, width, device):
    logits_sum = torch.zeros((class_count, height, width), dtype=torch.float32)
    count_sum = torch.zeros((height, width), dtype=torch.float32)
    for y0, y1, x0, x1 in tiles:
        x_tile = x_cpu[:, :, y0:y1, x0:x1].to(device)
        with autocast(enabled=device.type == "cuda"):
            output = net(x_tile)
        seg_logits = resize(input=output, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=True, warning=False)
        logits_sum[:, y0:y1, x0:x1] += seg_logits.squeeze(0).detach().float().cpu()
        count_sum[y0:y1, x0:x1] += 1.0
        del x_tile, output, seg_logits
    logits_sum /= count_sum.clamp_min(1.0).unsqueeze(0)
    return torch.argmax(logits_sum, dim=0).numpy()


def evaluate_prediction(label, predict, class_count):
    evaluator = Evaluator(num_class=class_count)
    label_np = label.cpu().numpy()
    label_np = np.where(label_np == -1, 255, label_np)
    evaluator.add_batch(np.expand_dims(label_np, axis=0), np.expand_dims(predict, axis=0))
    oa = evaluator.Pixel_Accuracy()
    miou, iou = evaluator.Mean_Intersection_over_Union()
    aa, acc = evaluator.Pixel_Accuracy_Class()
    kappa = evaluator.Kappa()
    return oa, aa, kappa, miou, iou, acc


def evaluate_model(net, x_cpu, x_device, label, tiles, class_count, height, width, use_tiles, device):
    net.eval()
    with torch.no_grad():
        if use_tiles:
            predict = predict_tiled(net, x_cpu, tiles, class_count, height, width, device)
        else:
            predict = predict_full(net, x_device, label.shape)
    metrics = evaluate_prediction(label, predict, class_count)
    return predict, metrics


def log_compute_cost(args, logger, net, channels, height, width, use_tiles):
    try:
        from calflops import calculate_flops
    except ImportError:
        logger.warning("calflops is not installed; skip compute-cost logging")
        return
    input_h = args.tile_size if use_tiles else height
    input_w = args.tile_size if use_tiles else width
    flops, macs, params = calculate_flops(model=net, input_shape=(1, channels, input_h, input_w))
    logger.info("params:{} flops:{} macs:{}".format(params, flops, macs))


def main():
    args = get_parser()
    if args.dataset_index < 0 or args.dataset_index >= len(DATASET_NAMES):
        raise ValueError("--dataset_index must be in [0, {}]".format(len(DATASET_NAMES) - 1))
    dataset_name = DATASET_NAMES[args.dataset_index]
    validate_args(args, dataset_name)
    seed_list = parse_seed_list(args.seed_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_name = "MambaHSI"
    num_list = [args.train_samples, args.val_samples]
    use_tiles = args.tile_size > 0

    save_folder = os.path.join(args.work_dir, args.exp_name, net_name, dataset_name)
    os.makedirs(save_folder, exist_ok=True)
    save_log_path = os.path.join(save_folder, "train_tr{}_val{}.log".format(num_list[0], num_list[1]))
    logger = setup_logger(name=dataset_name, logfile=save_log_path)
    logger.info(save_folder)
    logger.info(vars(args))

    data, gt = data_load_operate.load_data(dataset_name, args.data_set_path)
    img = preprocess_image(data, args, logger)
    height, width, channels = img.shape
    gt_reshape = gt.reshape(-1)
    class_count = int(max(np.unique(gt)))
    transform = transforms.Compose([transforms.ToTensor()])
    x_cpu = transform(np.array(img)).unsqueeze(0).float()
    x_device = None if use_tiles else x_cpu.to(device)
    tiles = build_tiles(height, width, args.tile_size, args.tile_overlap) if use_tiles else []
    if use_tiles:
        logger.info(
            "Tile mode: tile_size={} tile_overlap={} tiles={}".format(
                args.tile_size, args.tile_overlap, len(tiles)
            )
        )

    paras_dict = {
        "net_name": net_name,
        "dataset_index": args.dataset_index,
        "dataset_name": dataset_name,
        "num_list": num_list,
        "lr": args.lr,
        "seed_list": seed_list,
        "class_count": class_count,
    }

    oa_all = []
    aa_all = []
    kpp_all = []
    miou_all = []
    each_acc_all = []
    train_time_all = []
    test_time_all = []

    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)
        single_experiment_name = "run{}_seed{}".format(exp_idx, curr_seed)
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        save_vis_folder = os.path.join(save_single_experiment_folder, "vis")
        os.makedirs(save_vis_folder, exist_ok=True)

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, "result_tr{}_val{}.txt".format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, "pred_vis_tr{}_val{}.png".format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, "gt_vis_tr{}_val{}.png".format(num_list[0], num_list[1]))

        train_idx, val_idx, test_idx, all_idx, split_source = get_indices(
            args, dataset_name, curr_seed, gt_reshape, class_count, height, width
        )
        train_label = build_label_mask(gt_reshape, train_idx, height, width)
        val_label = build_label_mask(gt_reshape, val_idx, height, width)
        test_label = build_label_mask(gt_reshape, test_idx, height, width)
        logger.info(
            "Seed {} split={} train={} val={} test={}".format(
                curr_seed, split_source, len(train_idx), len(val_idx), len(test_idx)
            )
        )

        net = build_model(args, channels, class_count).to(device)
        logger.info(paras_dict)
        logger.info(net)
        optimizer = build_optimizer(args, net)
        scheduler = build_scheduler(args, optimizer)
        class_weights = compute_class_weights(train_label, class_count, args.class_weight_mode, device)
        loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
            reduction="none",
        )
        scaler = GradScaler(enabled=device.type == "cuda")
        logger.info(optimizer)

        if args.record_computecost:
            log_compute_cost(args, logger, net, channels, height, width, use_tiles)

        tic1 = time.perf_counter()
        best_val_acc = -1.0
        best_epoch = 0

        for epoch in range(args.max_epoch):
            if use_tiles:
                train_loss, train_tiles, step_groups = train_one_epoch_tiled(
                    net, x_cpu, train_label, tiles, loss_func, optimizer, scaler, args, device
                )
                train_extra = "tiles:{}|tile_update_groups:{}".format(train_tiles, step_groups)
            else:
                train_loss, trained_pixels = train_one_epoch_full(
                    net, x_device, train_label, loss_func, optimizer, scaler, device
                )
                train_extra = "pixels:{}".format(trained_pixels)

            if scheduler is not None:
                scheduler.step()

            predict_val, val_metrics = evaluate_model(
                net, x_cpu, x_device, val_label, tiles, class_count, height, width, use_tiles, device
            )
            oa, aa, kappa, miou, iou, acc = val_metrics
            logger.info(
                "Evaluate {}|loss:{:.6f}|OA:{:.6f}|AA:{:.6f}|Kappa:{:.6f}|mIoU:{:.6f}|{}".format(
                    epoch, train_loss, oa, aa, kappa, miou, train_extra
                )
            )

            if oa >= best_val_acc:
                best_epoch = epoch + 1
                best_val_acc = oa
                torch.save(net.state_dict(), save_weight_path)

            if args.save_vis_interval > 0 and (epoch + 1) % args.save_vis_interval == 0:
                save_single_predict_path = os.path.join(save_vis_folder, "predict_{}.png".format(epoch + 1))
                save_single_gt_path = os.path.join(save_vis_folder, "gt.png")
                vis_a_image(gt, predict_val, save_single_predict_path, save_single_gt_path)
            torch.cuda.empty_cache()

        train_time = time.perf_counter() - tic1
        logger.info("train_time: {} seconds".format(train_time))

        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        tic2 = time.perf_counter()
        best_net = build_model(args, channels, class_count).to(device)
        best_net.load_state_dict(torch.load(save_weight_path, map_location=device))
        predict_test, test_metrics = evaluate_model(
            best_net, x_cpu, x_device, test_label, tiles, class_count, height, width, use_tiles, device
        )
        oa_test, aa_test, kappa_test, miou_test, iou_test, acc_test = test_metrics
        logger.info(
            "Test best_epoch:{}|OA:{:.6f}|AA:{:.6f}|Kappa:{:.6f}|mIoU:{:.6f}".format(
                best_epoch, oa_test, aa_test, kappa_test, miou_test
            )
        )
        vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
        test_time = time.perf_counter() - tic2

        str_results = (
            "\n======================"
            + " exp_idx="
            + str(exp_idx)
            + " seed="
            + str(curr_seed)
            + " learning rate="
            + str(args.lr)
            + " epochs="
            + str(args.max_epoch)
            + " train samples="
            + str(args.train_samples)
            + " val samples="
            + str(args.val_samples)
            + " split="
            + split_source
            + " ======================"
            + "\nOA="
            + str(oa_test)
            + "\nAA="
            + str(aa_test)
            + "\nkpp="
            + str(kappa_test)
            + "\nmIOU_test:"
            + str(miou_test)
            + "\nIOU_test:"
            + str(iou_test)
            + "\nAcc_test:"
            + str(acc_test)
            + "\n"
        )
        logger.info(str_results)
        with open(results_save_path, "a+") as f:
            f.write(str_results)

        oa_all.append(oa_test)
        aa_all.append(aa_test)
        kpp_all.append(kappa_test)
        miou_all.append(miou_test)
        each_acc_all.append(acc_test)
        train_time_all.append(train_time)
        test_time_all.append(test_time)

        del net, best_net
        torch.cuda.empty_cache()

    oa_all = np.array(oa_all)
    aa_all = np.array(aa_all)
    kpp_all = np.array(kpp_all)
    miou_all = np.array(miou_all)
    each_acc_all = np.array(each_acc_all)
    train_time_all = np.array(train_time_all)
    test_time_all = np.array(test_time_all)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    logger.info("List of OA: {}".format(list(oa_all)))
    logger.info("List of AA: {}".format(list(aa_all)))
    logger.info("List of KPP: {}".format(list(kpp_all)))
    logger.info("List of mIoU: {}".format(list(miou_all)))
    logger.info("OA: {:.2f} ± {:.2f}".format(np.mean(oa_all) * 100, np.std(oa_all) * 100))
    logger.info("AA: {:.2f} ± {:.2f}".format(np.mean(aa_all) * 100, np.std(aa_all) * 100))
    logger.info("Kpp: {:.2f} ± {:.2f}".format(np.mean(kpp_all) * 100, np.std(kpp_all) * 100))
    logger.info("mIoU: {:.2f} ± {:.2f}".format(np.mean(miou_all) * 100, np.std(miou_all) * 100))
    logger.info(
        "Acc per class: {} ± {}".format(
            np.round(np.mean(each_acc_all, 0) * 100, decimals=2).tolist(),
            np.round(np.std(each_acc_all, 0) * 100, decimals=2).tolist(),
        )
    )
    logger.info(
        "Average training time: {:.2f} ± {:.3f}".format(np.mean(train_time_all), np.std(train_time_all))
    )
    logger.info(
        "Average testing time: {:.2f} ± {:.3f}".format(
            np.mean(test_time_all) * 1000, np.std(test_time_all) * 1000
        )
    )

    mean_result_path = os.path.join(save_folder, "mean_result.txt")
    with open(mean_result_path, "w") as f:
        str_results = (
            "\n\n***************Mean result of "
            + str(len(seed_list))
            + " times runs ********************"
            + "\nList of OA:"
            + str(list(oa_all))
            + "\nList of AA:"
            + str(list(aa_all))
            + "\nList of KPP:"
            + str(list(kpp_all))
            + "\nList of mIoU:"
            + str(list(miou_all))
            + "\nOA="
            + str(round(np.mean(oa_all) * 100, 2))
            + "+-"
            + str(round(np.std(oa_all) * 100, 2))
            + "\nAA="
            + str(round(np.mean(aa_all) * 100, 2))
            + "+-"
            + str(round(np.std(aa_all) * 100, 2))
            + "\nKpp="
            + str(round(np.mean(kpp_all) * 100, 2))
            + "+-"
            + str(round(np.std(kpp_all) * 100, 2))
            + "\nmIoU="
            + str(round(np.mean(miou_all) * 100, 2))
            + "+-"
            + str(round(np.std(miou_all) * 100, 2))
            + "\nAcc per class=\n"
            + str(np.round(np.mean(each_acc_all, 0) * 100, 2))
            + "+-"
            + str(np.round(np.std(each_acc_all, 0) * 100, 2))
            + "\nAverage training time="
            + str(np.round(np.mean(train_time_all), decimals=2))
            + "+-"
            + str(np.round(np.std(train_time_all), decimals=3))
            + "\nAverage testing time="
            + str(np.round(np.mean(test_time_all) * 1000, decimals=2))
            + "+-"
            + str(np.round(np.std(test_time_all) * 1000, decimals=3))
        )
        f.write(str_results)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
