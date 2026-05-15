import os
import time
import random
import argparse
import numpy as np
import torch
import utils.data_load_operate as data_load_operate
from utils.Loss import head_loss, resize
from utils.evaluation import Evaluator
from utils.HSICommonUtils import ImageStretching
from utils.setup_logger import setup_logger
from utils.visual_predict import visualize_predict
from model.MambaHSI import ImprovedMambaHSI as MambaHSI, VALID_ABLATIONS, VALID_OUTER_RESIDUAL_MODES
from calflops import calculate_flops
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'
scaler = GradScaler(enabled=torch.cuda.is_available())
FUSION_NAME = 'competitive'


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    visualize_predict(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=only_vis_label)
    visualize_predict(gt_vis, pred_vis, save_single_predict_path.replace('.png', '_mask.png'), save_single_gt_path, only_vis_label=True)

# random seed setting 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)  # 设置 CPU 上的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    random.seed(seed)  # 设置 Python 内建 random 模块的随机种子
    torch.backends.cudnn.deterministic = True  # 确保每次计算的结果是确定性的
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 自动优化，确保每次运行的一致性


def compute_balanced_class_weights(train_label, class_count, target_device):
    valid_labels = train_label[train_label >= 0].long().view(-1)
    class_counts = torch.bincount(valid_labels, minlength=class_count).float()
    class_weights = torch.zeros(class_count, dtype=torch.float32)
    nonzero_mask = class_counts > 0

    if nonzero_mask.any():
        total_valid = class_counts[nonzero_mask].sum()
        class_weights[nonzero_mask] = total_valid / (nonzero_mask.sum() * class_counts[nonzero_mask])

    return class_weights.to(target_device), class_counts.long().tolist()


def parse_hard_pairs(pair_str):
    """
    User input uses 1-based class ids, same as Acc per class display.
    Example: "2-3,10-11,11-12"
    Return: [(1, 2), (9, 10), (10, 11)] in 0-based labels.
    """
    pairs = []
    if pair_str is None or pair_str.strip() == '':
        return pairs

    for item in pair_str.split(','):
        item = item.strip()
        if not item:
            continue
        a, b = item.split('-')
        a = int(a.strip()) - 1
        b = int(b.strip()) - 1
        if a != b:
            pairs.append((a, b))
    return pairs


def to_numpy_label(label):
    if torch.is_tensor(label):
        return label.detach().cpu().numpy()
    return np.asarray(label)


def auto_select_spectral_nearest_pairs(raw_data, train_label_np, class_count, max_pairs=4, eps=1e-6):
    """
    Select spectrally nearest class pairs using only training pixels.
    This avoids using test-set confusion when you want a cleaner protocol.
    """
    raw = np.asarray(raw_data, dtype=np.float32)
    H, W, B = raw.shape
    raw_flat = raw.reshape(-1, B)
    label_flat = train_label_np.reshape(-1)

    train_mask = label_flat >= 0
    if train_mask.sum() == 0:
        return []

    mu = raw_flat[train_mask].mean(axis=0, keepdims=True)
    std = raw_flat[train_mask].std(axis=0, keepdims=True) + eps
    raw_norm = (raw_flat - mu) / std

    class_means = []
    valid_classes = []

    for c in range(class_count):
        idx = label_flat == c
        if idx.sum() > 0:
            class_means.append(raw_norm[idx].mean(axis=0))
            valid_classes.append(c)

    if len(valid_classes) < 2:
        return []

    class_means = np.stack(class_means, axis=0)

    pair_dist = []
    for i in range(len(valid_classes)):
        for j in range(i + 1, len(valid_classes)):
            dist = np.mean((class_means[i] - class_means[j]) ** 2)
            pair_dist.append((dist, valid_classes[i], valid_classes[j]))

    pair_dist.sort(key=lambda x: x[0])
    pairs = [(a, b) for _, a, b in pair_dist[:max_pairs]]
    return pairs


def auto_select_spectral_overlap_pairs(raw_data, train_label_np, class_count, max_pairs=4, eps=1e-6):
    """
    Select hard pairs by normalized spectral overlap using only training pixels.
    Smaller distance means closer class means under larger within-class variance.
    """
    raw = np.asarray(raw_data, dtype=np.float32)
    H, W, B = raw.shape

    raw_flat = raw.reshape(-1, B)
    label_flat = train_label_np.reshape(-1)

    train_mask = label_flat >= 0
    if train_mask.sum() == 0:
        return []

    mu_all = raw_flat[train_mask].mean(axis=0, keepdims=True)
    std_all = raw_flat[train_mask].std(axis=0, keepdims=True) + eps
    raw_norm = (raw_flat - mu_all) / std_all

    class_stats = {}
    for c in range(class_count):
        idx = label_flat == c
        if idx.sum() < 2:
            continue

        x = raw_norm[idx]
        class_stats[c] = {
            'mean': x.mean(axis=0),
            'var': x.var(axis=0),
        }

    classes = sorted(class_stats.keys())
    if len(classes) < 2:
        return []

    pair_dist = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a = classes[i]
            b = classes[j]

            ma = class_stats[a]['mean']
            mb = class_stats[b]['mean']
            va = class_stats[a]['var']
            vb = class_stats[b]['var']

            dist = np.mean((ma - mb) ** 2 / (va + vb + eps))
            pair_dist.append((dist, a, b))

    pair_dist.sort(key=lambda x: x[0])
    return [(a, b) for _, a, b in pair_dist[:max_pairs]]


def compute_pair_band_score(feat, train_label_np, hard_pairs, eps=1e-6):
    """
    feat: [H, W, C]
    return score: [C]
    """
    H, W, C = feat.shape
    feat_flat = feat.reshape(-1, C)
    label_flat = train_label_np.reshape(-1)

    score = np.zeros(C, dtype=np.float32)

    for cls_a, cls_b in hard_pairs:
        mask_a = label_flat == cls_a
        mask_b = label_flat == cls_b

        if mask_a.sum() < 2 or mask_b.sum() < 2:
            continue

        xa = feat_flat[mask_a]
        xb = feat_flat[mask_b]

        ma = xa.mean(axis=0)
        mb = xb.mean(axis=0)
        va = xa.var(axis=0)
        vb = xb.var(axis=0)

        pair_score = np.abs(ma - mb) / np.sqrt(va + vb + eps)
        score += pair_score.astype(np.float32)

    return score


def bootstrap_candidate_score(candidate_stack, train_label_np, hard_pairs,
                              n_boot=20, sample_ratio=0.8, seed=123, eps=1e-6):
    if n_boot <= 0:
        return compute_pair_band_score(candidate_stack, train_label_np, hard_pairs, eps=eps)

    H, W, C = candidate_stack.shape
    feat_flat = candidate_stack.reshape(-1, C)
    label_flat = train_label_np.reshape(-1)

    rng = np.random.default_rng(seed)
    all_scores = []

    for _ in range(n_boot):
        score_sum = np.zeros(C, dtype=np.float32)

        for cls_a, cls_b in hard_pairs:
            idx_a = np.where(label_flat == cls_a)[0]
            idx_b = np.where(label_flat == cls_b)[0]

            if len(idx_a) < 2 or len(idx_b) < 2:
                continue

            na = max(2, int(len(idx_a) * sample_ratio))
            nb = max(2, int(len(idx_b) * sample_ratio))

            sub_a = rng.choice(idx_a, size=na, replace=True)
            sub_b = rng.choice(idx_b, size=nb, replace=True)

            xa = feat_flat[sub_a]
            xb = feat_flat[sub_b]

            ma = xa.mean(axis=0)
            mb = xb.mean(axis=0)
            va = xa.var(axis=0)
            vb = xb.var(axis=0)

            score = np.abs(ma - mb) / np.sqrt(va + vb + eps)
            score_sum += score.astype(np.float32)

        all_scores.append(score_sum)

    all_scores = np.stack(all_scores, axis=0)
    mean_score = all_scores.mean(axis=0)
    std_score = all_scores.std(axis=0)

    stable_score = mean_score - 0.5 * std_score
    return stable_score


def select_diverse_candidates(candidate_stack, scores, train_label_np, kmax=8, corr_thr=0.95):
    H, W, C = candidate_stack.shape

    label_flat = train_label_np.reshape(-1)
    train_mask = label_flat >= 0

    if train_mask.sum() == 0:
        return [int(idx) for idx in np.argsort(scores)[::-1][:kmax]]

    feat = candidate_stack.reshape(-1, C)[train_mask]
    order = np.argsort(scores)[::-1]
    selected = []

    for idx in order:
        idx = int(idx)

        if len(selected) == 0:
            selected.append(idx)
        else:
            x = feat[:, idx]
            keep = True

            for j in selected:
                y = feat[:, j]

                if np.std(x) < 1e-6 or np.std(y) < 1e-6:
                    corr = 0.0
                else:
                    corr = np.corrcoef(x, y)[0, 1]

                if np.isnan(corr):
                    corr = 0.0

                if abs(corr) > corr_thr:
                    keep = False
                    break

            if keep:
                selected.append(idx)

        if len(selected) >= kmax:
            break

    return selected


def build_hpsdi_maps(raw_data, train_label, class_count, sdi_kmax=4,
                     sdi_pairs='', sdi_top_m=10, sdi_mode='slope',
                     sdi_max_pairs=4, sdi_pair_metric='overlap',
                     sdi_slope_source='all', sdi_score_mode='bootstrap',
                     sdi_boot_n=20, sdi_boot_ratio=0.8,
                     sdi_boot_seed=123, sdi_corr_thr=0.95, eps=1e-6):
    """
    raw_data: gaussian_filter output before PCA, [H, W, raw_bands]
    train_label: [H, W], 0-based, ignore=-1
    return:
        selected_maps: [H, W, <=sdi_kmax], normalized to roughly [-1, 1]
        candidate_sdi_names: list
        hard_pairs: list of 0-based pairs
    """
    if sdi_kmax <= 0:
        return None, [], []

    raw = np.asarray(raw_data, dtype=np.float32)
    train_label_np = to_numpy_label(train_label).astype(np.int64)

    hard_pairs = parse_hard_pairs(sdi_pairs)
    if len(hard_pairs) == 0:
        if sdi_pair_metric == 'overlap':
            hard_pairs = auto_select_spectral_overlap_pairs(
                raw,
                train_label_np,
                class_count,
                max_pairs=sdi_max_pairs,
                eps=eps
            )
        else:
            hard_pairs = auto_select_spectral_nearest_pairs(
                raw,
                train_label_np,
                class_count,
                max_pairs=sdi_max_pairs,
                eps=eps
            )

    if len(hard_pairs) == 0:
        return None, [], []

    H, W, B = raw.shape

    candidate_maps = []
    candidate_names = []
    candidate_seen = set()

    def add_candidate(candidate_map, candidate_name):
        if candidate_name in candidate_seen:
            return
        candidate_maps.append(candidate_map)
        candidate_names.append(candidate_name)
        candidate_seen.add(candidate_name)

    top_bands = None
    if sdi_slope_source == 'all':
        for i in range(B - 1):
            add_candidate(raw[:, :, i + 1] - raw[:, :, i], ('slope', i + 1, i))
    else:
        band_score = compute_pair_band_score(raw, train_label_np, hard_pairs, eps=eps)
        top_m = min(sdi_top_m, B)
        top_bands = np.argsort(band_score)[::-1][:top_m]

        for i in top_bands:
            i = int(i)

            if i > 0:
                add_candidate(raw[:, :, i] - raw[:, :, i - 1], ('slope', i, i - 1))

            if i < B - 1:
                add_candidate(raw[:, :, i + 1] - raw[:, :, i], ('slope', i + 1, i))

    if sdi_mode == 'slope_ratio':
        if top_bands is None:
            band_score = compute_pair_band_score(raw, train_label_np, hard_pairs, eps=eps)
            top_m = min(sdi_top_m, B)
            top_bands = np.argsort(band_score)[::-1][:top_m]

        for m in range(len(top_bands)):
            for n in range(m + 1, len(top_bands)):
                i = int(top_bands[m])
                j = int(top_bands[n])
                xi = raw[:, :, i]
                xj = raw[:, :, j]
                ratio = (xi - xj) / (np.abs(xi) + np.abs(xj) + eps)
                add_candidate(ratio, ('ratio', i, j))

    if len(candidate_maps) == 0:
        return None, [], hard_pairs

    candidate_stack = np.stack(candidate_maps, axis=-1).astype(np.float32)
    candidate_stack = np.nan_to_num(candidate_stack, nan=0.0, posinf=0.0, neginf=0.0)

    if sdi_score_mode == 'bootstrap':
        cand_score = bootstrap_candidate_score(
            candidate_stack,
            train_label_np,
            hard_pairs,
            n_boot=sdi_boot_n,
            sample_ratio=sdi_boot_ratio,
            seed=sdi_boot_seed,
            eps=eps
        )
    else:
        cand_score = compute_pair_band_score(candidate_stack, train_label_np, hard_pairs, eps=eps)

    if sdi_corr_thr > 0:
        selected_idx = select_diverse_candidates(
            candidate_stack,
            cand_score,
            train_label_np,
            kmax=sdi_kmax,
            corr_thr=sdi_corr_thr
        )
    else:
        select_k = min(sdi_kmax, candidate_stack.shape[-1])
        selected_idx = [int(idx) for idx in np.argsort(cand_score)[::-1][:select_k]]

    if len(selected_idx) == 0:
        return None, [], hard_pairs

    selected_maps = candidate_stack[:, :, selected_idx]
    candidate_sdi_names = [candidate_names[int(i)] for i in selected_idx]

    train_mask = train_label_np >= 0
    if train_mask.sum() > 0:
        mu = selected_maps[train_mask].mean(axis=0, keepdims=True)
        std = selected_maps[train_mask].std(axis=0, keepdims=True) + eps
        selected_maps = (selected_maps - mu.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

    selected_maps = np.clip(selected_maps, -3.0, 3.0) / 3.0
    selected_maps = selected_maps.astype(np.float32)

    return selected_maps, candidate_sdi_names, hard_pairs


def build_input_tensor_with_hpsdi(img_pca_stretched, raw_filtered, train_label,
                                  class_count, args, transform):
    """
    Keep the original PCA-30 transform path unchanged, then append SDI tensor channels.
    """
    x_pca = transform(np.array(img_pca_stretched))

    if args.sdi_kmax <= 0:
        return x_pca.unsqueeze(0).float(), [], []

    sdi_maps, candidate_sdi_names, hard_pairs = build_hpsdi_maps(
        raw_data=raw_filtered,
        train_label=train_label,
        class_count=class_count,
        sdi_kmax=args.sdi_kmax,
        sdi_pairs=args.sdi_pairs,
        sdi_top_m=args.sdi_top_m,
        sdi_mode=args.sdi_mode,
        sdi_max_pairs=args.sdi_max_pairs,
        sdi_pair_metric=args.sdi_pair_metric,
        sdi_slope_source=args.sdi_slope_source,
        sdi_score_mode=args.sdi_score_mode,
        sdi_boot_n=args.sdi_boot_n,
        sdi_boot_ratio=args.sdi_boot_ratio,
        sdi_boot_seed=args.sdi_boot_seed,
        sdi_corr_thr=args.sdi_corr_thr
    )

    if sdi_maps is None:
        return x_pca.unsqueeze(0).float(), [], hard_pairs

    x_sdi = torch.from_numpy(sdi_maps.transpose(2, 0, 1)).float()
    x = torch.cat([x_pca.float(), x_sdi], dim=0).unsqueeze(0)

    return x.float(), candidate_sdi_names, hard_pairs


def zero_init_extra_patch_channels(model, base_channels=30):
    conv = model.patch_embedding[0]
    if not isinstance(conv, torch.nn.Conv2d):
        return

    if conv.in_channels > base_channels:
        with torch.no_grad():
            conv.weight[:, base_channels:, :, :].zero_()


def sdi_group_lasso_loss(model, base_channels=30):
    conv = model.patch_embedding[0]
    if not isinstance(conv, torch.nn.Conv2d):
        return next(model.parameters()).new_tensor(0.0)

    if conv.in_channels <= base_channels:
        return conv.weight.new_tensor(0.0)

    W_sdi = conv.weight[:, base_channels:, :, :]
    norm_per_channel = torch.sqrt(torch.sum(W_sdi ** 2, dim=(0, 2, 3)) + 1e-12)
    return norm_per_channel.sum()


def add_sdi_group_lasso(loss, model, args, base_channels=30):
    if getattr(args, 'sdi_kmax', 0) > 0 and getattr(args, 'sdi_group_lasso', 0.0) > 0:
        loss = loss + args.sdi_group_lasso * sdi_group_lasso_loss(model, base_channels)
    return loss


@torch.no_grad()
def get_sdi_channel_importance(model, candidate_names, base_channels=30, relative_thr=0.05):
    conv = model.patch_embedding[0]
    if not isinstance(conv, torch.nn.Conv2d):
        return [], 0

    if conv.in_channels <= base_channels:
        return [], 0

    W = conv.weight
    W_pca = W[:, :base_channels, :, :]
    W_sdi = W[:, base_channels:, :, :]

    pca_norm = torch.sqrt(torch.sum(W_pca ** 2, dim=(0, 2, 3)) + 1e-12)
    sdi_norm = torch.sqrt(torch.sum(W_sdi ** 2, dim=(0, 2, 3)) + 1e-12)

    ref = pca_norm.mean().clamp_min(1e-12)
    rel = (sdi_norm / ref).detach().cpu().numpy()
    abs_norm = sdi_norm.detach().cpu().numpy()

    result = []
    for i, value in enumerate(rel):
        name = candidate_names[i] if i < len(candidate_names) else 'sdi_{}'.format(i)
        result.append((i, name, float(abs_norm[i]), float(value), bool(value > relative_thr)))

    effective_num = sum([item[-1] for item in result])
    return result, effective_num


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int, default=8)
    parser.add_argument('--data_set_path', type=str, default='./data')
    parser.add_argument('--work_dir', type=str, default='./')

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--train_samples', type=int, default=30)
    parser.add_argument('--val_samples', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='RUNS_sdi')
    parser.add_argument('--record_computecost', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--class_weight_mode', type=str, default='none', choices=['auto', 'none', 'balanced'])
    parser.add_argument('--pyramid_dilation', type=str, default='3')
    parser.add_argument('--token_num', type=int, default=4)
    parser.add_argument('--sdi_kmax', '--sdi_k', dest='sdi_kmax', type=int, default=0,
                        help='max number of SDI candidate channels; 0 means baseline')
    parser.add_argument('--sdi_mode', type=str, default='slope', choices=['slope', 'slope_ratio'],
                        help='slope: only local spectral difference; slope_ratio: slope + band ratio')
    parser.add_argument('--sdi_pairs', type=str, default='',
                        help='1-based hard class pairs, e.g. "2-3,10-11,11-12". Empty means auto spectral-nearest pairs.')
    parser.add_argument('--sdi_top_m', type=int, default=10,
                        help='top raw bands used to generate slope/ratio candidates')
    parser.add_argument('--sdi_max_pairs', type=int, default=4,
                        help='max auto-selected spectral-nearest class pairs')
    parser.add_argument('--sdi_pair_metric', type=str, default='overlap', choices=['nearest', 'overlap'],
                        help='auto hard-pair metric: nearest uses class mean distance; overlap normalizes by class variance')
    parser.add_argument('--sdi_slope_source', type=str, default='all', choices=['top_bands', 'all'],
                        help='slope candidates from all adjacent bands or only around top raw bands')
    parser.add_argument('--sdi_score_mode', type=str, default='bootstrap', choices=['single', 'bootstrap'],
                        help='candidate scoring mode')
    parser.add_argument('--sdi_boot_n', type=int, default=20,
                        help='number of bootstrap rounds for stable SDI candidate scoring')
    parser.add_argument('--sdi_boot_ratio', type=float, default=0.8,
                        help='bootstrap sampling ratio per hard-pair class')
    parser.add_argument('--sdi_boot_seed', type=int, default=123,
                        help='random seed for bootstrap SDI candidate scoring')
    parser.add_argument('--sdi_corr_thr', type=float, default=0.95,
                        help='absolute train-pixel correlation threshold for SDI candidate de-redundancy; <=0 disables it')
    parser.add_argument('--sdi_group_lasso', type=float, default=1e-5,
                        help='group lasso strength for SDI input channels')
    parser.add_argument('--sdi_prune_thr', type=float, default=0.05,
                        help='relative threshold used only for logging effective selected SDI channels')
    parser.add_argument('--ablation', type=str, default='full', choices=sorted(VALID_ABLATIONS))
    parser.add_argument('--outer_residual_mode', type=str, default='standard', choices=sorted(VALID_OUTER_RESIDUAL_MODES))
    parser.add_argument('--outer_residual_alpha', type=float, default=1.0)

    args = parser.parse_args()
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args \
    = get_parser()
record_computecost = args.record_computecost
seed_list = [0, 1, 2, 3, 4]
#seed_list = [0, 1, 2, 3, 4, 5 , 6, 7, 8, 9]
num_list = [args.train_samples, args.val_samples]

dataset_index = args.dataset_index
max_epoch = args.max_epoch
learning_rate = args.lr
pyramid_dilation = args.pyramid_dilation
base_save_net_name = 'MambaHSI_{}'.format(FUSION_NAME)


def format_float_for_name(value):
    return '{:g}'.format(value).replace('-', 'm').replace('.', 'p')


if args.outer_residual_mode == 'standard':
    outer_residual_tag = ''
elif args.outer_residual_mode == 'no_outer':
    outer_residual_tag = 'no_outer'
else:
    outer_residual_tag = 'outer_alpha{}'.format(format_float_for_name(args.outer_residual_alpha))

save_net_base = base_save_net_name if outer_residual_tag == '' else '{}_{}'.format(base_save_net_name, outer_residual_tag)
save_net_name = save_net_base if args.ablation == 'full' else '{}_{}'.format(save_net_base, args.ablation)
if args.sdi_kmax > 0:
    save_net_name = '{}_sdiPool{}_{}_gl{:g}'.format(
        save_net_name,
        args.sdi_kmax,
        args.sdi_mode,
        args.sdi_group_lasso
    )
data_set_name_list = ['UP', 'HanChuan', 'HongHu', 'Houston','LongKou','Salinas','indian','Botswana','XuZhou','Pavia']
data_set_name = data_set_name_list[dataset_index]
split_image = data_set_name in ['HanChuan', 'Houston','Pavia']

if args.label_smoothing is None:
    label_smoothing = 0.1 if data_set_name == 'indian' else 0.0
else:
    label_smoothing = args.label_smoothing

if args.class_weight_mode == 'auto':
    class_weight_mode = 'balanced' if data_set_name == 'indian' else 'none'
else:
    class_weight_mode = args.class_weight_mode

paras_dict = {
    'net_name': save_net_name,
    'save_net_name': save_net_name,
    'dataset_index': dataset_index,
    'num_list': num_list,
    'lr': learning_rate,
    'seed_list': seed_list,
    'label_smoothing': label_smoothing,
    'fusion_mode': FUSION_NAME,
    'ablation': args.ablation,
    'class_weight_mode': class_weight_mode,
    'pyramid_dilation': pyramid_dilation,
    'token_num': args.token_num,
    'sdi_kmax': args.sdi_kmax,
    'sdi_mode': args.sdi_mode,
    'sdi_pairs': args.sdi_pairs,
    'sdi_top_m': args.sdi_top_m,
    'sdi_max_pairs': args.sdi_max_pairs,
    'sdi_pair_metric': args.sdi_pair_metric,
    'sdi_slope_source': args.sdi_slope_source,
    'sdi_score_mode': args.sdi_score_mode,
    'sdi_boot_n': args.sdi_boot_n,
    'sdi_boot_ratio': args.sdi_boot_ratio,
    'sdi_boot_seed': args.sdi_boot_seed,
    'sdi_corr_thr': args.sdi_corr_thr,
    'sdi_group_lasso': args.sdi_group_lasso,
    'sdi_prune_thr': args.sdi_prune_thr,
    'outer_residual_mode': args.outer_residual_mode,
    'outer_residual_alpha': args.outer_residual_alpha,
}

transform = transforms.Compose([
    transforms.ToTensor(),
])


def compute_train_loss(net, input_tensor, label_tensor, loss_func):
    pred = net(input_tensor)
    return head_loss(loss_func, pred, label_tensor.long())


def get_fusion_status(model):
    return 'Fusion mode: {}'.format(FUSION_NAME)

if __name__ == '__main__':
    data_set_path = args.data_set_path
    work_dir = args.work_dir
    dataset_name = data_set_name

    save_folder = os.path.join(work_dir, args.exp_name, save_net_name, dataset_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("makedirs {}".format(save_folder))

    save_log_path = os.path.join(save_folder, 'train_tr{}_val{}.log'.format(num_list[0], num_list[1]))

    logger = setup_logger(name='{}'.format(dataset_name), logfile=save_log_path)
    torch.cuda.empty_cache()
    logger.info(save_folder)
    logger.info(get_fusion_status(model=None))

    data, gt = data_load_operate.load_data(data_set_name, data_set_path)

    data_filtered = gaussian_filter(data, sigma=1)

    pca = PCA(n_components=30)
    data_reshaped = data_filtered.reshape(-1, data_filtered.shape[2])
    data_pca = pca.fit_transform(data_reshaped)
    data_pca = data_pca.reshape(data_filtered.shape[0], data_filtered.shape[1], -1)

    height, width, channels = data_pca.shape
    gt_reshape = gt.reshape(-1)
    img = ImageStretching(data_pca)
    class_count = int(max(np.unique(gt)))

    ratio_list = [0.1, 0.01]  # [train_ratio, val_ratio]

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    evaluator = Evaluator(num_class=class_count)

    for exp_idx, curr_seed in enumerate(seed_list):
        setup_seed(curr_seed)

        single_experiment_name = 'run{}_seed{}'.format(str(exp_idx), str(curr_seed))
        save_single_experiment_folder = os.path.join(save_folder, single_experiment_name)
        if not os.path.exists(save_single_experiment_folder):
            os.mkdir(save_single_experiment_folder)
        save_vis_folder = os.path.join(save_single_experiment_folder, 'vis')
        if not os.path.exists(save_vis_folder):
            os.makedirs(save_vis_folder)
            print("makedirs {}".format(save_vis_folder))

        save_weight_path = os.path.join(save_single_experiment_folder, "best_tr{}_val{}.pth".format(num_list[0], num_list[1]))
        results_save_path = os.path.join(save_single_experiment_folder, 'result_tr{}_val{}.txt'.format(num_list[0], num_list[1]))
        predict_save_path = os.path.join(save_single_experiment_folder, 'pred_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))
        gt_save_path = os.path.join(save_single_experiment_folder, 'gt_vis_tr{}_val{}.png'.format(num_list[0], num_list[1]))

        train_data_index, val_data_index, test_data_index, _ = data_load_operate.sampling(
            ratio_list,
            num_list,
            gt_reshape,
            class_count,
            1,
        )
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(height, width, gt_reshape, index)

        x, candidate_sdi_names, selected_hard_pairs = build_input_tensor_with_hpsdi(
            img_pca_stretched=img,
            raw_filtered=data_filtered,
            train_label=train_label,
            class_count=class_count,
            args=args,
            transform=transform
        )
        x = x.to(device)
        model_in_channels = x.shape[1]

        net = MambaHSI(
            in_channels=model_in_channels,
            num_classes=class_count,
            hidden_dim=128,
            token_num=args.token_num,
            pyramid_dilation=pyramid_dilation,
            ablation=args.ablation,
            outer_residual_mode=args.outer_residual_mode,
            outer_residual_alpha=args.outer_residual_alpha,
        )

        if args.sdi_kmax > 0 and model_in_channels > 30:
            zero_init_extra_patch_channels(net, base_channels=30)

        logger.info(paras_dict)
        logger.info('model_in_channels: {}'.format(model_in_channels))
        logger.info('selected_hard_pairs_0based: {}'.format(selected_hard_pairs))
        logger.info('selected_hard_pairs_1based: {}'.format([(a + 1, b + 1) for a, b in selected_hard_pairs]))
        logger.info('candidate_sdi_names: {}'.format(candidate_sdi_names))
        logger.info(net)
        logger.info(get_fusion_status(net))

        if class_weight_mode == 'balanced':
            class_weights, class_counts = compute_balanced_class_weights(train_label, class_count, device)
            loss_func = torch.nn.CrossEntropyLoss(
                ignore_index=-1,
                weight=class_weights,
                label_smoothing=label_smoothing
            )
            logger.info('train_class_counts: {}'.format(class_counts))
            logger.info('class_weights: {}'.format([round(v, 4) for v in class_weights.detach().cpu().tolist()]))
        else:
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)

        train_label = train_label.to(device)
        test_label = test_label.to(device)
        val_label = val_label.to(device)

        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

        logger.info(optimizer)
        if record_computecost:
            net.eval()
            torch.cuda.empty_cache()

            flops, macs1, para = calculate_flops(model=net, input_shape=(1, x.shape[1], x.shape[2], x.shape[3]))

            logger.info("para:{}\n,flops:{}".format(para, flops))

        tic1 = time.perf_counter()
        best_val_acc = 0
        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0)

            net.train()

            if split_image:
                x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                loss_part1 = compute_train_loss(
                    net,
                    x_part1,
                    y_part1,
                    loss_func
                )
                loss_part1 = add_sdi_group_lasso(loss_part1, net, args, base_channels=30)
                optimizer.zero_grad()
                loss_part1.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                loss_part2 = compute_train_loss(
                    net,
                    x_part2,
                    y_part2,
                    loss_func
                )
                loss_part2 = add_sdi_group_lasso(loss_part2, net, args, base_channels=30)
                optimizer.zero_grad()
                loss_part2.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                logger.info(
                    'Iter:{}|cls_loss:{}'.format(
                        epoch,
                        (loss_part1 + loss_part2).detach().cpu().numpy()
                    )
                )


            else:
                try:
                    with autocast(enabled=device.type == 'cuda'):
                        loss = compute_train_loss(
                            net,
                            x,
                            y_train,
                            loss_func
                        )
                    loss = add_sdi_group_lasso(loss, net, args, base_channels=30)
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.empty_cache()
                    logger.info(
                        'Iter:{}|cls_loss:{}'.format(
                            epoch,
                            loss.detach().cpu().numpy()
                        )
                    )

                except RuntimeError:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    split_image = True
                    x_part1 = x[:, :, :x.shape[2] // 2 + 5, :]
                    y_part1 = y_train[:, :x.shape[2] // 2 + 5, :]
                    x_part2 = x[:, :, x.shape[2] // 2 - 5:, :]
                    y_part2 = y_train[:, x.shape[2] // 2 - 5:, :]

                    loss_part1 = compute_train_loss(
                        net,
                        x_part1,
                        y_part1,
                        loss_func
                    )
                    loss_part1 = add_sdi_group_lasso(loss_part1, net, args, base_channels=30)
                    optimizer.zero_grad()
                    loss_part1.backward()
                    optimizer.step()

                    loss_part2 = compute_train_loss(
                        net,
                        x_part2,
                        y_part2,
                        loss_func
                    )
                    loss_part2 = add_sdi_group_lasso(loss_part2, net, args, base_channels=30)
                    optimizer.zero_grad()
                    loss_part2.backward()
                    optimizer.step()

                    logger.info(
                        'Iter:{}|cls_loss:{}'.format(
                            epoch,
                            (loss_part1 + loss_part2).detach().cpu().numpy()
                        )
                    )

            torch.cuda.empty_cache()

            # Evaluation stage
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                output_val = net(x)
                y_val = val_label.unsqueeze(0)
                seg_logits = resize(input=output_val, size=y_val.shape[1:], mode='bilinear', align_corners=True)
                predict = torch.argmax(seg_logits, dim=1).cpu().numpy()
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255, axis=0), predict)
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                logger.info(get_fusion_status(net))
                logger.info('Evaluate {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA, mAcc, Kappa, mIOU, IOU, Acc))

                if OA >= best_val_acc:
                    best_val_acc = OA
                    torch.save(net.state_dict(), save_weight_path)

                if (epoch + 1) % 50 == 0:
                    save_single_predict_path = os.path.join(save_vis_folder, 'predict_{}.png'.format(str(epoch + 1)))
                    save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
                    vis_a_image(gt, predict, save_single_predict_path, save_single_gt_path)

            torch.cuda.empty_cache()
        toc1 = time.perf_counter()  # 记录结束时间
        train_time = toc1 - tic1  # 计算时间间隔
        logger.info(f"train_time: {train_time} seconds")

        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        tic2 = time.perf_counter()

        load_weight_path = save_weight_path
        best_net = MambaHSI(
            in_channels=model_in_channels,
            num_classes=class_count,
            hidden_dim=128,
            token_num=args.token_num,
            pyramid_dilation=pyramid_dilation,
            ablation=args.ablation,
            outer_residual_mode=args.outer_residual_mode,
            outer_residual_alpha=args.outer_residual_alpha,
        )
        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        sdi_importance, effective_sdi_num = get_sdi_channel_importance(
            best_net,
            candidate_sdi_names,
            base_channels=30,
            relative_thr=args.sdi_prune_thr
        )
        logger.info('sdi_channel_importance: {}'.format(sdi_importance))
        logger.info('effective_sdi_num: {}'.format(effective_sdi_num))
        logger.info(get_fusion_status(best_net))

        test_evaluator = Evaluator(num_class=class_count)

        with torch.no_grad():
            test_evaluator.reset()
            output_test = best_net(x)

            y_test = test_label.unsqueeze(0)
            seg_logits_test = resize(input=output_test, size=y_test.shape[1:], mode='bilinear', align_corners=True)
            predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = test_evaluator.Kappa()
            logger.info('Test {}|OA:{}|MACC:{}|Kappa:{}|MIOU:{}|IOU:{}|ACC:{}'.format(epoch, OA_test, mAcc_test, Kappa_test, mIOU_test, IOU_test, Acc_test))
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)
        toc2 = time.perf_counter()  # 记录结束时间
        test_time = toc2 - tic2

        str_results = '\n======================' \
                      + " exp_idx=" + str(exp_idx) \
                      + " seed=" + str(curr_seed) \
                      + " learning rate=" + str(learning_rate) \
                      + " epochs=" + str(max_epoch) \
                      + " train ratio=" + str(ratio_list[0]) \
                      + " val ratio=" + str(ratio_list[1]) \
                      + " ======================" \
                      + "\nOA=" + str(OA_test) \
                      + "\nAA=" + str(mAcc_test) \
                      + '\nkpp=' + str(Kappa_test) \
                      + '\nmIOU_test:' + str(mIOU_test) \
                      + "\nIOU_test:" + str(IOU_test) \
                      + "\nAcc_test:" + str(Acc_test) + "\n"
        logger.info(str_results)
        with open(results_save_path, 'a+') as f:
            f.write(str_results)

        OA_ALL.append(OA_test)
        AA_ALL.append(mAcc_test)
        KPP_ALL.append(Kappa_test)
        EACH_ACC_ALL.append(Acc_test)
        Train_Time_ALL.append(train_time)
        Test_Time_ALL.append(test_time)

        torch.cuda.empty_cache()

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    logger.info("\n====================Mean result of {} times runs =========================".format(len(seed_list)))

    logger.info('List of OA: {}'.format(list(OA_ALL)))
    logger.info('List of AA: {}'.format(list(AA_ALL)))
    logger.info('List of KPP: {}'.format(list(KPP_ALL)))
    logger.info('OA: {:.2f} ± {:.2f}'.format(np.mean(OA_ALL) * 100, np.std(OA_ALL) * 100))
    logger.info('AA: {:.2f} ± {:.2f}'.format(np.mean(AA_ALL) * 100, np.std(AA_ALL) * 100))
    logger.info('Kpp: {:.2f} ± {:.2f}'.format(np.mean(KPP_ALL) * 100, np.std(KPP_ALL) * 100))
    logger.info('Acc per class: {} ± {}'.format(
        np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2).tolist(),
        np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2).tolist()
    ))
    if len(Train_Time_ALL) > 0:
        avg_train_time = np.mean(Train_Time_ALL)
        std_train_time = np.std(Train_Time_ALL)
    else:
        avg_train_time, std_train_time = 0, 0
        logger.warning("Train_Time_ALL 为空，训练时间无法计算，使用默认值 0。")

    if len(Test_Time_ALL) > 0:
        avg_test_time = np.mean(Test_Time_ALL) * 1000
        std_test_time = np.std(Test_Time_ALL) * 1000
    else:
        avg_test_time, std_test_time = 0, 0
        logger.warning("Test_Time_ALL 为空，测试时间无法计算，使用默认值 0。")

    logger.info('Average training time: {:.2f} ± {:.3f}'.format(avg_train_time, std_train_time))
    logger.info('Average testing time: {:.2f} ± {:.3f}'.format(avg_test_time, std_test_time))

    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    with open(mean_result_path, 'w') as f:
        str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + ' times runs ********************' \
                      + '\nList of OA:' + str(list(OA_ALL)) \
                      + '\nList of AA:' + str(list(AA_ALL)) \
                      + '\nList of KPP:' + str(list(KPP_ALL)) \
                      + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                      + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                      + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) \
                      + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
            np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                      + "\nAverage training time=" + str(np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
            np.round(np.std(Train_Time_ALL), decimals=3)) \
                      + "\nAverage testing time=" + str(np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
            np.round(np.std(Test_Time_ALL) * 100, decimals=3))
        f.write(str_results)

    del net

torch.cuda.empty_cache()
