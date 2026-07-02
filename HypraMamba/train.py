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
from model.MambaHSI import (
    ImprovedMambaHSI as MambaHSI,
    VALID_ABLATIONS,
    VALID_HIGH_RES_SKIP_MODES,
    VALID_OUTER_RESIDUAL_MODES,
)
try:
    from calflops import calculate_flops
except ImportError:
    calculate_flops = None
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.6'
scaler = GradScaler(enabled=torch.cuda.is_available())
FUSION_NAME = 'competitive'
QUH_DATASETS = {'QUH-Pingan', 'QUH-Qingyun', 'QUH-Tangdaowan'}


def vis_a_image(gt_vis, pred_vis, save_single_predict_path, save_single_gt_path, only_vis_label=False):
    from utils.visual_predict import visualize_predict

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


def parse_class_weight_multipliers(value):
    if value is None:
        return {}
    value = value.strip()
    if value == '' or value.lower() in ('none', 'off', 'false'):
        return {}

    multipliers = {}
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise argparse.ArgumentTypeError(
                'Expected class multipliers like "6:1.15,16:1.20".'
            )
        class_id_text, multiplier_text = item.split(':', 1)
        try:
            class_id = int(class_id_text.strip())
            multiplier = float(multiplier_text.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                'Expected integer class id and float multiplier in "{}".'.format(item)
            )
        if class_id <= 0:
            raise argparse.ArgumentTypeError('Class ids are 1-based and must be positive.')
        if multiplier <= 0:
            raise argparse.ArgumentTypeError('Class weight multipliers must be positive.')
        multipliers[class_id - 1] = multiplier
    return multipliers


def format_class_weight_multipliers(multipliers):
    return {
        class_idx + 1: multiplier
        for class_idx, multiplier in sorted(multipliers.items())
    }


def build_loss_weights(train_label, class_count, class_weight_mode, class_weight_multipliers, target_device):
    valid_labels = train_label[train_label >= 0].long().view(-1)
    class_counts = torch.bincount(valid_labels, minlength=class_count).float()

    if class_weight_mode == 'balanced':
        class_weights = torch.zeros(class_count, dtype=torch.float32)
        nonzero_mask = class_counts > 0
        if nonzero_mask.any():
            total_valid = class_counts[nonzero_mask].sum()
            class_weights[nonzero_mask] = total_valid / (nonzero_mask.sum() * class_counts[nonzero_mask])
    else:
        class_weights = torch.ones(class_count, dtype=torch.float32)

    for class_idx, multiplier in class_weight_multipliers.items():
        if class_idx < 0 or class_idx >= class_count:
            raise ValueError(
                '--class_weight_multipliers contains class {}, but {} has only {} classes.'.format(
                    class_idx + 1,
                    data_set_name,
                    class_count
                )
            )
        class_weights[class_idx] *= multiplier

    if class_weight_mode == 'none' and len(class_weight_multipliers) == 0:
        return None, class_counts.long().tolist()
    return class_weights.to(target_device), class_counts.long().tolist()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_int_list(value):
    if isinstance(value, list):
        return value
    try:
        values = [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError('Expected a comma-separated integer list.')
    if len(values) == 0:
        raise argparse.ArgumentTypeError('Expected at least one integer.')
    return values


def validate_args(args, parser):
    positive_int_fields = [
        'pca_components',
        'hidden_dim',
        'token_num',
        'group_num',
        'pool_size',
        'cls_head_dim',
        'prca_num_scales',
        'prca_num_layers',
        'prca_num_heads',
        'lsp_reduction',
        'spa_mamba_d_state',
        'spa_mamba_d_conv',
        'spa_mamba_expand',
        'spe_mamba_d_state',
        'spe_mamba_d_conv',
        'spe_mamba_expand',
    ]
    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            parser.error('--{} must be a positive integer.'.format(field))

    if args.weight_decay < 0:
        parser.error('--weight_decay must be non-negative.')
    if args.cosine_eta_min < 0:
        parser.error('--cosine_eta_min must be non-negative.')
    if args.gaussian_sigma < 0:
        parser.error('--gaussian_sigma must be non-negative.')
    if args.stretch_low < 0 or args.stretch_high > 100:
        parser.error('--stretch_low and --stretch_high must be within [0, 100].')
    if args.hidden_dim % args.group_num != 0:
        parser.error('--hidden_dim must be divisible by --group_num.')
    if args.cls_head_dim % args.group_num != 0:
        parser.error('--cls_head_dim must be divisible by --group_num.')
    if args.hidden_dim % args.token_num != 0:
        parser.error('--hidden_dim must be divisible by --token_num.')
    if args.hidden_dim % args.prca_num_heads != 0:
        parser.error('--hidden_dim must be divisible by --prca_num_heads.')
    if args.stretch_high <= args.stretch_low:
        parser.error('--stretch_high must be greater than --stretch_low.')
    if args.tile_size < 0:
        parser.error('--tile_size must be non-negative.')
    if args.tile_overlap < 0:
        parser.error('--tile_overlap must be non-negative.')
    if args.tile_size > 0 and args.tile_overlap >= args.tile_size:
        parser.error('--tile_overlap must be smaller than --tile_size.')
    if args.tile_update_groups <= 0:
        parser.error('--tile_update_groups must be a positive integer.')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_index', type=int, default=8)
    parser.add_argument('--data_set_path', type=str, default='./data')
    parser.add_argument('--work_dir', type=str, default='./')

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='none', choices=['none', 'cosine'])
    parser.add_argument('--cosine_eta_min', type=float, default=0.0)
    parser.add_argument('--train_samples', type=int, default=100)
    parser.add_argument('--val_samples', type=int, default=30)
    parser.add_argument('--split_dir', type=str, default='./splits/quh_100_30_seed0-9')
    parser.add_argument('--seed_list', type=parse_int_list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--exp_name', type=str, default='RUNS')
    parser.add_argument('--record_computecost', type=str2bool, default=False)
    parser.add_argument('--save_vis', type=str2bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--class_weight_mode', type=str, default='balanced', choices=['auto', 'none', 'balanced'])
    parser.add_argument(
        '--class_weight_multipliers',
        type=parse_class_weight_multipliers,
        default={},
        help='Optional 1-based class loss multipliers, e.g. "6:1.15,16:1.20".'
    )
    parser.add_argument(
        '--checkpoint_metric',
        type=str,
        default='oa',
        choices=['oa', 'aa', 'miou', 'kappa'],
        help='Validation metric used to select the best checkpoint.'
    )
    parser.add_argument('--pca_components', type=int, default=30)
    parser.add_argument('--gaussian_sigma', type=float, default=1.0)
    parser.add_argument('--stretch_low', type=float, default=2.0)
    parser.add_argument('--stretch_high', type=float, default=98.0)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--token_num', type=int, default=4)
    parser.add_argument('--group_num', type=int, default=4)
    parser.add_argument('--use_residual', type=str2bool, default=True)
    parser.add_argument('--pool_size', type=int, default=2)
    parser.add_argument('--high_res_skip', type=str, default='none', choices=sorted(VALID_HIGH_RES_SKIP_MODES))
    parser.add_argument('--cls_head_dim', type=int, default=128)
    parser.add_argument('--prca_num_scales', type=int, default=3)
    parser.add_argument('--prca_num_layers', type=int, default=2)
    parser.add_argument('--prca_num_heads', type=int, default=4)
    parser.add_argument('--lsp_reduction', type=int, default=4)
    parser.add_argument('--spa_mamba_d_state', type=int, default=16)
    parser.add_argument('--spa_mamba_d_conv', type=int, default=4)
    parser.add_argument('--spa_mamba_expand', type=int, default=2)
    parser.add_argument('--spe_mamba_d_state', type=int, default=16)
    parser.add_argument('--spe_mamba_d_conv', type=int, default=4)
    parser.add_argument('--spe_mamba_expand', type=int, default=2)
    parser.add_argument('--pyramid_dilation', type=str, default='3')
    parser.add_argument('--ablation', type=str, default='full', choices=sorted(VALID_ABLATIONS))
    parser.add_argument('--outer_residual_mode', type=str, default='standard', choices=sorted(VALID_OUTER_RESIDUAL_MODES))
    parser.add_argument('--outer_residual_alpha', type=float, default=1.0)
    parser.add_argument('--spectral_diff_alpha', type=float, default=0.5)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--tile_update_groups', type=int, default=2)

    args = parser.parse_args()
    validate_args(args, parser)
    return args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args \
    = get_parser()
record_computecost = args.record_computecost
seed_list = args.seed_list
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
if args.spectral_diff_alpha != 1.0:
    save_net_base = '{}_diff_alpha{}'.format(save_net_base, format_float_for_name(args.spectral_diff_alpha))
save_net_name = save_net_base if args.ablation == 'full' else '{}_{}'.format(save_net_base, args.ablation)
data_set_name_list = [
    'UP',
    'HanChuan',
    'HongHu',
    'Houston',
    'LongKou',
    'Salinas',
    'indian',
    'Botswana',
    'XuZhou',
    'Pavia',
    'QUH-Pingan',
    'QUH-Qingyun',
    'QUH-Tangdaowan',
]
data_set_name = data_set_name_list[dataset_index]
split_image = data_set_name in ['HanChuan', 'Houston','Pavia']
tile_size = args.tile_size
if data_set_name in QUH_DATASETS and tile_size <= 0:
    tile_size = 256
tile_overlap = args.tile_overlap if tile_size > 0 else 0
use_tile_mode = tile_size > 0

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
    'weight_decay': args.weight_decay,
    'optimizer': args.optimizer,
    'scheduler': args.scheduler,
    'cosine_eta_min': args.cosine_eta_min,
    'split_dir': args.split_dir,
    'seed_list': seed_list,
    'label_smoothing': label_smoothing,
    'fusion_mode': FUSION_NAME,
    'ablation': args.ablation,
    'class_weight_mode': class_weight_mode,
    'class_weight_multipliers': format_class_weight_multipliers(args.class_weight_multipliers),
    'checkpoint_metric': args.checkpoint_metric,
    'pca_components': args.pca_components,
    'gaussian_sigma': args.gaussian_sigma,
    'stretch_low': args.stretch_low,
    'stretch_high': args.stretch_high,
    'hidden_dim': args.hidden_dim,
    'token_num': args.token_num,
    'group_num': args.group_num,
    'use_residual': args.use_residual,
    'pool_size': args.pool_size,
    'high_res_skip': args.high_res_skip,
    'cls_head_dim': args.cls_head_dim,
    'prca_num_scales': args.prca_num_scales,
    'prca_num_layers': args.prca_num_layers,
    'prca_num_heads': args.prca_num_heads,
    'lsp_reduction': args.lsp_reduction,
    'spa_mamba_d_state': args.spa_mamba_d_state,
    'spa_mamba_d_conv': args.spa_mamba_d_conv,
    'spa_mamba_expand': args.spa_mamba_expand,
    'spe_mamba_d_state': args.spe_mamba_d_state,
    'spe_mamba_d_conv': args.spe_mamba_d_conv,
    'spe_mamba_expand': args.spe_mamba_expand,
    'pyramid_dilation': pyramid_dilation,
    'outer_residual_mode': args.outer_residual_mode,
    'outer_residual_alpha': args.outer_residual_alpha,
    'spectral_diff_alpha': args.spectral_diff_alpha,
    'tile_size': tile_size,
    'tile_overlap': tile_overlap,
    'tile_update_groups': args.tile_update_groups,
    'save_vis': args.save_vis,
}

model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'token_num': args.token_num,
    'group_num': args.group_num,
    'use_residual': args.use_residual,
    'pyramid_dilation': pyramid_dilation,
    'ablation': args.ablation,
    'outer_residual_mode': args.outer_residual_mode,
    'outer_residual_alpha': args.outer_residual_alpha,
    'spectral_diff_alpha': args.spectral_diff_alpha,
    'pool_size': args.pool_size,
    'high_res_skip': args.high_res_skip,
    'cls_head_dim': args.cls_head_dim,
    'prca_num_scales': args.prca_num_scales,
    'prca_num_layers': args.prca_num_layers,
    'prca_num_heads': args.prca_num_heads,
    'lsp_reduction': args.lsp_reduction,
    'spa_mamba_d_state': args.spa_mamba_d_state,
    'spa_mamba_d_conv': args.spa_mamba_d_conv,
    'spa_mamba_expand': args.spa_mamba_expand,
    'spe_mamba_d_state': args.spe_mamba_d_state,
    'spe_mamba_d_conv': args.spe_mamba_d_conv,
    'spe_mamba_expand': args.spe_mamba_expand,
}

transform = transforms.Compose([
    transforms.ToTensor(),
])


def build_optimizer(net, optimizer_name, lr, weight_decay):
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError('Unsupported optimizer: {}'.format(optimizer_name))


def build_scheduler(optimizer, scheduler_name, max_epochs, eta_min):
    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=eta_min
        )
    if scheduler_name == 'none':
        return None
    raise ValueError('Unsupported scheduler: {}'.format(scheduler_name))


def select_checkpoint_score(metric_name, oa, aa, miou, kappa):
    if metric_name == 'oa':
        return oa
    if metric_name == 'aa':
        return aa
    if metric_name == 'miou':
        return miou
    if metric_name == 'kappa':
        return kappa
    raise ValueError('Unsupported checkpoint metric: {}'.format(metric_name))


def compute_train_loss(net, input_tensor, label_tensor, loss_func):
    pred = net(input_tensor)
    return head_loss(loss_func, pred, label_tensor.long())


def _tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def generate_tile_slices(height, width, tile_size, overlap):
    y_starts = _tile_starts(height, tile_size, overlap)
    x_starts = _tile_starts(width, tile_size, overlap)
    return [
        (y0, min(y0 + tile_size, height), x0, min(x0 + tile_size, width))
        for y0 in y_starts
        for x0 in x_starts
    ]


def count_labeled_pixels_in_tile(label_tensor, tile_slice):
    y0, y1, x0, x1 = tile_slice
    return int((label_tensor[y0:y1, x0:x1] >= 0).sum().item())


def _build_balanced_tile_groups(train_tiles, group_count):
    group_count = max(1, min(int(group_count), len(train_tiles)))
    groups = [[] for _ in range(group_count)]
    group_pixels = [0 for _ in range(group_count)]

    # Randomize equal-size/equal-density cases, then greedily balance labeled-pixel counts.
    shuffled_tiles = list(train_tiles)
    random.shuffle(shuffled_tiles)
    shuffled_tiles.sort(key=lambda item: item[1], reverse=True)
    for tile_slice, valid_pixels in shuffled_tiles:
        group_idx = min(range(group_count), key=lambda idx: group_pixels[idx])
        groups[group_idx].append((tile_slice, valid_pixels))
        group_pixels[group_idx] += valid_pixels

    return [group for group in groups if len(group) > 0]


def train_one_epoch_tiled(
        net,
        x_cpu,
        train_label_cpu,
        tile_slices,
        loss_func,
        optimizer,
        tile_update_groups):
    train_tiles = []
    for tile_slice in tile_slices:
        valid_pixels = count_labeled_pixels_in_tile(train_label_cpu, tile_slice)
        if valid_pixels > 0:
            train_tiles.append((tile_slice, valid_pixels))

    if len(train_tiles) == 0:
        raise RuntimeError('No training tiles contain labeled samples.')

    tile_groups = _build_balanced_tile_groups(train_tiles, tile_update_groups)
    y_train = train_label_cpu.unsqueeze(0)
    total_loss = 0.0
    total_valid_pixels = sum(valid_pixels for _, valid_pixels in train_tiles)
    used_tiles = 0
    used_groups = 0

    for tile_group in tile_groups:
        group_valid_pixels = sum(valid_pixels for _, valid_pixels in tile_group)
        if group_valid_pixels <= 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        for tile_slice, valid_pixels in tile_group:
            y0, y1, x0, x1 = tile_slice
            input_tile = x_cpu[:, :, y0:y1, x0:x1].to(device)
            label_tile = y_train[:, y0:y1, x0:x1].to(device)

            with autocast(enabled=device.type == 'cuda'):
                loss = compute_train_loss(
                    net,
                    input_tile,
                    label_tile,
                    loss_func
                )
                group_loss_weight = valid_pixels / group_valid_pixels
                weighted_loss = loss * group_loss_weight

            scaler.scale(weighted_loss).backward()

            total_loss += float(loss.detach().cpu()) * (valid_pixels / total_valid_pixels)
            used_tiles += 1

            del input_tile, label_tile, loss, weighted_loss

        scaler.step(optimizer)
        scaler.update()
        used_groups += 1

    return total_loss, used_tiles, used_groups


def predict_tiled(net, x_cpu, tile_slices, class_count, output_size):
    height, width = output_size
    logit_sum = np.zeros((class_count, height, width), dtype=np.float32)
    logit_count = np.zeros((height, width), dtype=np.float32)

    for tile_slice in tile_slices:
        y0, y1, x0, x1 = tile_slice
        input_tile = x_cpu[:, :, y0:y1, x0:x1].to(device)

        with autocast(enabled=device.type == 'cuda'):
            output_tile = net(input_tile)
            seg_logits_tile = resize(
                input=output_tile,
                size=(y1 - y0, x1 - x0),
                mode='bilinear',
                align_corners=True
            )

        logit_sum[:, y0:y1, x0:x1] += seg_logits_tile.squeeze(0).float().cpu().numpy()
        logit_count[y0:y1, x0:x1] += 1.0

        del input_tile, output_tile, seg_logits_tile

    logit_sum /= np.maximum(logit_count[None, :, :], 1.0)
    return np.expand_dims(np.argmax(logit_sum, axis=0), axis=0)


def count_parameters(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


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

    if args.pca_components > data.shape[2]:
        raise ValueError('--pca_components must be <= input channel count {}.'.format(data.shape[2]))

    data_filtered = gaussian_filter(data, sigma=args.gaussian_sigma)

    logger.info('PCA enabled: pca_components={}'.format(args.pca_components))
    pca = PCA(n_components=args.pca_components)
    data_reshaped = data_filtered.reshape(-1, data_filtered.shape[2])
    data_pca = pca.fit_transform(data_reshaped)
    data_pca = data_pca.reshape(data_filtered.shape[0], data_filtered.shape[1], -1)

    height, width, channels = data_pca.shape
    gt_reshape = gt.reshape(-1)
    img = ImageStretching(data_pca, low=args.stretch_low, high=args.stretch_high)
    class_count = int(max(np.unique(gt)))

    ratio_list = [0.1, 0.01]  # [train_ratio, val_ratio]

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    total_params_m = None
    trainable_params_m = None
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

        if data_set_name in QUH_DATASETS:
            train_data_index, val_data_index, test_data_index, split_path = data_load_operate.load_fixed_split(
                args.split_dir,
                data_set_name,
                curr_seed,
                num_list[0],
                num_list[1],
                gt_reshape=gt_reshape,
            )
            logger.info(
                'Loaded fixed split: {} train={} val={} test={}'.format(
                    split_path,
                    len(train_data_index),
                    len(val_data_index),
                    len(test_data_index)
                )
            )
        else:
            train_data_index, val_data_index, test_data_index, _ = data_load_operate.sampling(
                ratio_list,
                num_list,
                gt_reshape,
                class_count,
                1,
            )
        index = (train_data_index, val_data_index, test_data_index)
        train_label, val_label, test_label = data_load_operate.generate_image_iter(height, width, gt_reshape, index)

        net = MambaHSI(
            in_channels=channels,
            num_classes=class_count,
            **model_kwargs,
        )

        logger.info(paras_dict)
        logger.info(net)
        logger.info(get_fusion_status(net))

        x = transform(np.asarray(img, dtype=np.float32))
        x = x.unsqueeze(0).float()
        if use_tile_mode:
            tile_slices = generate_tile_slices(height, width, tile_size, tile_overlap)
            train_tile_slices = tile_slices
            logger.info(
                'Tile mode enabled: tile_size={} tile_overlap={} tile_count={}'.format(
                    tile_size,
                    tile_overlap,
                    len(tile_slices)
                )
            )
        else:
            tile_slices = None
            train_tile_slices = None
            x = x.to(device)

        class_weights, class_counts = build_loss_weights(
            train_label,
            class_count,
            class_weight_mode,
            args.class_weight_multipliers,
            device
        )
        loss_func = torch.nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        logger.info('train_class_counts: {}'.format(class_counts))
        if class_weights is None:
            logger.info('class_weights: None')
        else:
            logger.info('class_weights: {}'.format([round(v, 4) for v in class_weights.detach().cpu().tolist()]))
        if len(args.class_weight_multipliers) > 0:
            logger.info(
                'class_weight_multipliers(1-based): {}'.format(
                    format_class_weight_multipliers(args.class_weight_multipliers)
                )
            )

        if not use_tile_mode:
            train_label = train_label.to(device)
            test_label = test_label.to(device)
            val_label = val_label.to(device)

        net.to(device)

        optimizer = build_optimizer(net, args.optimizer, learning_rate, args.weight_decay)
        scheduler = build_scheduler(optimizer, args.scheduler, max_epoch, args.cosine_eta_min)

        logger.info(optimizer)
        logger.info('scheduler: {}'.format(scheduler))
        total_params, trainable_params = count_parameters(net)
        total_params_m = total_params / 1e6
        trainable_params_m = trainable_params / 1e6
        logger.info('Paras(M): {:.6f}'.format(total_params_m))
        logger.info('Trainable Paras(M): {:.6f}'.format(trainable_params_m))
        if record_computecost:
            net.eval()
            torch.cuda.empty_cache()

            if calculate_flops is None:
                logger.warning('calflops is not installed; FLOPs(G) is unavailable.')
            else:
                if use_tile_mode:
                    flops_shape = (1, x.shape[1], min(tile_size, x.shape[2]), min(tile_size, x.shape[3]))
                else:
                    flops_shape = (1, x.shape[1], x.shape[2], x.shape[3])
                flops, macs1, para = calculate_flops(model=net, input_shape=flops_shape)
                logger.info('calflops para: {}'.format(para))
                logger.info('calflops flops: {}'.format(flops))
                logger.info('FLOPs are tool estimates; verify Mamba custom ops support before reporting them.')

        tic1 = time.perf_counter()
        best_val_score = -float('inf')
        for epoch in range(max_epoch):
            y_train = train_label.unsqueeze(0)

            net.train()

            if use_tile_mode:
                avg_loss, used_tiles, used_groups = train_one_epoch_tiled(
                    net,
                    x,
                    train_label,
                    train_tile_slices,
                    loss_func,
                    optimizer,
                    args.tile_update_groups
                )
                logger.info(
                    'Iter:{}|cls_loss:{}|tiles:{}|tile_update_groups:{}'.format(
                        epoch,
                        avg_loss,
                        used_tiles,
                        used_groups
                    )
                )

            elif split_image:
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
                    optimizer.zero_grad()
                    loss_part1.backward()
                    optimizer.step()

                    loss_part2 = compute_train_loss(
                        net,
                        x_part2,
                        y_part2,
                        loss_func
                    )
                    optimizer.zero_grad()
                    loss_part2.backward()
                    optimizer.step()

                    logger.info(
                        'Iter:{}|cls_loss:{}'.format(
                            epoch,
                            (loss_part1 + loss_part2).detach().cpu().numpy()
                        )
                    )

            # Evaluation stage
            need_val_vis = args.save_vis and (epoch + 1) % 50 == 0
            net.eval()
            with torch.no_grad():
                evaluator.reset()
                y_val = val_label.unsqueeze(0)
                if use_tile_mode:
                    predict = predict_tiled(net, x, tile_slices, class_count, y_val.shape[1:])
                else:
                    output_val = net(x)
                    seg_logits = resize(input=output_val, size=y_val.shape[1:], mode='bilinear',
                                        align_corners=True)
                    predict = torch.argmax(seg_logits, dim=1).cpu().numpy()
                Y_val_np = val_label.cpu().numpy()
                Y_val_255 = np.where(Y_val_np == -1, 255, Y_val_np)
                evaluator.add_batch(np.expand_dims(Y_val_255, axis=0), predict)
                OA = evaluator.Pixel_Accuracy()
                mIOU, IOU = evaluator.Mean_Intersection_over_Union()
                mAcc, Acc = evaluator.Pixel_Accuracy_Class()
                Kappa = evaluator.Kappa()
                checkpoint_score = select_checkpoint_score(args.checkpoint_metric, OA, mAcc, mIOU, Kappa)

                if need_val_vis:
                    save_single_predict_path = os.path.join(
                        save_vis_folder,
                        'predict_{}.png'.format(str(epoch + 1))
                    )
                    save_single_gt_path = os.path.join(save_vis_folder, 'gt.png')
                    vis_a_image(gt, predict, save_single_predict_path, save_single_gt_path)

                logger.info(get_fusion_status(net))
                logger.info(
                    'Evaluate {}|OA:{}|AA:{}|mIOU:{}|Kappa:{}|checkpoint_metric:{}|score:{}'.format(
                        epoch,
                        OA,
                        mAcc,
                        mIOU,
                        Kappa,
                        args.checkpoint_metric,
                        checkpoint_score
                    )
                )

                if checkpoint_score >= best_val_score:
                    best_val_score = checkpoint_score
                    torch.save(net.state_dict(), save_weight_path)

            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
        toc1 = time.perf_counter()  # 记录结束时间
        train_time = toc1 - tic1  # 计算时间间隔
        logger.info(f"train_time: {train_time} seconds")

        logger.info("\n\n====================Starting evaluation for testing set.========================\n")
        tic2 = time.perf_counter()

        load_weight_path = save_weight_path
        best_net = MambaHSI(
            in_channels=channels,
            num_classes=class_count,
            **model_kwargs,
        )
        best_net.to(device)
        best_net.load_state_dict(torch.load(load_weight_path))
        best_net.eval()
        logger.info(get_fusion_status(best_net))

        test_evaluator = Evaluator(num_class=class_count)

        with torch.no_grad():
            test_evaluator.reset()
            y_test = test_label.unsqueeze(0)
            if use_tile_mode:
                predict_test = predict_tiled(best_net, x, tile_slices, class_count, y_test.shape[1:])
            else:
                output_test = best_net(x)
                seg_logits_test = resize(input=output_test, size=y_test.shape[1:], mode='bilinear', align_corners=True)
                predict_test = torch.argmax(seg_logits_test, dim=1).cpu().numpy()
            Y_test_np = test_label.cpu().numpy()
            Y_test_255 = np.where(Y_test_np == -1, 255, Y_test_np)
            test_evaluator.add_batch(np.expand_dims(Y_test_255, axis=0), predict_test)
            OA_test = test_evaluator.Pixel_Accuracy()
            mIOU_test, IOU_test = test_evaluator.Mean_Intersection_over_Union()
            mAcc_test, Acc_test = test_evaluator.Pixel_Accuracy_Class()
            Kappa_test = test_evaluator.Kappa()
            logger.info('Test {}|OA:{}|AA:{}|Kappa:{}'.format(epoch, OA_test, mAcc_test, Kappa_test))
        toc2 = time.perf_counter()  # 记录结束时间
        test_time = toc2 - tic2
        if args.save_vis:
            vis_a_image(gt, predict_test, predict_save_path, gt_save_path)

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
                      + "\nAcc_test:" + str(Acc_test) \
                      + "\nTrain time(s)=" + str(train_time) \
                      + "\nTest time(s)=" + str(test_time) + "\n"
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
        avg_test_time = np.mean(Test_Time_ALL)
        std_test_time = np.std(Test_Time_ALL)
    else:
        avg_test_time, std_test_time = 0, 0
        logger.warning("Test_Time_ALL 为空，测试时间无法计算，使用默认值 0。")

    logger.info('Paras(M): {:.6f}'.format(total_params_m))
    logger.info('Trainable Paras(M): {:.6f}'.format(trainable_params_m))
    logger.info('Average training time(s): {:.2f} ± {:.3f}'.format(avg_train_time, std_train_time))
    logger.info('Average testing time(s): {:.2f} ± {:.3f}'.format(avg_test_time, std_test_time))

    mean_result_path = os.path.join(save_folder, 'mean_result.txt')
    with open(mean_result_path, 'w') as f:
        str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + ' times runs ********************' \
                      + '\nParas(M)=' + str(round(total_params_m, 6)) \
                      + '\nTrainable Paras(M)=' + str(round(trainable_params_m, 6)) \
                      + '\nList of OA:' + str(list(OA_ALL)) \
                      + '\nList of AA:' + str(list(AA_ALL)) \
                      + '\nList of KPP:' + str(list(KPP_ALL)) \
                      + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                      + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                      + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) \
                      + '\nAcc per class=\n' + str(np.round(np.mean(EACH_ACC_ALL, 0) * 100, 2)) + '+-' + str(
            np.round(np.std(EACH_ACC_ALL, 0) * 100, 2)) \
                      + "\nAverage training time(s)=" + str(np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
            np.round(np.std(Train_Time_ALL), decimals=3)) \
                      + "\nAverage testing time(s)=" + str(np.round(np.mean(Test_Time_ALL), decimals=2)) + '+-' + str(
            np.round(np.std(Test_Time_ALL), decimals=3))
        f.write(str_results)

    del net

torch.cuda.empty_cache()
