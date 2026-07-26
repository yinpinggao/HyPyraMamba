#!/usr/bin/env python3
"""Build a Kaggle submission for tree-species-hsi-2026 with a trained PyS2CF checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from torch.cuda.amp import autocast
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils.data_load_operate as data_load_operate
from model.MambaHSI import ImprovedMambaHSI as MambaHSI
from utils.Loss import resize


def get_parser():
    parser = argparse.ArgumentParser(description='TreeSpeciesHSI competition submission')
    parser.add_argument('--data_set_path', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_*.pth')
    parser.add_argument(
        '--preprocess',
        type=str,
        default='',
        help='Optional preprocess_*.npz from training. If empty, refit PCA on train cube.',
    )
    parser.add_argument('--model_kwargs_json', type=str, default='')
    parser.add_argument('--output_csv', type=str, default='./submissions/tree_species_hsi_submission.csv')
    parser.add_argument(
        '--layout',
        type=str,
        default='native_chw',
        choices=['native_chw', 'scene_info_hw'],
        help='Test-cube spatial layout. native_chw matches train (C,d1,d2)->(d1,d2,C).',
    )
    parser.add_argument(
        '--flatten_mode',
        type=str,
        default='meta',
        choices=['meta', 'no_transpose_C', 'transpose_C', 'transpose_F', 'no_transpose_F'],
        help=(
            'How to flatten HxW predictions into submission ids after rewriting to '
            'scene_info (height, width). '
            "'meta' uses scene_info transpose/flatten_order; "
            'other modes override that for ablation.'
        ),
    )
    parser.add_argument(
        '--pred_cache_dir',
        type=str,
        default='',
        help='If set, save/load per-scene HxW prediction maps (.npy) to avoid re-inference.',
    )
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=32)
    parser.add_argument('--pca_components', type=int, default=30)
    parser.add_argument('--gaussian_sigma', type=float, default=1.0)
    parser.add_argument('--gaussian_spectral_sigma', type=float, default=None)
    parser.add_argument('--stretch_low', type=float, default=2.0)
    parser.add_argument('--stretch_high', type=float, default=98.0)
    parser.add_argument('--class_count', type=int, default=17)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


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


def apply_stretch(data_pca, low, high, stretch_mins=None, stretch_maxs=None):
    bands = []
    for i in range(data_pca.shape[2]):
        band = data_pca[:, :, i]
        if stretch_mins is None or stretch_maxs is None:
            bmin = np.percentile(band, low)
            bmax = np.percentile(band, high)
        else:
            bmin = float(stretch_mins[i])
            bmax = float(stretch_maxs[i])
        denom = bmax - bmin if bmax > bmin else 1.0
        band = (band - bmin) / denom
        bands.append(band)
    image = np.clip(np.stack(bands, axis=-1), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def preprocess_cube(data, pca, gaussian_sigmas, stretch_low, stretch_high, stretch_mins=None, stretch_maxs=None):
    data = np.asarray(data, dtype=np.float32)
    filtered = gaussian_filter(data, sigma=gaussian_sigmas)
    flat = filtered.reshape(-1, filtered.shape[2])
    projected = pca.transform(flat).reshape(filtered.shape[0], filtered.shape[1], -1)
    # Per-scene stretch is more robust for test domains than freezing train percentiles.
    img = apply_stretch(projected, stretch_low, stretch_high, stretch_mins=None, stretch_maxs=None)
    tensor = transforms.ToTensor()(np.asarray(img, dtype=np.float32)).unsqueeze(0).float()
    return tensor


def predict_tiled(net, x_cpu, tile_slices, class_count, device):
    _, _, height, width = x_cpu.shape
    logit_sum = np.zeros((class_count, height, width), dtype=np.float32)
    logit_count = np.zeros((height, width), dtype=np.float32)

    for y0, y1, x0, x1 in tile_slices:
        input_tile = x_cpu[:, :, y0:y1, x0:x1].to(device)
        with torch.no_grad():
            with autocast(enabled=device.type == 'cuda'):
                output_tile = net(input_tile)
                seg_logits_tile = resize(
                    input=output_tile,
                    size=(y1 - y0, x1 - x0),
                    mode='bilinear',
                    align_corners=True,
                )
        logit_sum[:, y0:y1, x0:x1] += seg_logits_tile.squeeze(0).float().cpu().numpy()
        logit_count[y0:y1, x0:x1] += 1.0
        del input_tile, output_tile, seg_logits_tile

    logit_sum /= np.maximum(logit_count[None, :, :], 1.0)
    # Model outputs 0..C-1; competition labels are 1..C.
    return np.argmax(logit_sum, axis=0).astype(np.int32) + 1


def flatten_prediction(pred_hw, transpose, flatten_order):
    arr = pred_hw.T if transpose else pred_hw
    return arr.reshape(-1, order=flatten_order)


def resolve_flatten_params(meta, flatten_mode):
    if flatten_mode == 'meta':
        return bool(meta['transpose']), str(meta['flatten_order'])
    if flatten_mode == 'no_transpose_C':
        return False, 'C'
    if flatten_mode == 'no_transpose_F':
        return False, 'F'
    if flatten_mode == 'transpose_C':
        return True, 'C'
    if flatten_mode == 'transpose_F':
        return True, 'F'
    raise ValueError('Unsupported flatten_mode: {}'.format(flatten_mode))


def load_or_fit_pca(args):
    gaussian_spectral_sigma = (
        args.gaussian_sigma if args.gaussian_spectral_sigma is None else args.gaussian_spectral_sigma
    )
    gaussian_sigmas = (args.gaussian_sigma, args.gaussian_sigma, gaussian_spectral_sigma)

    if args.preprocess and os.path.exists(args.preprocess):
        payload = np.load(args.preprocess)
        n_components = int(payload['pca_components_n'])
        components = np.asarray(payload['pca_components'], dtype=np.float64)
        mean = np.asarray(payload['pca_mean'], dtype=np.float64)
        pca = PCA(n_components=n_components)
        pca.components_ = components
        pca.mean_ = mean
        # Reconstruct the minimal fitted state sklearn's transform() expects.
        pca.n_features_in_ = components.shape[1]
        pca.n_components_ = components.shape[0]
        if 'explained_variance_' in payload.files:
            pca.explained_variance_ = np.asarray(payload['explained_variance_'], dtype=np.float64)
        else:
            # Transform only needs components_/mean_; provide a dummy variance vector.
            pca.explained_variance_ = np.ones(components.shape[0], dtype=np.float64)
        if 'explained_variance_ratio_' in payload.files:
            pca.explained_variance_ratio_ = np.asarray(
                payload['explained_variance_ratio_'], dtype=np.float64
            )
        else:
            pca.explained_variance_ratio_ = pca.explained_variance_ / max(
                float(pca.explained_variance_.sum()), 1.0
            )
        pca.singular_values_ = np.sqrt(np.maximum(pca.explained_variance_, 0.0))
        class_count = int(payload['class_count']) if 'class_count' in payload else args.class_count
        print('Loaded PCA from', args.preprocess)
        return pca, gaussian_sigmas, class_count

    print('Fitting PCA on train cube ...')
    data, _, _, _, _ = data_load_operate.load_tree_species_train_cube(args.data_set_path)
    filtered = gaussian_filter(np.asarray(data, dtype=np.float32), sigma=gaussian_sigmas)
    pca = PCA(n_components=args.pca_components)
    pca.fit(filtered.reshape(-1, filtered.shape[2]))
    return pca, gaussian_sigmas, args.class_count


def load_model(args, class_count, in_channels):
    model_kwargs = {
        'hidden_dim': 128,
        'token_num': 4,
        'group_num': 4,
        'use_residual': True,
        'pyramid_dilation': '3',
        'ablation': 'full',
        'outer_residual_mode': 'standard',
        'outer_residual_alpha': 1.0,
        'spectral_diff_alpha': 0.5,
        'spectral_fusion_scale': 1.0,
        'pool_size': 2,
        'high_res_skip': 'none',
        'cls_head_dim': 128,
        'prca_num_scales': 3,
        'prca_num_layers': 2,
        'prca_num_heads': 4,
        'lsp_reduction': 4,
        'spa_mamba_d_state': 16,
        'spa_mamba_d_conv': 4,
        'spa_mamba_expand': 2,
        'spe_mamba_d_state': 16,
        'spe_mamba_d_conv': 4,
        'spe_mamba_expand': 2,
    }
    kwargs_path = args.model_kwargs_json
    if not kwargs_path:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        # run folder -> dataset folder
        candidate = os.path.join(os.path.dirname(ckpt_dir), 'model_kwargs.json')
        if os.path.exists(candidate):
            kwargs_path = candidate
        elif os.path.exists(os.path.join(ckpt_dir, 'model_kwargs.json')):
            kwargs_path = os.path.join(ckpt_dir, 'model_kwargs.json')
    if kwargs_path and os.path.exists(kwargs_path):
        with open(kwargs_path, 'r') as f:
            model_kwargs.update(json.load(f))
        print('Loaded model kwargs from', kwargs_path)

    net = MambaHSI(in_channels=in_channels, num_classes=class_count, **model_kwargs)
    state = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(state)
    net.to(torch.device(args.device))
    net.eval()
    return net


def main():
    args = get_parser()
    device = torch.device(args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or '.', exist_ok=True)
    if args.pred_cache_dir:
        os.makedirs(args.pred_cache_dir, exist_ok=True)

    need_model = True
    if args.pred_cache_dir:
        cache_paths = [
            os.path.join(args.pred_cache_dir, 'scene1_pred.npy'),
            os.path.join(args.pred_cache_dir, 'scene2_pred.npy'),
        ]
        if all(os.path.exists(p) for p in cache_paths):
            need_model = False
            print('Using cached prediction maps from', args.pred_cache_dir)

    if need_model:
        pca, gaussian_sigmas, class_count = load_or_fit_pca(args)
        net = load_model(args, class_count=class_count, in_channels=pca.n_components_)
    else:
        pca = gaussian_sigmas = net = None
        class_count = args.class_count

    scene_names = ['scene1', 'scene2']
    rows = [('id', 'label')]
    for scene_name in scene_names:
        meta = data_load_operate.load_tree_species_scene_meta(args.data_set_path, scene_name)
        transpose, flatten_order = resolve_flatten_params(meta, args.flatten_mode)
        print(
            'Scene {} flatten: mode={} transpose={} order={} (meta was transpose={} order={})'.format(
                scene_name,
                args.flatten_mode,
                transpose,
                flatten_order,
                meta['transpose'],
                meta['flatten_order'],
            )
        )

        cache_path = ''
        if args.pred_cache_dir:
            cache_path = os.path.join(
                args.pred_cache_dir,
                '{}_{}_pred.npy'.format(meta['scene'], args.layout),
            )

        if cache_path and os.path.exists(cache_path):
            pred = np.load(cache_path)
            print('Loaded pred cache', cache_path, pred.shape)
        else:
            print('Predicting', scene_name, 'layout={} ...'.format(args.layout))
            cube, meta = data_load_operate.load_tree_species_test_cube(
                args.data_set_path, scene_name, layout=args.layout
            )
            print('  cube', cube.shape, 'scene_info', meta['height'], meta['width'], 'h5', meta.get('h5_shape'))
            x = preprocess_cube(
                cube,
                pca,
                gaussian_sigmas,
                args.stretch_low,
                args.stretch_high,
            )
            tile_slices = generate_tile_slices(cube.shape[0], cube.shape[1], args.tile_size, args.tile_overlap)
            pred = predict_tiled(net, x, tile_slices, class_count, device)
            if cache_path:
                np.save(cache_path, pred.astype(np.uint8))
                print('Saved pred cache', cache_path)
            del cube, x
            torch.cuda.empty_cache()

        flat, map_hw = data_load_operate.pack_tree_species_prediction(
            pred, meta, transpose=transpose, flatten_order=flatten_order
        )
        print(
            '  packed map_hw={} flat_len={} transpose={} order={}'.format(
                map_hw.shape, flat.shape[0], transpose, flatten_order
            )
        )
        for idx, label in enumerate(flat):
            rows.append(('{}_{:08d}'.format(meta['scene'], idx), int(label)))
        del pred, flat, map_hw

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print('Wrote', args.output_csv, 'rows=', len(rows) - 1)


if __name__ == '__main__':
    main()
