#!/usr/bin/env python3
"""Draw a paper-style hyperspectral cube from a .mat HSI file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA


def robust_norm(x: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def rgb_from_bands(cube: np.ndarray, bands_1based: list[int]) -> np.ndarray:
    bands = [b - 1 for b in bands_1based]
    if min(bands) < 0 or max(bands) >= cube.shape[2]:
        raise ValueError(f"RGB bands {bands_1based} are outside 1..{cube.shape[2]}")
    channels = [robust_norm(cube[:, :, b]) for b in bands]
    rgb = np.dstack(channels)
    return (rgb * 255).astype(np.uint8)


def jet_colormap(vals: np.ndarray) -> np.ndarray:
    """Small NumPy implementation of the classic blue-green-yellow-red map."""
    v = np.clip(vals, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.dstack([r, g, b])


def heatmap(x: np.ndarray) -> np.ndarray:
    vals = robust_norm(x)
    mapped = jet_colormap(vals)
    return (mapped * 255).astype(np.uint8)


def image_stretching(image: np.ndarray) -> np.ndarray:
    band_list = []
    for i in range(image.shape[2]):
        band_data = image[:, :, i]
        band_min = np.percentile(band_data, 2)
        band_max = np.percentile(band_data, 98)
        if band_max <= band_min:
            stretched = np.zeros_like(band_data, dtype=np.float32)
        else:
            stretched = (band_data - band_min) / (band_max - band_min)
        band_list.append(stretched)
    image_data = np.stack(band_list, axis=-1)
    image_data = np.clip(image_data, 0, 1)
    return (image_data * 255).astype(np.uint8)


def resize_rgb(arr: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(arr).resize(size, Image.Resampling.BICUBIC)


def warp_to_quad(src: Image.Image, quad: list[tuple[float, float]], canvas_size: tuple[int, int]) -> Image.Image:
    src = src.convert("RGBA")
    h, w = np.asarray(src).shape[:2]
    src_corners = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    coeffs = perspective_coeffs(quad, src_corners)
    return src.transform(canvas_size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def perspective_coeffs(
    dst: list[tuple[float, float]],
    src: list[tuple[float, float]],
) -> list[float]:
    matrix = []
    vector = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    return np.linalg.solve(np.asarray(matrix), np.asarray(vector)).tolist()


def alpha_composite(base: Image.Image, layer: Image.Image) -> None:
    base.alpha_composite(layer)


def draw_outline(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], width: int = 5) -> None:
    closed = pts + [pts[0]]
    draw.line(closed, fill=(10, 20, 35, 255), width=width, joint="curve")


def lerp_point(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)


def make_cube(
    cube: np.ndarray,
    rgb_bands: list[int],
    spectral_bands: int,
    front_size: int,
    depth: int,
    out_path: Path,
    transparent: bool,
    style: str,
    edge_width: int | None,
    guide_width: int | None,
) -> None:
    h, w, c = cube.shape
    sampled = np.linspace(0, c - 1, min(spectral_bands, c)).round().astype(int)

    front = resize_rgb(rgb_from_bands(cube, rgb_bands), (front_size, front_size))
    top_data = cube[h // 2, :, sampled].T
    side_data = cube[:, w // 2, sampled]
    top = resize_rgb(heatmap(top_data), (front_size, depth))
    side = resize_rgb(heatmap(side_data), (depth, front_size))

    margin = 42 if style == "envi" else 70
    dx = depth
    dy = -int(depth * (0.52 if style == "envi" else 0.58))
    x0 = margin
    y0 = margin - dy
    fw = fh = front_size
    canvas_size = (x0 + fw + dx + margin, y0 + fh + margin)
    if style == "envi" and not transparent:
        bg = (0, 0, 0, 255)
    else:
        bg = (255, 255, 255, 0 if transparent else 255)
    canvas = Image.new("RGBA", canvas_size, bg)

    front_quad = [(x0, y0), (x0 + fw, y0), (x0 + fw, y0 + fh), (x0, y0 + fh)]
    top_quad = [(x0, y0), (x0 + dx, y0 + dy), (x0 + fw + dx, y0 + dy), (x0 + fw, y0)]
    side_quad = [(x0 + fw, y0), (x0 + fw + dx, y0 + dy), (x0 + fw + dx, y0 + fh + dy), (x0 + fw, y0 + fh)]

    if style != "envi":
        shadow = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.polygon(
            [(x0 + fw + 12, y0 + 25), (x0 + fw + dx + 14, y0 + dy + 20),
             (x0 + fw + dx + 14, y0 + fh + dy + 20), (x0 + fw + 12, y0 + fh + 25)],
            fill=(0, 0, 0, 55),
        )
        alpha_composite(canvas, shadow)

    draw = ImageDraw.Draw(canvas)
    line_width = edge_width if edge_width is not None else (3 if style == "envi" else 5)
    slice_line_width = guide_width if guide_width is not None else (2 if style == "envi" else 3)

    if style == "envi":
        alpha_composite(canvas, warp_to_quad(side, side_quad, canvas_size))
        alpha_composite(canvas, warp_to_quad(top, top_quad, canvas_size))
        alpha_composite(canvas, warp_to_quad(front, front_quad, canvas_size))

        draw = ImageDraw.Draw(canvas)
        draw_outline(draw, top_quad, width=line_width)
        draw_outline(draw, side_quad, width=line_width)
        draw_outline(draw, front_quad, width=line_width)
        draw.line([top_quad[1], top_quad[2], side_quad[2]], fill=(0, 0, 0, 255), width=line_width)
        for t in (0.24, 0.48, 0.72):
            side_top = lerp_point(side_quad[0], side_quad[1], t)
            side_bottom = lerp_point(side_quad[3], side_quad[2], t)
            top_left = lerp_point(top_quad[0], top_quad[1], t)
            top_right = lerp_point(top_quad[3], top_quad[2], t)
            draw.line([side_top, side_bottom], fill=(0, 0, 0, 190), width=slice_line_width)
            draw.line([top_left, top_right], fill=(0, 0, 0, 170), width=slice_line_width)
    else:
        alpha_composite(canvas, warp_to_quad(side, side_quad, canvas_size))
        alpha_composite(canvas, warp_to_quad(top, top_quad, canvas_size))
        alpha_composite(canvas, warp_to_quad(front, front_quad, canvas_size))

        draw = ImageDraw.Draw(canvas)
        draw_outline(draw, top_quad, width=line_width)
        draw_outline(draw, side_quad, width=line_width)
        draw_outline(draw, front_quad, width=line_width)
        draw.line([top_quad[1], top_quad[2], side_quad[2]], fill=(0, 0, 0, 255), width=line_width)
        for t in (0.33, 0.66):
            sx0 = side_quad[0][0] * (1 - t) + side_quad[1][0] * t
            sy0 = side_quad[0][1] * (1 - t) + side_quad[1][1] * t
            sx1 = side_quad[3][0] * (1 - t) + side_quad[2][0] * t
            sy1 = side_quad[3][1] * (1 - t) + side_quad[2][1] * t
            draw.line([(sx0, sy0), (sx1, sy1)], fill=(0, 0, 0, 210), width=slice_line_width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_mat_cube(path: Path, key: str | None) -> np.ndarray:
    mat = sio.loadmat(path)
    if key is None:
        candidates = [(k, v) for k, v in mat.items() if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
        if len(candidates) != 1:
            names = ", ".join(k for k, _ in candidates) or "none"
            raise ValueError(f"Use --key. Found 3-D variables: {names}")
        key, cube = candidates[0]
    else:
        cube = mat[key]
    if cube.ndim != 3:
        raise ValueError(f"{key} must be H x W x C, got {cube.shape}")
    return cube


def reduce_with_current_pca(cube: np.ndarray, n_components: int, sigma: float) -> np.ndarray:
    filtered = gaussian_filter(cube, sigma=sigma)
    reshaped = filtered.reshape(-1, filtered.shape[2])
    reduced = PCA(n_components=n_components).fit_transform(reshaped)
    return reduced.reshape(filtered.shape[0], filtered.shape[1], n_components)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1] / "data" / "indian"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=default_root / "Indian_pines_corrected.mat")
    parser.add_argument("--key", default="indian_pines_corrected")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--style", choices=("envi", "clean"), default="envi")
    parser.add_argument("--rgb-bands", type=int, nargs=3, metavar=("R", "G", "B"))
    parser.add_argument("--spectral-bands", type=int, default=200)
    parser.add_argument("--front-size", type=int, default=420)
    parser.add_argument("--depth", type=int, default=230)
    parser.add_argument("--transparent", action="store_true", help="Export transparent background.")
    parser.add_argument("--edge-width", type=int, help="Outer cube edge width in pixels.")
    parser.add_argument("--guide-width", type=int, help="Internal spectral guide line width in pixels.")
    parser.add_argument("--pca-components", type=int, help="Apply the same PCA preprocessing as train.py.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Sigma used before PCA, matching train.py default.")
    parser.add_argument("--no-pca-stretch", action="store_true", help="Skip ImageStretching after PCA.")
    parser.add_argument("--save-pca-mat", type=Path, help="Optional path for saving the reduced PCA cube as .mat.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cube = load_mat_cube(args.mat, args.key)
    if args.pca_components is not None:
        cube = reduce_with_current_pca(cube, args.pca_components, args.gaussian_sigma)
        if not args.no_pca_stretch:
            cube = image_stretching(cube)
        if args.save_pca_mat is not None:
            args.save_pca_mat.parent.mkdir(parents=True, exist_ok=True)
            sio.savemat(args.save_pca_mat, {f"pca_{args.pca_components}": cube})

    out_path = args.out
    if out_path is None:
        suffix = f"pca{args.pca_components}" if args.pca_components is not None else ("envi_like" if args.style == "envi" else "clean")
        out_path = args.mat.parent / "visualizations" / f"indian_pines_spectral_cube_{suffix}.png"
    rgb_bands = args.rgb_bands
    if rgb_bands is None:
        rgb_bands = [3, 2, 1] if args.pca_components is not None else [29, 19, 10]

    make_cube(
        cube=cube,
        rgb_bands=rgb_bands,
        spectral_bands=args.spectral_bands,
        front_size=args.front_size,
        depth=args.depth,
        out_path=out_path,
        transparent=args.transparent,
        style=args.style,
        edge_width=args.edge_width,
        guide_width=args.guide_width,
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
