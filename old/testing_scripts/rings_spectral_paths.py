#!/usr/bin/env python3
# rings_spectral_paths.py
#
# Unsupervised tree-ring detection for trimmed boards:
# - No circles. Curvilinear ring boundaries traced as smooth paths.
# - Uses spectral contrast (SAM over all bands) to keep "true" boundaries.
# - Outputs: overlay PNG (true=green, false=red) + CSV with ring widths & SAM.

import os
import argparse
import logging
from glob import glob

import numpy as np
import pandas as pd
import cv2
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# ------------------------ Logging ------------------------
def setup_logger():
    log = logging.getLogger("RingsSpectralPaths")
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(ch)
    return log

# ------------------------ I/O ------------------------
def load_cubes(input_dir, logger):
    files = sorted(glob(os.path.join(input_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files in {input_dir}")
    cubes = {}
    for fn in files:
        key = os.path.splitext(os.path.basename(fn))[0]
        arr = np.load(fn)
        cube = arr[list(arr.keys())[0]].astype(np.float32)
        logger.info(f"Loaded '{key}' shape={cube.shape}")
        cubes[key] = cube
    return cubes

# ------------------------ Preprocess ------------------------
def zscore_per_band(cube):
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-8
    norm = (flat - mu) / sd
    return norm.reshape(H, W, B)

def pc1_from_cube(cube_norm):
    H, W, B = cube_norm.shape
    flat = cube_norm.reshape(-1, B)
    pc1 = PCA(n_components=1, svd_solver="randomized", random_state=42).fit_transform(flat)
    return pc1.reshape(H, W).astype(np.float32)

# ------------------------ Edge map (no circles) ------------------------
def edge_map_from_pc1(pc1, blur_sigma=1.0):
    # Light smoothing, then Sobel gradients → edge strength
    img = gaussian_filter(pc1, blur_sigma)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # robust 0-1 normalize by percentiles
    p1, p99 = np.percentile(mag, (1, 99))
    mag = (mag - p1) / max(1e-8, (p99 - p1))
    return np.clip(mag, 0, 1), gx, gy

# ------------------------ Seam search (dynamic programming) ------------------------
def find_best_path(edge, dmax=3, lambda_smooth=0.5):
    """
    Max-sum dynamic programming from left→right across width.
    edge: H×W in [0,1]
    Returns y_path (length W), total_score.
    """
    H, W = edge.shape
    # DP tables
    dp = np.full((H, W), -1e9, dtype=np.float32)
    prev = np.full((H, W), -1, dtype=np.int32)
    dp[:, 0] = edge[:, 0]
    for x in range(1, W):
        for y in range(H):
            y0 = max(0, y - dmax)
            y1 = min(H, y + dmax + 1)
            prev_slice = dp[y0:y1, x - 1] - lambda_smooth * np.abs(np.arange(y0, y1) - y)
            j = np.argmax(prev_slice)
            dp[y, x] = edge[y, x] + prev_slice[j]
            prev[y, x] = y0 + j
    # backtrack
    y_end = int(np.argmax(dp[:, -1]))
    total = float(dp[y_end, -1])
    y_path = np.zeros(W, dtype=np.int32)
    y_path[-1] = y_end
    for x in range(W - 1, 0, -1):
        y_path[x - 1] = prev[y_path[x], x]
    # average edge value along path as sanity score
    mean_edge = float(edge[y_path, np.arange(W)].mean())
    return y_path, total, mean_edge

def mask_around_path(edge, y_path, band_px=6):
    H, W = edge.shape
    mask = edge.copy()
    xs = np.arange(W)
    for x, y in zip(xs, y_path):
        y0 = max(0, y - band_px)
        y1 = min(H, y + band_px + 1)
        mask[y0:y1, x] = 0.0
    return mask

# ------------------------ Spectral scoring via SAM ------------------------
def spectral_angle(a, b, eps=1e-8):
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    v = np.clip(num / den, -1.0, 1.0)
    return float(np.arccos(v))  # radians

def sam_along_path(cube, gx, gy, y_path, step_px=2):
    """
    For each x along the path, sample spectra on the two sides of the boundary
    along the NORMAL direction to PC1 gradient. Returns mean SAM angle (radians).
    """
    H, W, B = cube.shape
    angles = []
    xs = np.arange(W)
    for x, y in zip(xs, y_path):
        # local gradient at (y, x)
        g = np.array([gx[y, x], gy[y, x]], dtype=np.float32)
        if np.linalg.norm(g) < 1e-6:
            n = np.array([0.0, 1.0], dtype=np.float32)   # fallback
        else:
            n = g / np.linalg.norm(g)  # normal direction (across boundary)
        # sample inside/outside
        p1 = (int(round(x - step_px * n[0])), int(round(y - step_px * n[1])))
        p2 = (int(round(x + step_px * n[0])), int(round(y + step_px * n[1])))
        # clamp
        p1 = (np.clip(p1[0], 0, W - 1), np.clip(p1[1], 0, H - 1))
        p2 = (np.clip(p2[0], 0, W - 1), np.clip(p2[1], 0, H - 1))
        a = cube[p1[1], p1[0], :]
        b = cube[p2[1], p2[0], :]
        ang = spectral_angle(a, b)
        angles.append(ang)
    return float(np.mean(angles)), np.array(angles, dtype=np.float32)

# ------------------------ Ring widths from kept paths ------------------------
def ring_widths_from_paths(paths):
    """
    Given a list of kept paths (each is y[x]), sort by mean y and compute
    median vertical separations between consecutive paths as ring widths.
    """
    if len(paths) < 2:
        return []
    W = paths[0].shape[0]
    # sort by mean y (top→bottom)
    order = np.argsort([p.mean() for p in paths])
    paths_sorted = [paths[i] for i in order]
    widths = []
    for i in range(len(paths_sorted) - 1):
        dy = paths_sorted[i + 1] - paths_sorted[i]  # per-column distance
        widths.append(float(np.median(dy)))
    return widths

# ------------------------ Visualization ------------------------
def overlay_paths_on_band1(cube, paths_true, paths_false, out_path):
    band1 = cube[:, :, 0]
    H, W = band1.shape
    vis = cv2.normalize(band1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    xs = np.arange(W)

    def draw(paths, color):
        for y_path in paths:
            pts = np.stack([xs, y_path], axis=1).astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=1)

    draw(paths_false, (0, 0, 255))   # red = false
    draw(paths_true,  (0, 255, 0))   # green = true
    cv2.imwrite(out_path, vis)

# ------------------------ Main pipeline ------------------------
def process_cube(cube, max_paths, edge_blur, dmax, lam, band_px,
                 sam_step, sam_thresh, min_mean_edge, logger):
    """
    Returns dict with: paths_true, paths_false, sam_scores_true, widths, edge_map
    """
    cube_norm = zscore_per_band(cube)
    pc1 = pc1_from_cube(cube_norm)
    edge, gx, gy = edge_map_from_pc1(pc1, blur_sigma=edge_blur)

    H, W = edge.shape
    paths_true, paths_false, sam_true = [], [], []

    work = edge.copy()
    for k in range(max_paths):
        y_path, total, mean_edge = find_best_path(work, dmax=dmax, lambda_smooth=lam)
        if mean_edge < min_mean_edge:
            break
        # score spectrally with SAM across the path
        mean_sam, all_sam = sam_along_path(cube, gx, gy, y_path, step_px=sam_step)
        # classify
        if mean_sam >= sam_thresh:
            paths_true.append(y_path)
            sam_true.append(mean_sam)
        else:
            paths_false.append(y_path)
        # mask around the path to find the next one
        work = mask_around_path(work, y_path, band_px=band_px)

    widths = ring_widths_from_paths(paths_true)
    return {
        "paths_true": paths_true,
        "paths_false": paths_false,
        "sam_true": sam_true,
        "edge_map": edge,
        "widths": widths
    }

# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser(description="Unsupervised ring boundaries via spectral SAM + path tracing (no circles)")
    ap.add_argument("-i","--input_dir",  default="prepocessing/processed_cubes")
    ap.add_argument("-o","--output_dir", default="spectral_paths")
    # path search & edge params
    ap.add_argument("--max_paths", type=int, default=40, help="Max candidate boundaries to trace")
    ap.add_argument("--edge_blur", type=float, default=1.0, help="Gaussian sigma for PC1 before Sobel")
    ap.add_argument("--dmax", type=int, default=3, help="Max vertical step per column (path smoothness)")
    ap.add_argument("--lambda_smooth", type=float, default=0.6, help="Smoothness penalty (higher = smoother)")
    ap.add_argument("--band_px", type=int, default=6, help="Mask half-band after selecting a path")
    # spectral scoring
    ap.add_argument("--sam_step", type=int, default=2, help="Pixels to sample either side of boundary for SAM")
    ap.add_argument("--sam_thresh", type=float, default=0.08, help="Keep path if mean SAM ≥ this (radians)")
    # stopping / sanity
    ap.add_argument("--min_mean_edge", type=float, default=0.18, help="Stop if next path’s mean edge < this")
    args = ap.parse_args()

    log = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    cubes = load_cubes(args.input_dir, log)

    for key, cube in cubes.items():
        log.info(f"=== {key} ===")
        out = process_cube(
            cube,
            max_paths=args.max_paths,
            edge_blur=args.edge_blur,
            dmax=args.dmax,
            lam=args.lambda_smooth,
            band_px=args.band_px,
            sam_step=args.sam_step,
            sam_thresh=args.sam_thresh,
            min_mean_edge=args.min_mean_edge,
            logger=log
        )

        # overlay
        png = os.path.join(args.output_dir, f"{key}_annot.png")
        overlay_paths_on_band1(cube, out["paths_true"], out["paths_false"], png)
        log.info(f"Overlay → {png}")

        # CSV metrics
        rows = []
        # sort kept paths by mean y so ring_idx increases top→bottom
        order = np.argsort([p.mean() for p in out["paths_true"]]) if out["paths_true"] else []
        for i, idx in enumerate(order):
            rows.append({
                "ring_idx": i+1,
                "mean_sam": out["sam_true"][idx],
                "median_width_px": (out["widths"][i] if i < len(out["widths"]) else np.nan)
            })
        df = pd.DataFrame(rows, columns=["ring_idx","mean_sam","median_width_px"])
        csv_path = os.path.join(args.output_dir, f"{key}_rings.csv")
        df.to_csv(csv_path, index=False)
        log.info(f"Metrics → {csv_path}")

if __name__ == "__main__":
    main()
