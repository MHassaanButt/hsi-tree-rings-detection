#!/usr/bin/env python3
# rings_consensus_polar.py
import os
import argparse
import logging
from glob import glob

import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ----------------------- Logging -----------------------
def setup_logger():
    log = logging.getLogger("RingsConsensusPolar")
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(ch)
    return log

# ----------------------- I/O -----------------------
def load_cubes(input_dir, logger):
    files = glob(os.path.join(input_dir, "*.npz"))
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

# ----------------------- Geometry helpers -----------------------
def _to_float_center(center):
    # Ensure plain Python floats (cv2 warpPolar is picky)
    return (float(center[0]), float(center[1]))

def _max_radius_cover_all(center, H, W):
    """Max distance from center to any image corner (covers whole image)."""
    cx, cy = center
    corners = [(0.0, 0.0), (W-1.0, 0.0), (0.0, H-1.0), (W-1.0, H-1.0)]
    dists = [np.hypot(cx-x, cy-y) for (x, y) in corners]
    return float(np.max(dists))

def warp_polar_band(band, center, maxR):
    center = _to_float_center(center)
    dsize = (360, int(np.ceil(maxR)))  # (width=angles, height=radii)
    p = cv2.warpPolar(
        np.ascontiguousarray(band.astype(np.float32)),
        dsize,
        center,
        float(maxR),
        cv2.WARP_POLAR_LINEAR
    )
    # Replace NaNs (if any) with the band’s finite min
    if np.isnan(p).any():
        finite_vals = p[~np.isnan(p)]
        fill = float(finite_vals.min()) if finite_vals.size else 0.0
        p = np.where(np.isnan(p), fill, p)
    return p

def unwrap_to_polar(cube, center=None):
    H, W, B = cube.shape
    if center is None:
        center = (W // 2, H // 2)
    center = _to_float_center(center)
    maxR = _max_radius_cover_all(center, H, W)

    polar = np.zeros((int(np.ceil(maxR)), 360, B), dtype=np.float32)  # (R, Theta, B)
    for b in range(B):
        polar[:, :, b] = warp_polar_band(cube[:, :, b], center, maxR)
    return polar, center, maxR

# ----------------------- SAM radial change -----------------------
def sam_radial_change(polar):
    """
    Spectral Angle Mapper between consecutive radii.
    polar: (R, T, B)
    returns: sam_map (R-1, T) in radians
    """
    A = polar[:-1, :, :]  # (R-1, T, B)
    B = polar[1:, :, :]
    dot = (A * B).sum(axis=2)
    na = np.linalg.norm(A, axis=2) + 1e-8
    nb = np.linalg.norm(B, axis=2) + 1e-8
    cos = np.clip(dot / (na * nb), -1.0 + 1e-7, 1.0 - 1e-7)
    return np.arccos(cos).astype(np.float32)  # (R-1, T)

def consensus_curve_from_sam(sam_map, smooth=5, k_mad=2.0):
    """
    For each radius r, compute the fraction of angles where SAM > (median + k*MAD).
    Returns coverage curve (R-1,)
    """
    Rm1, T = sam_map.shape
    cov = np.zeros(Rm1, dtype=np.float32)
    for r in range(Rm1):
        row = sam_map[r, :]
        med = np.median(row)
        mad = np.median(np.abs(row - med)) * 1.4826  # robust sigma
        thr = med + k_mad * mad
        cov[r] = float((row > thr).mean())
    if smooth > 1:
        cov = uniform_filter1d(cov, size=smooth, mode="nearest")
    return cov

def detect_ring_radii(cov, min_distance=15, min_height=0.25):
    """
    Peaks in the consensus curve are ring boundaries.
    cov: (R-1,) in [0,1]
    """
    peaks, props = find_peaks(cov, distance=min_distance, height=min_height)
    return peaks.astype(int), props

# ----------------------- Metrics & annotation -----------------------
def ring_metrics_from_radii(cube, radii, center):
    """
    Compute width and a simple intensity 'density' per ring shell using band-1.
    """
    H, W, _ = cube.shape
    band1 = cube[:, :, 0]
    cX, cY = center
    yy, xx = np.indices((H, W))
    rr = np.sqrt((xx - cX) ** 2 + (yy - cY) ** 2)

    rows = []
    for i in range(len(radii) - 1):
        r0, r1 = int(radii[i]), int(radii[i + 1])
        shell = (rr >= r0) & (rr < r1)
        dens = float(band1[shell].mean()) if np.any(shell) else np.nan
        width = int(r1 - r0)
        rows.append((i + 1, int(r0), int(r1), width, dens))
    return pd.DataFrame(rows, columns=["ring_idx", "inner_r_px", "outer_r_px", "width_px", "density_band1"])

def annotate_on_band1(cube, center, radii, cov, out_path):
    band1 = cube[:, :, 0]
    H, W = band1.shape
    img = cv2.normalize(band1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cX, cY = int(round(center[0])), int(round(center[1]))

    # Normalize cov for color mapping
    if len(cov) > 0:
        cov_n = (cov - cov.min()) / max(1e-8, (cov.max() - cov.min()))
    else:
        cov_n = np.array([])

    for idx, r in enumerate(radii):
        c = float(cov_n[min(idx, max(0, len(cov_n) - 1))]) if len(cov_n) else 0.5
        color = (int(255 * (1 - c)), int(255 * c), 0)  # red→green
        cv2.circle(img, (cX, cY), int(r), color, 2)
        cv2.putText(img, f"{idx+1}", (cX + int(r) + 2, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)

def save_consensus_plot(cov, out_path):
    plt.figure(figsize=(8, 2.2))
    plt.plot(cov, lw=1.5)
    plt.ylim(0, 1)
    plt.xlabel("Radius (px)")
    plt.ylabel("Angle coverage")
    plt.title("Consensus: fraction of angles with strong spectral change")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ----------------------- Optional: auto-center -----------------------
def quick_score_for_center(cube, center, smooth=5, k_mad=1.5, min_distance=15):
    polar, _, _ = unwrap_to_polar(cube, center)
    sam = sam_radial_change(polar)
    cov = consensus_curve_from_sam(sam, smooth=smooth, k_mad=k_mad)
    peaks, props = find_peaks(cov, distance=min_distance, height=0.2)
    return float(len(peaks) * np.mean(props["peak_heights"])) if len(peaks) else 0.0

def auto_estimate_center(cube, logger, grid=7):
    H, W, _ = cube.shape
    base = (W // 2, H // 2)
    # Sweep only in X (common for boards); extend if needed
    offsets_x = np.linspace(-W, W, grid, dtype=int)
    best_center, best_score = base, -1.0
    for dx in offsets_x:
        cand = (base[0] + int(dx), base[1])
        score = quick_score_for_center(cube, cand)
        if score > best_score:
            best_center, best_score = cand, score
    best_center = _to_float_center(best_center)
    logger.info(f"Auto-center best at {best_center} (score={best_score:.3f})")
    return best_center

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Unsupervised tree-ring detection with SAM consensus in polar space")
    ap.add_argument("-i", "--input_dir",  default="prepocessing/processed_cubes")
    ap.add_argument("-o", "--output_dir", default="consensus_rings")
    ap.add_argument("--auto_center", action="store_true", help="Try to auto-estimate a better polar center")
    ap.add_argument("--center", type=str, default="", help="Manual center as 'x,y' (overrides auto)")
    ap.add_argument("--smooth", type=int, default=5, help="Smoothing window for consensus curve")
    ap.add_argument("--k_mad", type=float, default=2.0, help="Robust threshold = median + k*MAD")
    ap.add_argument("--min_distance", type=int, default=20, help="Min px between ring peaks")
    ap.add_argument("--min_coverage", type=float, default=0.25, help="Min angle coverage for a ring (0–1)")
    args = ap.parse_args()

    log = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)
    cubes = load_cubes(args.input_dir, log)

    # Parse manual center if provided
    manual_center = None
    if args.center:
        try:
            x, y = args.center.split(",")
            manual_center = (float(x), float(y))
        except Exception:
            log.warning("Could not parse --center; expected 'x,y'. Ignoring.")

    for key, cube in cubes.items():
        log.info(f"=== {key} ===")

        # center selection
        if manual_center is not None:
            center = manual_center
        elif args.auto_center:
            center = auto_estimate_center(cube, log)
        else:
            center = _to_float_center((cube.shape[1] // 2, cube.shape[0] // 2))

        # unwrap + SAM
        polar, center, maxR = unwrap_to_polar(cube, center)
        sam = sam_radial_change(polar)
        cov = consensus_curve_from_sam(sam, smooth=args.smooth, k_mad=args.k_mad)

        # peak detection
        radii, props = detect_ring_radii(cov, min_distance=args.min_distance,
                                         min_height=args.min_coverage)
        log.info(f"Detected {len(radii)} rings (min_coverage={args.min_coverage})")

        # metrics & saves
        df = ring_metrics_from_radii(cube, radii, center)
        csvp = os.path.join(args.output_dir, f"{key}_rings.csv")
        df.to_csv(csvp, index=False)
        log.info(f"Saved metrics → {csvp}")

        out_img = os.path.join(args.output_dir, f"{key}_annot.png")
        annotate_on_band1(cube, center, radii, cov, out_img)
        log.info(f"Saved annotation → {out_img}")

        cov_plot = os.path.join(args.output_dir, f"{key}_consensus.png")
        save_consensus_plot(cov, cov_plot)

if __name__ == "__main__":
    main()
