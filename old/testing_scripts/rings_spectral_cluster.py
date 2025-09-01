#!/usr/bin/env python3
import os
import argparse
import logging
from glob import glob

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ----------------------- logging -----------------------
def setup_logger():
    log = logging.getLogger("RingsSpectralCluster")
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

# ----------------------- preprocessing -----------------------
def zscore_per_band(cube):
    # cube: H×W×B
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0) + 1e-8
    norm = (flat - mu) / sd
    return norm.reshape(H, W, B)

def pca_pc1(cube_norm):
    H, W, B = cube_norm.shape
    flat = cube_norm.reshape(-1, B)
    pc1 = PCA(n_components=1, svd_solver="randomized", random_state=42).fit_transform(flat)
    pc1_img = pc1.reshape(H, W).astype(np.float32)
    return pc1_img

# ----------------------- geometry helpers -----------------------
def to_float_center(center):
    return (float(center[0]), float(center[1]))

def max_radius_cover_all(center, H, W):
    cx, cy = center
    corners = [(0.0, 0.0), (W-1.0, 0.0), (0.0, H-1.0), (W-1.0, H-1.0)]
    return float(max(np.hypot(cx-x, cy-y) for (x, y) in corners))

def unwrap_polar(image2d, center):
    # image2d: H×W float32
    H, W = image2d.shape
    center = to_float_center(center)
    maxR = max_radius_cover_all(center, H, W)
    dsize = (360, int(np.ceil(maxR)))  # (theta, radius)
    p = cv2.warpPolar(np.ascontiguousarray(image2d), dsize, center, float(maxR),
                      flags=cv2.WARP_POLAR_LINEAR)
    if np.isnan(p).any():
        finite = p[~np.isnan(p)]
        fill = float(finite.min()) if finite.size else 0.0
        p = np.where(np.isnan(p), fill, p)
    return p  # R×theta

def auto_center_pc1(pc1, logger, grid=9):
    H, W = pc1.shape
    base = (W//2, H//2)
    # sweep mainly in X (rings center often left of board)
    offsets_x = np.linspace(-W, W, grid, dtype=int)
    best, best_score = base, -1.0
    for dx in offsets_x:
        c = (base[0] + int(dx), base[1])
        pol = unwrap_polar(pc1, c)
        # radial gradient → coverage score
        g = np.abs(np.diff(pol, axis=0, prepend=pol[0:1,:]))
        cov = (g > (np.median(g,axis=1,keepdims=True) +
                    2.0*np.median(np.abs(g - np.median(g,axis=1,keepdims=True)), axis=1, keepdims=True)
                   )).mean(axis=1)
        peaks, props = find_peaks(uniform_filter1d(cov, 5), distance=15, height=0.2)
        score = float(len(peaks) * (np.mean(props["peak_heights"]) if len(peaks) else 0.0))
        if score > best_score:
            best, best_score = c, score
    logger.info(f"Auto-center → {best} (score={best_score:.3f})")
    return best

# ----------------------- boundary detection in polar -----------------------
def detect_polar_edges(pc1_polar, smooth=5, k_mad=2.0, min_distance=15):
    """
    Returns arrays r_idx, t_idx with boundary pixels in polar.
    """
    # radial gradient (vertical in polar)
    grad = np.abs(np.diff(pc1_polar, axis=0, prepend=pc1_polar[0:1,:]))
    R, T = grad.shape

    # adaptive threshold per radius: median + k*MAD
    med = np.median(grad, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(grad - med), axis=1, keepdims=True)
    thr = med + k_mad * mad
    cand = grad > thr

    # smooth coverage to later enforce distances
    cov = uniform_filter1d(cand.mean(axis=1), size=max(1, smooth))
    peaks, _ = find_peaks(cov, distance=min_distance, height=np.mean(cov))
    # keep only boundary pixels within a small band around those radii
    keep = np.zeros_like(cand, dtype=bool)
    band = 2  # +/- pixels around each peak
    for r in peaks:
        r0, r1 = max(0, r-band), min(R, r+band+1)
        keep[r0:r1, :] |= cand[r0:r1, :]

    r_idx, t_idx = np.where(keep)
    return r_idx.astype(np.int32), t_idx.astype(np.int32), cov

# ----------------------- map boundary pixels back to image -----------------------
def polar_to_xy(r_idx, t_idx, center):
    # theta in degrees; convert to radians
    theta = (t_idx.astype(np.float32) * np.pi) / 180.0
    x = center[0] + r_idx * np.cos(theta)
    y = center[1] + r_idx * np.sin(theta)
    return x, y  # float arrays

# ----------------------- spectral clustering of boundary pixels -----------------------
def cluster_boundaries_by_spectrum(cube, x, y, sample_max=40000):
    """
    Sample boundary pixels, pull their full spectra, cluster with KMeans(k=2).
    Returns labels for all boundary pixels (mapped from the sample model).
    """
    H, W, B = cube.shape
    # clamp coords to valid pixels
    xi = np.clip(np.round(x).astype(int), 0, W-1)
    yi = np.clip(np.round(y).astype(int), 0, H-1)

    # unique indices to avoid duplicates
    lin = yi * W + xi
    uniq, inv = np.unique(lin, return_inverse=True)

    # assemble spectra for unique points
    yy = (uniq // W).astype(int)
    xx = (uniq %  W).astype(int)
    spectra = cube[yy, xx, :]  # Nuniq × B

    # optionally subsample for fitting
    if spectra.shape[0] > sample_max:
        sel = np.random.RandomState(42).choice(spectra.shape[0], sample_max, replace=False)
        fitX = spectra[sel]
    else:
        fitX = spectra

    # scale per feature (band) – z-score so KMeans isn't dominated by bright bands
    mu = fitX.mean(axis=0, keepdims=True)
    sd = fitX.std(axis=0, keepdims=True) + 1e-8
    fitZ = (fitX - mu) / sd

    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(fitZ)

    # map all unique spectra through same normalization and model
    spectraZ = (spectra - mu) / sd
    uniq_labels = km.predict(spectraZ)

    # expand back to every boundary pixel
    full_labels = uniq_labels[inv]
    return full_labels, uniq_labels, (yy, xx), km.inertia_

# ----------------------- choose "true" cluster by angular coverage -----------------------
def choose_true_cluster(r_idx, t_idx, labels, R, T):
    """
    Build coverage curves per cluster across radii. The cluster with higher
    mean coverage is considered 'true'.
    """
    cov0 = np.zeros(R, dtype=np.float32)
    cov1 = np.zeros(R, dtype=np.float32)
    for r, t, lb in zip(r_idx, t_idx, labels):
        if lb == 0:
            cov0[r] += 1
        else:
            cov1[r] += 1
    # normalize by number of angles (approx coverage)
    cov0 /= float(T)
    cov1 /= float(T)
    mean0, mean1 = cov0.mean(), cov1.mean()
    true_label = 0 if mean0 >= mean1 else 1
    return true_label, cov0, cov1

# ----------------------- peak picking on true coverage -----------------------
def radii_from_coverage(cov_true, smooth=7, min_distance=18, min_height=None):
    if smooth > 1:
        cov_s = uniform_filter1d(cov_true, size=smooth)
    else:
        cov_s = cov_true
    if min_height is None:
        min_height = max(0.2, cov_s.mean())
    peaks, props = find_peaks(cov_s, distance=min_distance, height=min_height)
    return peaks.astype(int), cov_s, props

# ----------------------- metrics & overlays -----------------------
def ring_metrics(cube, radii, center):
    H, W, _ = cube.shape
    pc1 = pca_pc1(zscore_per_band(cube))
    cX, cY = center
    yy, xx = np.indices((H, W), dtype=np.float32)
    rr = np.sqrt((xx - cX)**2 + (yy - cY)**2)

    rows = []
    for i in range(len(radii)-1):
        r0, r1 = int(radii[i]), int(radii[i+1])
        shell = (rr >= r0) & (rr < r1)
        dens = float(pc1[shell].mean()) if np.any(shell) else np.nan
        rows.append((i+1, r0, r1, int(r1-r0), dens))
    return pd.DataFrame(rows, columns=["ring_idx","inner_r_px","outer_r_px","width_px","density_pc1"])

def overlay_on_band1(cube, center, r_idx, t_idx, labels, true_label, out_path):
    band1 = cube[:, :, 0]
    H, W = band1.shape
    img = cv2.normalize(band1, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # map polar pixels to xy and draw small colored dots
    x, y = polar_to_xy(r_idx, t_idx, center)
    xi = np.clip(np.round(x).astype(int), 0, W-1)
    yi = np.clip(np.round(y).astype(int), 0, H-1)

    true_mask = labels == true_label
    false_mask = ~true_mask

    img[yi[true_mask], xi[true_mask]] = (0, 255, 0)   # green
    img[yi[false_mask], xi[false_mask]] = (255, 0, 0) # red

    cv2.imwrite(out_path, img)

def save_curve_plot(curves, labels, out_path):
    plt.figure(figsize=(8,2.2))
    for c, lab in zip(curves, labels):
        plt.plot(c, label=lab)
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel("Radius (px)")
    plt.ylabel("Coverage")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ----------------------- main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Unsupervised spectral+geometric tree-ring detector (no labels)")
    ap.add_argument("-i","--input_dir",  default="prepocessing/processed_cubes")
    ap.add_argument("-o","--output_dir", default="spectral_cluster_rings")
    ap.add_argument("--auto_center", action="store_true")
    ap.add_argument("--center", type=str, default="", help="Manual center 'x,y' (overrides auto)")
    ap.add_argument("--smooth", type=int, default=5, help="Smoothing for polar coverage")
    ap.add_argument("--k_mad", type=float, default=2.0, help="MAD factor for edge threshold")
    ap.add_argument("--min_distance", type=int, default=18, help="Min ring spacing (px)")
    ap.add_argument("--sample_max", type=int, default=40000, help="Max boundary spectra used to fit KMeans")
    args = ap.parse_args()

    log = setup_logger()
    os.makedirs(args.output_dir, exist_ok=True)

    cubes = load_cubes(args.input_dir, log)

    manual_center = None
    if args.center:
        try:
            x, y = args.center.split(",")
            manual_center = (float(x), float(y))
        except Exception:
            log.warning("Could not parse --center 'x,y'; ignoring.")

    for key, cube in cubes.items():
        log.info(f"=== {key} ===")
        # Normalize & PC1 for geometry + edges
        cube_norm = zscore_per_band(cube)
        pc1 = pca_pc1(cube_norm)

        # pick center
        if manual_center is not None:
            center = manual_center
        elif args.auto_center:
            center = auto_center_pc1(pc1, log)
        else:
            center = (cube.shape[1]//2, cube.shape[0]//2)

        # polar edges on PC1
        polar_pc1 = unwrap_polar(pc1, center)
        r_idx, t_idx, cov_all = detect_polar_edges(polar_pc1, smooth=args.smooth,
                                                   k_mad=args.k_mad,
                                                   min_distance=args.min_distance)
        R, T = polar_pc1.shape
        log.info(f"Boundary pixels found: {len(r_idx)}")

        if len(r_idx) == 0:
            # save empty artifacts for traceability
            cv2.imwrite(os.path.join(args.output_dir, f"{key}_annot.png"),
                        cv2.normalize(cube[:,:,0], None, 0,255, cv2.NORM_MINMAX).astype(np.uint8))
            pd.DataFrame(columns=["ring_idx","inner_r_px","outer_r_px","width_px","density_pc1"]) \
                .to_csv(os.path.join(args.output_dir, f"{key}_rings.csv"), index=False)
            save_curve_plot([cov_all], ["edge_cov"], os.path.join(args.output_dir, f"{key}_coverage.png"))
            log.info("No boundaries — wrote placeholders.")
            continue

        # cluster boundary pixels by their full spectra
        x, y = polar_to_xy(r_idx, t_idx, center)
        labels, uniq_labels, (yy, xx), inertia = cluster_boundaries_by_spectrum(
            cube, x, y, sample_max=args.sample_max)

        # decide which cluster is 'true' by coverage consistency
        true_label, cov0, cov1 = choose_true_cluster(r_idx, t_idx, labels, R, T)
        cov_true = cov0 if true_label == 0 else cov1
        cov_false = cov1 if true_label == 0 else cov0

        # peak pick on true coverage
        ring_radii, cov_s, props = radii_from_coverage(cov_true,
                                                       smooth=max(3, args.smooth),
                                                       min_distance=args.min_distance)
        log.info(f"Rings selected: {len(ring_radii)}")

        # metrics
        df = ring_metrics(cube, ring_radii, center)
        df.to_csv(os.path.join(args.output_dir, f"{key}_rings.csv"), index=False)

        # overlays & curves
        overlay_on_band1(cube, center, r_idx, t_idx, labels, true_label,
                         os.path.join(args.output_dir, f"{key}_annot.png"))
        save_curve_plot([cov_true, cov_false, cov_all],
                        ["true_cov", "false_cov", "edge_cov"],
                        os.path.join(args.output_dir, f"{key}_coverage.png"))

        # also save a simple text summary
        with open(os.path.join(args.output_dir, f"{key}_summary.txt"), "w") as f:
            f.write(f"center={center}\n")
            f.write(f"boundary_pixels={len(r_idx)}\n")
            f.write(f"true_label={true_label}\n")
            f.write(f"rings_detected={len(ring_radii)}\n")
            if len(ring_radii) > 1:
                widths = np.diff(ring_radii)
                f.write(f"mean_width_px={float(widths.mean()):.2f}\n")

if __name__ == "__main__":
    main()
