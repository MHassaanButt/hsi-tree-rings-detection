#!/usr/bin/env python3
import os, argparse, logging
from glob import glob
import numpy as np
import pandas as pd
import cv2
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter, uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import nnls

# ---------- logging ----------
def get_logger():
    lg = logging.getLogger("ChemRings")
    lg.setLevel(logging.DEBUG)
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    lg.addHandler(h)
    return lg

# ---------- I/O ----------
def load_cubes(inp, log):
    files = sorted(glob(os.path.join(inp, "*.npz")))
    if not files: raise FileNotFoundError(f"No .npz in {inp}")
    cubes = {}
    for f in files:
        key = os.path.splitext(os.path.basename(f))[0]
        arr = np.load(f)
        cube = arr[list(arr.keys())[0]].astype(np.float32)
        cubes[key] = cube
        log.info(f"Loaded {key} {cube.shape}")
    return cubes

def load_wavelengths(path, nbands):
    if path and os.path.exists(path):
        wl = np.load(path).astype(np.float32)
        assert wl.size == nbands, f"wavelengths size {wl.size} != {nbands}"
        return wl
    # fallback: linear assumption (edit to your sensor)
    return np.linspace(400, 1000, nbands, dtype=np.float32)  # nm

def load_library(lib_dir, names=("cellulose","lignin","water")):
    # Each CSV: two cols: wavelength_nm, reflectance
    lib = {}
    for n in names:
        p = os.path.join(lib_dir, f"{n}.csv")
        if not os.path.exists(p):
            continue
        data = np.loadtxt(p, delimiter=",", dtype=np.float32)
        wl_lib, refl = data[:,0], data[:,1]
        lib[n] = (wl_lib, refl)
    return lib

def resample_spectrum(wl_from, spec_from, wl_to):
    return np.interp(wl_to, wl_from, spec_from).astype(np.float32)

# ---------- preprocessing ----------
def snv_per_pixel(cube):
    H,W,B = cube.shape
    X = cube.reshape(-1,B)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-8
    Z = (X - mu) / sd
    return Z.reshape(H,W,B).astype(np.float32)

def first_derivative(cube, window=11, poly=2):
    # Savitzky–Golay along spectral axis
    H,W,B = cube.shape
    X = cube.reshape(-1,B)
    d = savgol_filter(X, window_length=min(window,B//2*2+1), polyorder=min(poly,2), deriv=1, axis=1)
    return d.reshape(H,W,B).astype(np.float32)

# ---------- spectral angle mapper ----------
def sam_similarity(cube, ref):  # returns cos(angle) in [-1,1], higher is more similar
    # normalize both
    H,W,B = cube.shape
    X = cube.reshape(-1,B)
    Xn = X / (np.linalg.norm(X,axis=1,keepdims=True)+1e-8)
    rn = ref / (np.linalg.norm(ref)+1e-8)
    cos = (Xn @ rn).reshape(H,W).astype(np.float32)
    return cos

# ---------- NNLS abundances ----------
def nnls_abundances(cube, M):  # M: K×B endmember matrix
    H,W,B = cube.shape
    X = cube.reshape(-1,B)
    K = M.shape[0]
    A = np.zeros((X.shape[0], K), dtype=np.float32)
    for i in range(X.shape[0]):
        a, _ = nnls(M.T, X[i])
        s = a.sum()
        if s>0: a = a/s
        A[i] = a
    return A.reshape(H,W,K)

# ---------- path finding on chemistry gradient (no circles) ----------
def edge_map(img, blur=1.0):
    g = gaussian_filter(img, blur)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)  # vertical gradient (rings transverse)
    mag = np.abs(gy)
    p1,p99 = np.percentile(mag,(1,99))
    mag = np.clip((mag-p1)/max(1e-6,p99-p1),0,1)
    return mag

def find_best_path(edge, dmax=3, lam=0.6):
    H,W = edge.shape
    dp = np.full((H,W), -1e9, np.float32)
    prev = np.full((H,W), -1, np.int32)
    dp[:,0] = edge[:,0]
    for x in range(1,W):
        for y in range(H):
            y0,y1 = max(0,y-dmax), min(H,y+dmax+1)
            slice_prev = dp[y0:y1, x-1] - lam*np.abs(np.arange(y0,y1)-y)
            j = np.argmax(slice_prev)
            dp[y,x] = edge[y,x] + slice_prev[j]
            prev[y,x] = y0+j
    y_end = int(np.argmax(dp[:, -1]))
    y_path = np.zeros(W, np.int32); y_path[-1]=y_end
    for x in range(W-1,0,-1): y_path[x-1]=prev[y_path[x],x]
    mean_edge = float(edge[y_path, np.arange(W)].mean())
    return y_path, mean_edge

def mask_path(edge, y_path, band_px=6):
    H,W = edge.shape
    xs = np.arange(W)
    for x,y in zip(xs,y_path):
        y0,y1 = max(0,y-band_px), min(H,y+band_px+1)
        edge[y0:y1,x] = 0.0
    return edge

# ---------- rings from paths ----------
def ring_widths(paths):
    if len(paths)<2: return []
    order = np.argsort([p.mean() for p in paths])
    ps = [paths[i] for i in order]
    widths=[]
    for i in range(len(ps)-1):
        dy = ps[i+1]-ps[i]
        widths.append(float(np.median(dy)))
    return widths

def overlay_paths(band1, true_paths, out_png):
    H,W = band1.shape
    vis = cv2.normalize(band1,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    xs = np.arange(W)
    for y in true_paths:
        pts = np.stack([xs, y],1).astype(np.int32)
        cv2.polylines(vis,[pts],False,(0,255,0),1)
    cv2.imwrite(out_png, vis)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Chemistry-driven ring detection (SAM + NNLS, no circles)")
    ap.add_argument("-i","--input_dir", default="prepocessing/processed_cubes")
    ap.add_argument("-o","--output_dir", default="chem_rings")
    ap.add_argument("--wavelengths", default="", help="Path to wavelengths .npy (length=B). If omitted, 400–1000 nm linear.")
    ap.add_argument("--lib_dir", default="lib", help="Folder with cellulose.csv, lignin.csv, water.csv from USGS lib (wl_nm, reflectance)")
    # preprocessing
    ap.add_argument("--deriv", action="store_true", help="Use first derivative spectra (Savitzky–Golay)")
    # seam params
    ap.add_argument("--max_paths", type=int, default=40)
    ap.add_argument("--dmax", type=int, default=3)
    ap.add_argument("--lambda_smooth", type=float, default=0.6)
    ap.add_argument("--band_px", type=int, default=6)
    # decision thresholds
    ap.add_argument("--edge_min", type=float, default=0.18, help="min mean chemistry-edge to accept a path")
    args = ap.parse_args()

    log = get_logger()
    os.makedirs(args.output_dir, exist_ok=True)
    cubes = load_cubes(args.input_dir, log)

    for key, cube in cubes.items():
        H,W,B = cube.shape
        wl = load_wavelengths(args.wavelengths, B)
        lib = load_library(args.lib_dir, names=("cellulose","lignin","water"))
        if not lib:
            log.warning("No library CSVs found in %s; continuing with only PCA map for rings.", args.lib_dir)

        # --- preprocess ---
        X = snv_per_pixel(cube)
        if args.deriv:
            X = first_derivative(X)

        # --- build references ---
        refs = {}
        for name,(wlf,rf) in lib.items():
            refs[name] = resample_spectrum(wlf, rf, wl)

        # --- SAM maps (similarity) & NNLS abundances if we have at least 2 refs ---
        sam_maps = {}
        if refs:
            for n,r in refs.items():
                sam_maps[f"sam_{n}"] = sam_similarity(X, r)

        abund = None
        if len(refs)>=2:
            M = np.stack([refs[n] for n in refs.keys()], axis=0).astype(np.float32)  # K×B
            abund = nnls_abundances(X, M)  # H×W×K (sum≈1)
        # save chem maps for debugging
        np.savez_compressed(os.path.join(args.output_dir, f"{key}_chem_maps.npz"),
                            **sam_maps, abund=abund if abund is not None else np.array([]), wavelengths=wl)

        # --- chemistry index to drive boundaries ---
        # Prefer abundances; fallback to SAM similarities; fallback to PC1.
        if abund is not None:
            # assume order of refs.keys() -> map to indices
            names = list(refs.keys())
            idx_lig = names.index("lignin") if "lignin" in names else 0
            idx_cel = names.index("cellulose") if "cellulose" in names else 0
            chem = abund[:,:,idx_lig] / (abund[:,:,idx_cel] + 1e-6)
        elif "sam_lignin" in sam_maps and "sam_cellulose" in sam_maps:
            # SAM → higher is more similar; use ratio
            chem = sam_maps["sam_lignin"] / (sam_maps["sam_cellulose"] + 1e-6)
        else:
            # final fallback: PC1 to at least get texture
            flat = X.reshape(-1,B); pc1 = PCA(1, svd_solver="randomized", random_state=42).fit_transform(flat).reshape(H,W).astype(np.float32)
            chem = pc1

        # --- edges on chemistry & path tracing (no circles) ---
        edge = edge_map(chem, blur=1.0)
        work = edge.copy()
        true_paths=[]
        for _ in range(args.max_paths):
            y_path, m = find_best_path(work, dmax=args.dmax, lam=args.lambda_smooth)
            if m < args.edge_min: break
            true_paths.append(y_path)
            work = mask_path(work, y_path, band_px=args.band_px)

        # --- ring widths & outputs ---
        widths = ring_widths(true_paths)
        overlay_paths(cube[:,:,0], true_paths, os.path.join(args.output_dir, f"{key}_annot.png"))
        pd.DataFrame({"ring_idx": np.arange(1, len(widths)+1),
                      "median_width_px": widths}).to_csv(os.path.join(args.output_dir, f"{key}_rings.csv"), index=False)
        log.info(f"{key}: kept {len(true_paths)} boundaries; median widths saved.")

if __name__=="__main__":
    main()
