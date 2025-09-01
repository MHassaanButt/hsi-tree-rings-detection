#!/usr/bin/env python3
import argparse, os, glob, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def pct_stretch(img, p_low=2, p_high=98):
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo: hi = lo + 1e-6
    x = (img - lo) / (hi - lo)
    return np.clip(x, 0, 1)

def nearest_band(wl_nm, target):
    idx = np.argmin(np.abs(wl_nm - target))
    return int(idx)

def simple_hist(values, path):
    plt.figure(figsize=(5,3))
    plt.hist(values.ravel(), bins=128)
    plt.xlabel("Value"); plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(path, dpi=180); plt.close()

def plot_mean_std(wl, mean_spec, std_spec, path):
    plt.figure(figsize=(6,3))
    if wl is None:
        x = np.arange(len(mean_spec))
        plt.plot(x, mean_spec, label="mean")
        plt.fill_between(x, mean_spec-std_spec, mean_spec+std_spec, alpha=0.3, label="±std")
        plt.xlabel("Band index")
    else:
        plt.plot(wl, mean_spec, label="mean")
        plt.fill_between(wl, mean_spec-std_spec, mean_spec+std_spec, alpha=0.3, label="±std")
        plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()

def plot_band_variance(wl, var_per_band, path, shade_ranges=None):
    plt.figure(figsize=(6,3))
    x = wl if wl is not None else np.arange(len(var_per_band))
    plt.plot(x, var_per_band)
    if wl is not None and shade_ranges:
        for a,b in shade_ranges:
            aa, bb = min(a,b), max(a,b)
            plt.axvspan(aa, bb, alpha=0.15, hatch="//", label=f"{aa}-{bb} nm")
    plt.xlabel("Wavelength (nm)" if wl is not None else "Band index")
    plt.ylabel("Variance across pixels")
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()

def make_quicklook_rgb(cube, wl, out_png, rgb_nm=None, p_low=2, p_high=98):
    H,W,B = cube.shape
    if wl is not None and rgb_nm:
        bands = [nearest_band(wl, t) for t in rgb_nm]
    elif wl is not None:
        # Safe SWIR default avoiding strongest water dips: ~1650/1200/1000 nm
        bands = [nearest_band(wl, t) for t in [1650, 1200, 1000]]
    else:
        bands = [int(B*0.8), int(B*0.5), int(B*0.2)]
    R = pct_stretch(cube[..., bands[0]], p_low, p_high)
    G = pct_stretch(cube[..., bands[1]], p_low, p_high)
    Bc= pct_stretch(cube[..., bands[2]], p_low, p_high)
    rgb = np.dstack([R,G,Bc])
    plt.figure(figsize=(4,4))
    plt.imshow(rgb)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(out_png, dpi=180); plt.close()
    return bands

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder with *.npz cubes")
    ap.add_argument("--pattern", default="*_wl.npz", help="Glob pattern (default: *_wl.npz)")
    ap.add_argument("--output", required=True, help="Output folder for QC")
    ap.add_argument("--max-sample-pixels", type=int, default=100000, help="Sampling cap for spectra/hist")
    ap.add_argument("--rgb-nm", nargs=3, type=float, default=None, help="e.g., --rgb-nm 1650 1200 1000")
    ap.add_argument("--rgb-pct", nargs=2, type=float, default=[2,98], help="Low/High percentiles")
    args = ap.parse_args()

    out_root = ensure_dir(args.output)
    summary_rows = []
    notes_all = {}

    files = sorted(glob.glob(os.path.join(args.input, args.pattern)))
    if not files:
        print("No files matched pattern."); return

    for fp in files:
        base = os.path.basename(fp)
        name = os.path.splitext(base)[0]
        out_dir = ensure_dir(os.path.join(out_root, name))
        notes = []

        with np.load(fp) as data:
            if "block" not in data:
                notes.append("Missing 'block' key.")
                continue
            cube = data["block"]
            wl = data["wavelengths_nm"] if "wavelengths_nm" in data else None

        if cube.ndim != 3:
            notes.append(f"Block not 3D (ndim={cube.ndim})."); continue

        H,W,B = cube.shape
        stats = {
            "file": base, "H":H, "W":W, "B":B,
            "dtype": str(cube.dtype),
            "min": float(np.nanmin(cube)),
            "max": float(np.nanmax(cube)),
            "mean": float(np.nanmean(cube)),
            "std": float(np.nanstd(cube)),
            "nan": int(np.isnan(cube).sum()),
            "inf": int(np.isinf(cube).sum()),
            "neg": int((cube<0).sum()),
            "gt1": int((cube>1).sum()),
            "wl_min": float(wl.min()) if wl is not None else None,
            "wl_max": float(wl.max()) if wl is not None else None,
            "has_wl": int(wl is not None),
        }
        if stats["nan"]>0: notes.append("Has NaNs.")
        if stats["inf"]>0: notes.append("Has Infs.")
        if stats["neg"]>0: notes.append("Has negatives.")
        if stats["gt1"]>0: notes.append(">1 reflectance values.")

        # Band variance across pixels
        X = cube.reshape(-1, B)  # N x B
        var_per_band = np.var(X, axis=0)
        plot_band_variance(wl, var_per_band, os.path.join(out_dir, "band_variance.png"),
                           shade_ranges=[(1350,1450),(1800,1950)] if wl is not None else None)

        # Sampling for spectra/hist
        N = X.shape[0]
        if N > args.max_sample_pixels:
            idx = np.random.RandomState(0).choice(N, size=args.max_sample_pixels, replace=False)
            Xs = X[idx]
        else:
            Xs = X

        # Mean ± std spectrum
        mean_spec = Xs.mean(axis=0)
        std_spec  = Xs.std(axis=0)
        plot_mean_std(wl, mean_spec, std_spec, os.path.join(out_dir, "mean_std_spectrum.png"))

        # Histogram
        simple_hist(Xs, os.path.join(out_dir, "value_histogram.png"))

        # Quicklook RGB
        bands = make_quicklook_rgb(cube, wl, os.path.join(out_dir, "quicklook_rgb.png"),
                                   rgb_nm=args.rgb_nm, p_low=args.rgb_pct[0], p_high=args.rgb_pct[1])

        # Write notes + per-file summary JSON
        with open(os.path.join(out_dir, "notes.txt"), "w") as f:
            if wl is None:
                notes.append("No wavelengths_nm in file — plots use band indices.")
            for n in notes: f.write(n+"\n")

        info = {
            **stats,
            "rgb_bands": ",".join(map(str,bands)),
            "out_dir": out_dir
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(info, f, indent=2)

        summary_rows.append([
            base, H, W, B, stats["dtype"], f"{stats['min']:.4f}", f"{stats['max']:.4f}",
            stats["nan"], stats["inf"], stats["neg"], stats["gt1"],
            f"{stats['wl_min']:.2f}" if stats["wl_min"] is not None else "",
            f"{stats['wl_max']:.2f}" if stats["wl_max"] is not None else "",
            info["rgb_bands"]
        ])

    # Global summary CSV
    with open(os.path.join(out_root, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file","H","W","B","dtype","min","max","NaN","Inf","neg","gt1","wl_min","wl_max","rgb_bands"])
        w.writerows(summary_rows)
    print("QC done ->", out_root)

if __name__ == "__main__":
    main()
