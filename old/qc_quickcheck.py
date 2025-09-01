#!/usr/bin/env python3
import argparse, os, glob, sys
import numpy as np

WL_CANDIDATE_KEYS = ["wavelengths", "wavelength", "bands", "lambda", "lambdas", "wl"]

def guess_cube_key(npz):
    keys = list(npz.keys())
    # Prefer a key literally named 'cube'
    if "cube" in keys and np.asarray(npz["cube"]).ndim == 3:
        return "cube"
    # Else choose the largest 3D array
    candidates = [(k, np.asarray(npz[k])) for k in keys if np.asarray(npz[k]).ndim == 3]
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[1].size, reverse=True)
    return candidates[0][0]

def find_wavelength_key(npz, bands=None):
    keys = list(npz.keys())
    for k in WL_CANDIDATE_KEYS:
        if k in keys:
            arr = np.asarray(npz[k])
            if arr.ndim == 1:
                if bands is None or arr.shape[0] == bands:
                    return k
    # fallback: any 1D array with length == bands
    if bands is not None:
        for k in keys:
            arr = np.asarray(npz[k])
            if arr.ndim == 1 and arr.shape[0] == bands:
                return k
    return None

def infer_orientation(shape):
    # Try to guess if shape is (H, W, B) or (B, H, W)
    if len(shape) != 3:
        return "unknown"
    h, w, c = shape
    if 40 <= c <= 1500:
        return "(H, W, B)"
    b, h2, w2 = shape
    if 40 <= b <= 1500:
        return "(B, H, W)"
    # fallback heuristic: assume last dim is bands if small-ish
    return "(H, W, B)" if c < max(h, w) else "(B, H, W)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory with .npz cubes")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.npz")))
    if not files:
        print(f"No .npz files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} .npz files.")
    for f in files:
        print("="*80)
        print(f"FILE: {os.path.basename(f)}")
        try:
            npz = np.load(f, allow_pickle=True)
        except Exception as e:
            print(f"  ERROR: cannot open ({e})")
            continue

        keys = list(npz.keys())
        print(f"  Keys: {keys}")

        cube_key = guess_cube_key(npz)
        if cube_key is None:
            print("  WARNING: No 3D array found (cube not detected).")
            continue

        cube = np.asarray(npz[cube_key])
        shape = cube.shape
        dtype = cube.dtype
        orient = infer_orientation(shape)
        bands = shape[2] if orient == "(H, W, B)" else shape[0]
        print(f"  Cube key: {cube_key} | shape={shape} | dtype={dtype} | orientation guess={orient} | bands={bands}")

        wl_key = find_wavelength_key(npz, bands=bands)
        if wl_key:
            wl = np.asarray(npz[wl_key]).astype(float)
            monotonic = np.all(np.diff(wl) > 0)
            wl_min, wl_max = float(np.nanmin(wl)), float(np.nanmax(wl))
            unit = "nm" if wl_max > 100 else "um?"  # rough guess
            print(f"  Wavelength key: {wl_key} | len={wl.size} | min={wl_min:.3f} | max={wl_max:.3f} | monotonic={monotonic} | unit_guess={unit}")
        else:
            print("  WARNING: No wavelength vector found.")

        # Quick value peek (sample)
        try:
            # sample a small random subset without loading full mem if possible
            rng = np.random.default_rng(0)
            if orient == "(H, W, B)":
                H, W, B = shape
                ys = rng.integers(0, H, size=min(200, H))
                xs = rng.integers(0, W, size=min(200, W))
                sample = cube[ys[:, None], xs[None, :], :].reshape(-1, B)
            elif orient == "(B, H, W)":
                B, H, W = shape
                ys = rng.integers(0, H, size=min(200, H))
                xs = rng.integers(0, W, size=min(200, W))
                sample = cube[:, ys[:, None], xs[None, :]].reshape(B, -1).T
            else:
                sample = cube.reshape(-1, cube.shape[-1])[:1000]
            smin, smax = np.nanmin(sample), np.nanmax(sample)
            nneg = np.sum(sample < 0)
            ngt1 = np.sum(sample > 1.0) if np.issubdtype(dtype, np.floating) else "N/A"
            print(f"  Sample value range: [{smin:.6g}, {smax:.6g}] | negatives={nneg} | >1.0={ngt1}")
        except Exception as e:
            print(f"  Could not sample values: {e}")

        npz.close()

if __name__ == "__main__":
    main()
