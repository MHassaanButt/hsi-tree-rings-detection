#!/usr/bin/env python3
import argparse, os, re, glob, sys
import numpy as np

# ----------------------------
# ENVI .hdr wavelength parsing
# ----------------------------
def parse_wavelengths_from_hdr(hdr_path):
    if hdr_path is None:
        return None, None
    with open(hdr_path, "r", errors="ignore") as f:
        txt = f.read()

    def grab_list(key):
        # Matches: key = { 1, 2, 3 }
        m = re.search(rf"{key}\s*=\s*\{{(.*?)\}}", txt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        vals = [v.strip() for v in m.group(1).replace("\n", " ").split(",") if v.strip()]
        out = []
        for v in vals:
            v = v.replace("nm", "").strip()
            try:
                out.append(float(v))
            except:
                pass
        return np.array(out, dtype=float) if len(out) else None

    def grab_scalar(key):
        # Matches: key = value  (single-line)
        m = re.search(rf"{key}\s*=\s*([^\r\n]+)", txt, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    # bands (optional but helpful)
    bands = None
    bands_str = grab_scalar("bands")
    if bands_str:
        try:
            bands = int(re.findall(r"\d+", bands_str)[0])
        except:
            bands = None

    units = (grab_scalar("wavelength units") or "").lower()

    # Try several keys without using boolean or on arrays
    wl = None
    for key in ["wavelengths", "wavelength", "band names"]:
        arr = grab_list(key)
        if arr is not None and arr.size > 0:
            wl = arr
            break

    # Fallback: linear start/step
    if wl is None:
        start = grab_scalar("wavelength start") or grab_scalar("wavelength_start")
        step  = (grab_scalar("wavelength step") or grab_scalar("wavelength_step") or
                 grab_scalar("wavelength increment") or grab_scalar("wavelength_increment"))
        if start and step and bands:
            try:
                start = float(start.replace("nm", "").strip())
                step  = float(step.replace("nm", "").strip())
                wl = start + step * np.arange(bands, dtype=float)
            except:
                wl = None

    if wl is None:
        return None, bands

    # Convert µm → nm if declared as micrometers
    if units.startswith("micro"):  # micrometers / microns
        wl = wl * 1000.0

    return wl.astype(np.float32), bands

# --------------------------------------
# Find species-matching HDRs by patterns
# --------------------------------------
def find_hdr_for_species(hdr_root, species):
    # tolerant globbing (handles dots/dashes/underscores)
    patterns = [
        os.path.join(hdr_root, f"{species}*swir*/*.hdr"),
        os.path.join(hdr_root, f"{species}*swir*.*.hdr"),
        os.path.join(hdr_root, f"{species}*swir*.hdr"),
    ]
    hits = []
    for p in patterns:
        hits += glob.glob(p)
    # also search directly under root if not found
    if not hits:
        for g in [f"{species}*swir*.hdr", f"{species}*swir*.*.hdr"]:
            hits += glob.glob(os.path.join(hdr_root, g))
    return sorted(set(hits))

# ---------------
# Main attachment
# ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes", required=True, help="Folder of *.npz (trimmed blocks)")
    ap.add_argument("--hdr-root", required=True, help="Folder containing species *.hdr (and White/dark hdrs)")
    ap.add_argument("--suffix", default="_wl", help="Suffix for output npz, default=_wl")
    ap.add_argument("--fallback-white", dest="fallback_white", default="White_swir_",
                    help="Prefix to locate White hdr as fallback (default: White_swir_)")
    args = ap.parse_args()

    # Locate a WHITE HDR fallback (often contains wavelengths)
    white_hdrs = glob.glob(os.path.join(args.hdr_root, f"{args.fallback_white}*.hdr"))
    white_hdr = white_hdrs[0] if white_hdrs else None
    white_wl, white_bands = parse_wavelengths_from_hdr(white_hdr)

    npzs = sorted(glob.glob(os.path.join(args.cubes, "*.npz")))
    if not npzs:
        print("No .npz found in", args.cubes)
        sys.exit(1)

    for npz in npzs:
        base = os.path.basename(npz)
        species = base.split("_block")[0]  # e.g., abachi_block0 -> abachi
        hdr_candidates = find_hdr_for_species(args.hdr_root, species)

        wl = None
        bands_hdr = None
        used_hdr = None

        for hp in hdr_candidates:
            w, b = parse_wavelengths_from_hdr(hp)
            if w is not None:
                wl, bands_hdr, used_hdr = w, b, hp
                break

        if wl is None and white_wl is not None:
            wl, bands_hdr, used_hdr = white_wl, white_bands, white_hdr
            print(f"[INFO] {base}: using fallback WHITE hdr -> {os.path.basename(white_hdr)}")

        # Load cube
        with np.load(npz) as data:
            if "block" in data and getattr(data["block"], "ndim", 0) == 3:
                cube = data["block"]
                extra = {k: data[k] for k in data.files if k != "block"}
            else:
                cube = None
                extra = {}
                for k in data.files:
                    arr = data[k]
                    if getattr(arr, "ndim", 0) == 3:
                        cube = arr
                    else:
                        extra[k] = arr
                if cube is None:
                    print(f"[SKIP] {base}: no 3D array found.")
                    continue

        H, W, B = cube.shape
        if wl is None:
            print(f"[WARN] {base}: no wavelengths found for species '{species}'. Wrote file without wavelengths.")
            out_path = os.path.join(args.cubes, base.replace(".npz", f"{args.suffix}.npz"))
            np.savez_compressed(out_path, block=cube, **extra)
            continue

        if bands_hdr and bands_hdr != B:
            print(f"[WARN] {base}: hdr bands={bands_hdr} != cube B={B}. Using first {min(bands_hdr, B)}.")
            m = min(bands_hdr, B)
            wl_use = wl[:m]
            cube = cube[..., :m]
        else:
            wl_use = wl

        out_path = os.path.join(args.cubes, base.replace(".npz", f"{args.suffix}.npz"))
        np.savez_compressed(out_path, block=cube, wavelengths_nm=wl_use.astype(np.float32), **extra)
        src = os.path.basename(used_hdr) if used_hdr else "unknown_hdr"
        print(f"[OK]   {base} -> {os.path.basename(out_path)}  "
              f"({H}x{W}x{cube.shape[-1]}, wl[{wl_use[0]:.2f}..{wl_use[-1]:.2f}] nm from {src})")

if __name__ == "__main__":
    main()
