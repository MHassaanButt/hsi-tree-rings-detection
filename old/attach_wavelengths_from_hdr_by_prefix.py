#!/usr/bin/env python3
import argparse, os, glob, re, sys
import numpy as np

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_envi_hdr(path):
    # Minimal ENVI .hdr parser for wavelengths + metadata
    md = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    # key = value pairs
    for m in re.finditer(r"(?im)^\s*([a-z0-9_]+)\s*=\s*(.+?)\s*$", txt):
        k = m.group(1).strip().lower()
        v = m.group(2).strip()
        md[k] = v
    # wavelengths
    wl = None
    m = re.search(r"wavelengths\s*=\s*\{([^}]*)\}", txt, flags=re.IGNORECASE|re.DOTALL)
    if m:
        numbers = re.split(r"[, \t\r\n]+", m.group(1).strip())
        wl = np.array([float(s) for s in numbers if s != ""], dtype=float)
    # bands
    bands = None
    if "bands" in md:
        try: bands = int(re.sub(r"[^\d]", "", md["bands"]))
        except: pass
    return wl, bands

def species_from_block_name(filename):
    # e.g., abachi_block0.npz -> abachi
    base = os.path.splitext(os.path.basename(filename))[0]
    sp = base.split("_block")[0]
    return sp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True, help="Dir with preprocessed *_block*.npz files (key 'block')")
    ap.add_argument("--hdr-root", required=True, help="Dir that contains the original *.hdr files")
    ap.add_argument("--out", required=True, help="Output dir for cubes with wavelengths")
    args = ap.parse_args()

    ensure_dir(args.out)
    block_files = sorted(glob.glob(os.path.join(args.blocks, "*_block*.npz")))
    if not block_files:
        print(f"No *_block*.npz in {args.blocks}", file=sys.stderr); sys.exit(1)

    for bf in block_files:
        sp = species_from_block_name(bf)
        hdrs = sorted(glob.glob(os.path.join(args.hdr_root, f"{sp}_swir_**.hdr")))
        if not hdrs:
            print(f"[WARN] {os.path.basename(bf)}: no HDR found for species prefix '{sp}'")
            continue

        # Load block and infer bands
        npz = np.load(bf, allow_pickle=True)
        if "block" not in npz:
            print(f"[WARN] {os.path.basename(bf)}: key 'block' missing; skipping.")
            npz.close(); continue
        cube = np.asarray(npz["block"])
        H, W, B = cube.shape

        # Find an HDR whose wavelengths length matches B
        wl = None; chosen_hdr = None
        for h in hdrs:
            wl_h, bands_h = parse_envi_hdr(h)
            if wl_h is not None and wl_h.size == B:
                wl = wl_h.astype(float); chosen_hdr = h; break
            if wl is None and bands_h == B:
                # fallback if hdr has bands but no explicit wavelengths
                wl = None; chosen_hdr = h  # keep, maybe later we build linspace
        if chosen_hdr is None:
            print(f"[WARN] {os.path.basename(bf)}: no HDR with matching bands={B}")
            npz.close(); continue

        # If still no wavelengths array, but we know bands, try to infer from fwhm or start/end (not in minimal parser)
        if wl is None:
            print(f"[WARN] {os.path.basename(bf)}: HDR '{os.path.basename(chosen_hdr)}' lacks wavelength list; "
                  f"you may provide start/end nm or a CSV to build one.")
            npz.close(); continue

        # Save out
        out_path = os.path.join(args.out, os.path.basename(bf))
        out_dict = {k: npz[k] for k in npz.files}
        out_dict["wavelengths"] = wl
        np.savez_compressed(out_path, **out_dict)
        npz.close()
        print(f"Wrote {out_path} with wavelengths[{B}] {wl[0]:.1f}…{wl[-1]:.1f} nm (HDR={os.path.basename(chosen_hdr)})")

if __name__ == "__main__":
    main()
