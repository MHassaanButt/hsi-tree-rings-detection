#!/usr/bin/env python3
import argparse, os, glob, re
import numpy as np

def read_text(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdr-root", required=True)
    args = ap.parse_args()

    hdrs = sorted(glob.glob(os.path.join(args.hdr_root, "*.hdr")))
    if not hdrs:
        print("No .hdr files found.")
        return

    for h in hdrs:
        txt = read_text(h)
        bands = re.search(r"(?im)^\s*bands\s*=\s*([0-9]+)", txt)
        inter = re.search(r"(?im)^\s*interleave\s*=\s*([a-z0-9]+)", txt)
        dtype = re.search(r"(?im)^\s*data\s*type\s*=\s*([0-9]+)", txt)
        print("="*80)
        print(os.path.basename(h))
        print(f"  bands={bands.group(1) if bands else '?'}  interleave={inter.group(1) if inter else '?'}  dtype={dtype.group(1) if dtype else '?'}")
        # Dump any lines mentioning wave
        for m in re.finditer(r"(?im)^\s*.*wave.*$", txt):
            print("  " + m.group(0).strip())

        # Check if wavelengths block exists & its length
        m = re.search(r"wavelengths?\s*=\s*\{([^}]*)\}", txt, flags=re.IGNORECASE|re.DOTALL)
        if m:
            nums = re.split(r"[, \t\r\n]+", m.group(1).strip())
            nums = [s for s in nums if s]
            print(f"  -> wavelengths block found: count={len(nums)}, first={nums[0]}, last={nums[-1]}")
        else:
            print("  -> NO explicit wavelengths block.")

if __name__ == "__main__":
    main()
