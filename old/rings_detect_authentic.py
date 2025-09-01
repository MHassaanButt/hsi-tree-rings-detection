#!/usr/bin/env python3
# End-to-end tree-ring detection from hyperspectral cubes:
# MNF-like whitening -> SPA endmembers -> FCLS abundances (+TV-like smoothing)
# -> ring-contrast map -> multiscale ridge tracing (curved lines) -> chemistry report

import argparse, os, json, csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional SciPy (nice to have; code falls back if missing)
try:
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import nnls

    SCIPY_OK = True
except Exception:
    gaussian_filter = None
    nnls = None
    SCIPY_OK = False


# ------------------------- utilities -------------------------


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def im01(x, p1=2, p2=98):
    lo, hi = np.percentile(x, p1), np.percentile(x, p2)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def load_cube(npz_path):
    with np.load(npz_path) as d:
        # try common keys
        for k in ["block", "cube", "data"]:
            if k in d:
                C = d[k].astype(np.float32)  # H,W,B
                break
        else:
            raise ValueError(f"No cube array found in {npz_path}")
        wlk = None
        for k in ["wavelengths_nm", "wavelengths", "lambda", "wl"]:
            if k in d:
                wlk = k
                break
        wl = d[wlk].astype(np.float32) if wlk else None
    return C, wl


def nearest_band(wl, nm):
    return int(np.argmin(np.abs(wl - float(nm))))


def cc_label(bin_img):
    H, W = bin_img.shape
    lab = np.zeros((H, W), np.int32)
    cur = 0
    areas = []
    for y in range(H):
        for x in range(W):
            if not bin_img[y, x] or lab[y, x]:
                continue
            cur += 1
            st = [(y, x)]
            lab[y, x] = cur
            area = 0
            while st:
                yy, xx = st.pop()
                area += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = yy + dy, xx + dx
                    if (
                        0 <= ny < H
                        and 0 <= nx < W
                        and bin_img[ny, nx]
                        and lab[ny, nx] == 0
                    ):
                        lab[ny, nx] = cur
                        st.append((ny, nx))
            areas.append(area)
    return lab, areas


def wood_mask(C, wl, nm=1200, pthr=10, border_frac=0.04):
    H, W, B = C.shape
    band = nearest_band(wl, nm) if wl is not None else int(B * 0.5)
    img = C[..., band]
    x = im01(img, 1, 99)
    thr = np.percentile(x, pthr)
    M = x > thr
    lab, areas = cc_label(M)
    if not areas:
        M = np.ones((H, W), bool)
    else:
        M = lab == (1 + int(np.argmax(areas)))
    # interior crop
    by, bx = int(max(2, round(border_frac * H))), int(max(2, round(border_frac * W)))
    Min = np.zeros_like(M)
    Min[by : H - by, bx : W - bx] = True
    return M & Min


# --------------------- MNF-like whitening ---------------------


def mnf_whiten(X, H, W, B, var_keep=0.999):
    # noise via band differences
    D = X.reshape(H, W, B)
    E = D[:, :, 1:] - D[:, :, :-1]  # H x W x (B-1)
    E = E.reshape(-1, B - 1)
    mu = E.mean(axis=0, keepdims=True)
    Cn = np.cov((E - mu).T) + 1e-8 * np.eye(B - 1, dtype=np.float32)
    # pad to B
    Cn_full = np.eye(B, dtype=np.float32)
    Cn_full[:-1, :-1] = Cn
    # whitening matrix
    w, V = np.linalg.eigh(Cn_full)
    Wn = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-8))) @ V.T
    Xw = X @ Wn.T

    # PCA on whitened data
    muX = Xw.mean(axis=0, keepdims=True)
    Cx = np.cov((Xw - muX).T)
    s, U = np.linalg.eigh(Cx)
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]
    cs = np.cumsum(s) / (np.sum(s) + 1e-12)
    r = int(np.searchsorted(cs, var_keep) + 1)
    Uw = U[:, :r]
    Y = (Xw - muX) @ Uw  # N x r
    return (
        Y.astype(np.float32),
        Wn.astype(np.float32),
        muX.astype(np.float32),
        Uw.astype(np.float32),
    )


# --------------------- SPA endmember picker -------------------


def spa(Y, R):
    # Y: N x d (whitened+PCA); returns indices in Y (and hence in X)
    N, d = Y.shape
    R = int(R)
    idx = []
    RY = Y.copy()
    for _ in range(R):
        norms = np.sum(RY * RY, axis=1)
        j = int(np.argmax(norms))
        idx.append(j)
        v = Y[j : j + 1, :]  # 1 x d
        denom = float(np.dot(v.ravel(), v.ravel())) + 1e-12
        proj = (RY @ v.T) / denom  # N x 1
        RY = RY - proj * v  # remove selected direction
    return np.array(idx, dtype=int)


def endmembers_from_idx(X, idx, B):
    E = X[idx, :B]  # rows of original reflectance
    return E.T.astype(np.float32)  # B x K


def dedupe_endmembers(E, cos_thr=0.995):
    K = E.shape[1]
    keep = []
    for j in range(K):
        if not keep:
            keep.append(j)
            continue
        ok = True
        for i in keep:
            num = float(np.dot(E[:, i], E[:, j]))
            den = np.linalg.norm(E[:, i]) * np.linalg.norm(E[:, j]) + 1e-12
            if den > 0 and (num / den) >= cos_thr:
                ok = False
                break
        if ok:
            keep.append(j)
    return E[:, keep], keep


# --------------------- Robust FCLS (ANC+ASC) ------------------


def fcls_image(E, X, ridge=1e-4, use_nnls=True):
    """
    Stable FCLS: solve (E^T E + ridge I)a = E^T x, clip >=0, normalize to sum=1.
    E: (B,K), X: (N,B). Returns A: (N,K)
    """
    B, K = E.shape
    N = X.shape[0]
    A = np.zeros((N, K), np.float32)

    AtA = E.T @ E + ridge * np.eye(K, dtype=np.float32)
    try:
        AtA_inv = np.linalg.inv(AtA)
    except np.linalg.LinAlgError:
        AtA_inv = np.linalg.pinv(AtA)

    Et = E.T
    for i in range(N):
        a = AtA_inv @ (Et @ X[i])
        a = np.maximum(0.0, a)
        s = a.sum()
        if s > 0:
            A[i] = a / s
        elif use_nnls:
            if nnls is not None:
                a_nnls, _ = nnls(E, X[i])
                s = a_nnls.sum()
                A[i] = (a_nnls / s) if s > 0 else a_nnls
            else:
                a = np.maximum(0.0, np.linalg.pinv(E) @ X[i])
                s = a.sum()
                A[i] = (a / s) if s > 0 else a
        else:
            A[i] = a
    return A


# --------------- TV-like smoothing (edge-aware) ---------------


def smooth_tv_like(Aimg, sigma=1.2, iters=2):
    out = Aimg.copy()
    for _ in range(iters):
        if SCIPY_OK and gaussian_filter is not None:
            out = gaussian_filter(
                out, sigma=(0.6, sigma)
            )  # light along rows, more across
        out = np.clip(out, 0, 1)
    return out


# --------------- Frangi-like ridge & multiscale ----------------


def frangi_like(R, sigma=2.5, beta=0.5, c=0.5):
    if SCIPY_OK and gaussian_filter is not None:
        # second derivatives by Gaussian (order via finite diffs trick)
        # approximate Hessian via smoothed second derivatives
        from scipy.ndimage import gaussian_filter as gf

        dxx = gf(R, sigma=sigma, order=(0, 2))
        dyy = gf(R, sigma=sigma, order=(2, 0))
        dxy = gf(R, sigma=sigma, order=(1, 1))
    else:
        # crude finite differences fallback
        dxx = np.zeros_like(R)
        dxx[:, 1:-1] = R[:, 2:] - 2 * R[:, 1:-1] + R[:, :-2]
        dyy = np.zeros_like(R)
        dyy[1:-1, :] = R[2:, :] - 2 * R[1:-1, :] + R[:-2, :]
        dxy = np.zeros_like(R)

    tmp = np.sqrt((dxx - dyy) ** 2 + 4.0 * (dxy**2))
    l1 = 0.5 * ((dxx + dyy) - tmp)
    l2 = 0.5 * ((dxx + dyy) + tmp)
    swap = np.abs(l1) < np.abs(l2)
    l1[swap], l2[swap] = l2[swap], l1[swap]
    RA = np.abs(l2) / (np.abs(l1) + 1e-8)
    S = np.sqrt(l1 * l1 + l2 * l2)
    V = np.exp(-(RA**2) / (2 * beta * beta)) * (1.0 - np.exp(-(S**2) / (2 * c * c)))
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    return V.astype(np.float32)


def multiscale_ridge(R, sigmas=(1.5, 2.5, 3.5)):
    Vs = [frangi_like(R, s) for s in sigmas]
    return np.max(np.stack(Vs, 0), 0)


# ------------------- polyline tracing -------------------------


def trace_polylines(V, mask, min_span_frac=0.65, thr_ptile=92, border_frac=0.04):
    H, W = V.shape
    by, bx = int(max(2, round(border_frac * H))), int(max(2, round(border_frac * W)))
    M = mask.copy()
    M[:by, :] = False
    M[-by:, :] = False
    M[:, :bx] = False
    M[:, -bx:] = False

    thr = np.percentile(V[M], thr_ptile) if np.any(M) else np.percentile(V, thr_ptile)
    B = (V >= thr) & M
    lab, areas = cc_label(B)
    rings = []
    for lbl, area in enumerate(areas, start=1):
        ys, xs = np.where(lab == lbl)
        if xs.size < 10:
            continue
        xmin, xmax = xs.min(), xs.max()
        span = (xmax - xmin + 1) / float(W)
        if span < min_span_frac:
            continue
        xs_u = np.unique(xs)
        poly = []
        for x in xs_u:
            ymed = int(np.median(ys[xs == x]))
            poly.append((int(x), ymed))
        rings.append(poly)
    return rings, B


# ---------------- continuum removal (convex hull) --------------


def continuum_remove(wl, spec):
    """
    Upper convex hull continuum removal (USGS-style).
    Returns CR spectrum and continuum.
    """
    x = wl.astype(np.float64)
    y = spec.astype(np.float64)
    pts = np.stack([x, y], 1)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    upper = []
    for p in pts:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) >= 0:
            upper.pop()
        upper.append(p)
    hull_x = np.array([p[0] for p in upper])
    hull_y = np.array([p[1] for p in upper])

    cont = np.interp(x, hull_x, hull_y)
    cont = np.maximum(cont, y)  # ensure upper
    cr = np.divide(y, cont + 1e-12)
    return cr.astype(np.float32), cont.astype(np.float32)


def band_depth(wl, cr_spec, center, halfwin=20):
    m = (wl >= center - halfwin) & (wl <= center + halfwin)
    if not np.any(m):
        return np.nan
    return float(1.0 - np.min(cr_spec[m]))


# ----------------------------- main ---------------------------


def main():
    ap = argparse.ArgumentParser(
        "End-to-end HSI ring detection with robust unmixing + chemistry"
    )
    ap.add_argument("--cube", required=True, help="path to *_wl.npz")
    ap.add_argument(
        "--k", type=int, default=3, help="# endmembers (3 = late, early, background)"
    )
    ap.add_argument(
        "--sigmas", nargs="*", type=float, default=[1.5, 2.5, 3.5], help="ridge scales"
    )
    ap.add_argument(
        "--thr-ptile", type=float, default=92.0, help="ridge percentile threshold"
    )
    ap.add_argument(
        "--min-span-frac",
        type=float,
        default=0.65,
        help="min span across width to keep polyline",
    )
    ap.add_argument(
        "--border-margin-frac",
        type=float,
        default=0.04,
        help="ignore edges by this margin",
    )
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args()

    out = ensure_dir(args.out)

    # Load + mask
    C, wl = load_cube(args.cube)  # H,W,B
    H, W, B = C.shape
    M = wood_mask(C, wl, nm=1200, pthr=10, border_frac=args.border_margin_frac)

    # Stack and whiten
    X = C.reshape(-1, B)
    Y, Wn, muX, Uw = mnf_whiten(X, H, W, B, var_keep=0.999)

    # Endmember selection (SPA on whitened space) -> reflectance E
    em_idx = spa(Y, args.k)
    E = endmembers_from_idx(X, em_idx, B)  # B x K
    E = np.clip(E, 0, 1)
    E, kept = dedupe_endmembers(E, cos_thr=0.995)
    K = E.shape[1]
    if K < max(2, args.k // 2):
        # fall back to top K columns from SPA if dedupe was too aggressive
        K = max(2, args.k)
        E = endmembers_from_idx(X, em_idx[:K], B)
        E = np.clip(E, 0, 1)

    # Abundances (robust FCLS)
    A = fcls_image(E, X, ridge=1e-4, use_nnls=True)  # N x K
    Aimg = A.reshape(H, W, E.shape[1])
    # Smooth abundances and renormalize
    for r in range(E.shape[1]):
        Aimg[..., r] = smooth_tv_like(Aimg[..., r], sigma=1.2, iters=2)
    S = np.sum(Aimg, axis=2, keepdims=True) + 1e-8
    Aimg = Aimg / S
    A = Aimg.reshape(-1, E.shape[1])

    # Early vs late: choose pair with max SWIR distance (avoid strong water bands)
    mean_specs = []
    for r in range(E.shape[1]):
        wts = A[:, r][:, None]
        m = (wts * X).sum(0) / (wts.sum() + 1e-8)
        mean_specs.append(m)
    mean_specs = np.stack(mean_specs, 0)  # K x B

    mask_sw = np.ones(B, bool)
    if wl is not None:
        for a, b in [(1350, 1450), (1800, 1950), (2450, wl[-1])]:
            mask_sw &= ~((wl >= a) & (wl <= b))
    dists = {}
    for i in range(E.shape[1]):
        for j in range(i + 1, E.shape[1]):
            dists[(i, j)] = np.linalg.norm((mean_specs[i] - mean_specs[j])[mask_sw])
    (i_late, i_early) = max(dists, key=dists.get)

    # Ring-contrast map with directional detrend
    R = (Aimg[..., i_late] - Aimg[..., i_early]).astype(np.float32)
    if SCIPY_OK and gaussian_filter is not None:
        trend = gaussian_filter(R, sigma=(12.0, 0.8))  # heavy along rows, light across
        R = R - trend
        R = gaussian_filter(R, sigma=(0.8, 1.6))
    else:
        R = R - R.mean()

    # suppress border firing
    by = int(max(2, round(args.border_margin_frac * H)))
    bx = int(max(2, round(args.border_margin_frac * W)))
    M[:by, :] = False
    M[-by:, :] = False
    M[:, :bx] = False
    M[:, -bx:] = False
    R[~M] = np.median(R[M])

    # Ridge & tracing
    V = multiscale_ridge(R, sigmas=tuple(args.sigmas))
    rings, Bbin = trace_polylines(
        V,
        M,
        min_span_frac=args.min_span_frac,
        thr_ptile=args.thr_ptile,
        border_frac=args.border_margin_frac,
    )

    # ---------- SAVE: overlays ----------
    plt.figure(figsize=(4, 6))
    plt.imshow(im01(R), cmap="gray")
    for poly in rings:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        plt.plot(xs, ys, c="c", lw=1.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "rings_on_contrast.png"), dpi=220)
    plt.close()

    # Endmember mean spectra (reflectance)
    if wl is not None:
        plt.figure(figsize=(7, 3))
        for r in range(mean_specs.shape[0]):
            plt.plot(wl, mean_specs[r], label=f"E{r}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out, "endmembers_mean.png"), dpi=200)
        plt.close()

    np.save(os.path.join(out, "E.npy"), E)
    np.save(os.path.join(out, "A.npy"), Aimg)

    # Polylines & count
    with open(os.path.join(out, "rings_polylines.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ring_id", "x", "y"])
        for rid, poly in enumerate(rings):
            for x, y in poly:
                w.writerow([rid, x, y])
    with open(os.path.join(out, "rings_count.txt"), "w") as f:
        f.write(str(len(rings)) + "\n")

    # ---------- Chemistry (convex-hull CR, positive band depths) ----------
    if wl is not None:
        w_l = A[:, i_late][:, None]
        w_e = A[:, i_early][:, None]
        mL = (w_l * X).sum(0) / (w_l.sum() + 1e-8)
        mE = (w_e * X).sum(0) / (w_e.sum() + 1e-8)
        crL, cL = continuum_remove(wl, mL)
        crE, cE = continuum_remove(wl, mE)

        # band depths (positive)
        d1200_L = band_depth(wl, crL, 1200, 25)
        d1200_E = band_depth(wl, crE, 1200, 25)
        d1730_L = band_depth(wl, crL, 1730, 25)
        d1730_E = band_depth(wl, crE, 1730, 25)
        w2100 = (wl >= 2100) & (wl <= 2330)
        d21_L = float(1 - np.min(crL[w2100])) if np.any(w2100) else np.nan
        d21_E = float(1 - np.min(crE[w2100])) if np.any(w2100) else np.nan

        with open(os.path.join(out, "chem_banddepths.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature", "latewood_depth", "earlywood_depth"])
            w.writerow(["~1200 nm", d1200_L, d1200_E])
            w.writerow(["~1730 nm", d1730_L, d1730_E])
            w.writerow(["~2100–2330 nm (min)", d21_L, d21_E])

        plt.figure(figsize=(7, 3))
        plt.plot(wl, crE, label="earlywood (CR)")
        plt.plot(wl, crL, label="latewood (CR)")
        for nm in [1200, 1730, 2100, 2330]:
            plt.axvline(nm, ls="--", lw=0.6)
        plt.legend()
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Continuum-removed")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "chem_continuum_removed.png"), dpi=200)
        plt.close()

    # ---------- run metadata ----------
    info = {
        "cube": os.path.basename(args.cube),
        "shape": [int(H), int(W), int(B)],
        "K": int(E.shape[1]),
        "late_idx": int(i_late),
        "early_idx": int(i_early),
        "sigmas": list(args.sigmas),
        "thr_ptile": float(args.thr_ptile),
        "min_span_frac": float(args.min_span_frac),
        "border_margin_frac": float(args.border_margin_frac),
        "rings_detected": int(len(rings)),
        "scipy": bool(SCIPY_OK),
    }
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[OK] {len(rings)} rings | outputs -> {out}")


if __name__ == "__main__":
    main()
