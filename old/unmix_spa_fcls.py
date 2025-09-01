#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def pct_stretch(x, p1=2, p2=98):
    lo,hi = np.percentile(x, p1), np.percentile(x, p2)
    if hi<=lo: hi=lo+1e-6
    return np.clip((x-lo)/(hi-lo),0,1)

def nearest_band(wl_nm, target): return int(np.argmin(np.abs(wl_nm - target)))

def build_mask_from_exclusions(wl, excl_pairs):
    if wl is None or not excl_pairs: return np.ones(wl.shape[0], dtype=bool)
    mask = np.ones_like(wl, dtype=bool)
    for a,b in excl_pairs:
        a,b = min(a,b), max(a,b)
        mask &= ~((wl>=a) & (wl<=b))
    return mask

# ---------- SPA (Successive Projection Algorithm) ----------
def spa(X, k):
    """
    X: (B, N) matrix (bands x pixels), columns ~ spectra
    Returns indices of selected columns (endmembers) in X
    """
    norms = np.linalg.norm(X, axis=0) + 1e-12
    Xn = X / norms
    idxs = []
    R = Xn.copy()
    for _ in range(k):
        j = np.argmax(np.sum(R*R, axis=0))
        idxs.append(int(j))
        A = Xn[:, idxs]
        AtA = A.T @ A
        try: AtA_inv = np.linalg.inv(AtA)
        except np.linalg.LinAlgError: AtA_inv = np.linalg.pinv(AtA)
        P = A @ (AtA_inv @ A.T)
        R = Xn - P @ Xn
    return idxs

# ---------- FCLS via projected gradient on simplex ----------
def project_to_simplex(V):
    k, m = V.shape
    out = np.empty_like(V)
    for j in range(m):
        v = V[:, j]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u * (np.arange(1, k+1)) > (cssv - 1))[0]
        if len(rho_idx)==0:
            theta = 0.0
        else:
            rho = rho_idx[-1]
            theta = (cssv[rho] - 1.0) / float(rho + 1)
        w = np.maximum(v - theta, 0.0)
        out[:, j] = w
    return out

def fcls_batch(E, Y, iters=120, step=None, batch=5000):
    """
    E: (B, K) endmembers; Y: (B, N) pixels
    Returns A: (K, N) abundances
    """
    B,K = E.shape
    _,N = Y.shape
    ET = E.T
    G = ET @ E
    if step is None:
        s = np.linalg.svd(E, compute_uv=False)
        L = (s[0]**2) + 1e-6
        step = 1.0 / L
    A = np.full((K, N), 1.0/K, dtype=np.float32)
    for start in range(0, N, batch):
        end = min(N, start+batch)
        y = Y[:, start:end]
        c = ET @ y
        a = A[:, start:end]
        for _ in range(iters):
            grad = G @ a - c
            a = a - step * grad
            a = project_to_simplex(a)
        A[:, start:end] = a
    return A

# ---------- MNF denoising/whitening ----------
def mnf_transform(Y, H, W, n_comp=None, eps=1e-6):
    """
    Y: (N, B) pixels (row-wise). Returns Wm (B,r), Y_mnf (N,r), lambdas (r,)
    """
    B = Y.shape[1]
    C = Y.reshape(H, W, B).astype(np.float32)
    diffs = []
    if H>1: diffs.append((C[1:,:,:] - C[:-1,:,:]).reshape(-1, B))
    if W>1: diffs.append((C[:,1:,:] - C[:,:-1,:]).reshape(-1, B))
    Nmat = np.vstack(diffs) if diffs else (C.reshape(-1,B)[1:,:] - C.reshape(-1,B)[:-1,:])
    Nmat = Nmat - Nmat.mean(axis=0, keepdims=True)
    Cn = (Nmat.T @ Nmat) / max(1, (Nmat.shape[0]-1)) + eps*np.eye(B, dtype=np.float32)

    Yc = Y - Y.mean(axis=0, keepdims=True)
    Cs = (Yc.T @ Yc) / max(1, (Yc.shape[0]-1)) + eps*np.eye(B, dtype=np.float32)

    evals_n, Vn = np.linalg.eigh(Cn)
    evals_n = np.maximum(evals_n, eps)
    Cn_inv_sqrt = (Vn @ np.diag(1.0/np.sqrt(evals_n)) @ Vn.T).astype(np.float32)
    M = Cn_inv_sqrt @ (Cs @ Cn_inv_sqrt)
    L, U = np.linalg.eigh(M)
    order = np.argsort(L)[::-1]
    L = L[order]; U = U[:, order]
    Wm_full = Cn_inv_sqrt @ U
    if n_comp is None:
        r = int((L > 1.0).sum()); r = max(2, min(r, B))
    else:
        r = int(n_comp)
    Wm = Wm_full[:, :r].astype(np.float32)   # (B,r)
    Y_mnf = (Y @ Wm).astype(np.float32)      # (N,r)
    return Wm, Y_mnf, L[:r]

# ---------- TV (Chambolle) ----------
def tv_chambolle(img, weight=0.1, n_iter=40, eps=1e-6):
    m, n = img.shape
    px = np.zeros((m, n), dtype=np.float32)
    py = np.zeros((m, n), dtype=np.float32)
    u  = img.astype(np.float32).copy()
    taut = 0.125
    for _ in range(n_iter):
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        px_new = px + taut * ux
        py_new = py + taut * uy
        norm = np.maximum(1.0, np.sqrt(px_new**2 + py_new**2) / weight)
        px = px_new / norm
        py = py_new / norm
        div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
        u = img + div_p
    return np.clip(u, 0.0, 1.0)

def save_endmember_plot(wl, E, path):
    plt.figure(figsize=(6,3))
    x = wl if wl is not None else np.arange(E.shape[0])
    for i in range(E.shape[1]):
        plt.plot(x, E[:,i], label=f"E{i}")
    plt.xlabel("Wavelength (nm)" if wl is not None else "Band")
    plt.ylabel("Reflectance")
    plt.legend(ncol=min(4, E.shape[1]), fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()

def save_abundance_maps(Ab, H, W, out_dir, tv=None):
    K = Ab.shape[0]
    for k in range(K):
        ak = Ab[k].reshape(H,W)
        img = pct_stretch(ak, 2, 98)
        if tv is not None and tv>0:
            img = tv_chambolle(img, weight=tv, n_iter=50)
        plt.figure(figsize=(4,4))
        plt.imshow(img, cmap="gray")
        plt.title(f"Abundance k={k}")
        plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"abundance_k{k}.png"), dpi=180); plt.close()
    if K>=3:
        rgb = np.dstack([pct_stretch(Ab[0].reshape(H,W)),
                         pct_stretch(Ab[1].reshape(H,W)),
                         pct_stretch(Ab[2].reshape(H,W))])
        if tv is not None and tv>0:
            for c in range(3):
                rgb[...,c] = tv_chambolle(rgb[...,c], weight=tv, n_iter=40)
        plt.figure(figsize=(4,4))
        plt.imshow(rgb); plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"abundance_rgb012.png"), dpi=180); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True, help="Path to *_wl.npz")
    ap.add_argument("--k", type=int, default=3, help="Number of endmembers")
    ap.add_argument("--exclude-wl", nargs="*", type=float, default=None,
                    help="Pairs to exclude, e.g., --exclude-wl 1350 1450 1800 1950 2450 2538")
    ap.add_argument("--sample", type=int, default=8000, help="Pixels to sample for endmember extraction")
    ap.add_argument("--iters", type=int, default=120, help="FCLS iterations")
    ap.add_argument("--mnf", type=int, default=0, help="MNF components (0=auto by λ>1, negative=skip MNF)")
    ap.add_argument("--tv", type=float, default=0.0, help="TV weight for abundance maps (0 disables)")
    ap.add_argument("--out", required=True, help="Output folder")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out)

    data = np.load(args.cube)
    cube = data["block"].astype(np.float32)
    wl   = data["wavelengths_nm"] if "wavelengths_nm" in data else None
    H,W,B = cube.shape
    Y = cube.reshape(-1, B).astype(np.float32)      # (N,B)
    Y = np.nan_to_num(Y, nan=0.0, posinf=1.0, neginf=0.0)

    # Exclude noisy ranges (water absorption etc.)
    excl = []
    if args.exclude_wl and len(args.exclude_wl)%2==0 and wl is not None:
        for i in range(0, len(args.exclude_wl), 2):
            excl.append((args.exclude_wl[i], args.exclude_wl[i+1]))
    mask = build_mask_from_exclusions(wl, excl) if wl is not None else np.ones(B, dtype=bool)
    wl_use = wl[mask] if wl is not None else None
    Yb = Y[:, mask]                                 # (N,B')

    # Optional MNF: used only to choose pixels robustly
    used_mnf = False
    kept_dims = 0
    if args.mnf < 0:
        X_for_spa = Yb                               # (N,B')
    else:
        if args.mnf == 0:
            Wm, Yb_mnf, lambdas = mnf_transform(Yb, H, W, n_comp=None)
        else:
            Wm, Yb_mnf, lambdas = mnf_transform(Yb, H, W, n_comp=args.mnf)
        used_mnf = True
        kept_dims = Wm.shape[1]
        X_for_spa = Yb_mnf                           # (N,r)

    # Sample pixels for endmember extraction
    N = Yb.shape[0]
    rng = np.random.RandomState(0)
    if N > args.sample:
        idx = rng.choice(N, size=args.sample, replace=False)
    else:
        idx = np.arange(N)

    # Build matrix for SPA (bands x samples)
    X = X_for_spa[idx].T                              # (r or B', M)

    # --- SPA to pick endmember PIXELS (indices into idx) ---
    em_idx = spa(X, args.k)

    # >>> THIS IS THE IMPORTANT FIX <<<
    # Map SPA-selected columns back to the ORIGINAL band-excluded space
    chosen_pixel_idx = idx[em_idx]                    # indices into full Yb (N x B')
    E = Yb[chosen_pixel_idx, :].T.astype(np.float32)  # (B', K) true reflectance, ~[0,1]

    # Save & plot endmembers in nm
    np.save(os.path.join(out_dir, "endmembers.npy"), E)
    save_endmember_plot(wl_use, E, os.path.join(out_dir, "endmembers.png"))

    # FCLS on ALL pixels in original (band-excluded) space
    Ab = fcls_batch(E, Yb.T, iters=args.iters, step=None, batch=6000)
    np.save(os.path.join(out_dir, "abundances.npy"), Ab)

    # Save maps (+optional TV)
    save_abundance_maps(Ab, H, W, out_dir, tv=args.tv if args.tv>0 else None)

    # Log run
    log = {
        "cube": os.path.basename(args.cube),
        "H": H, "W": W, "B": B, "B_used": int(mask.sum()),
        "K": args.k,
        "excluded_ranges": excl,
        "mnf_used": used_mnf,
        "mnf_kept_dims": int(kept_dims)
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(log, f, indent=2)
    print("Unmixing done ->", out_dir)

if __name__ == "__main__":
    main()
