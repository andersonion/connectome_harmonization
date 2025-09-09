#!/usr/bin/env python3
"""
Harmonize structural (DWI) or functional (fMRI) connectomes with location-only, anchored adjustment,
and generate QC (site AUC, PCA, subject panels).

Inputs:
  - Wide features CSV: first column is subject_id, remaining columns are edge features (upper-tri vec).
  - Covariates CSV: must contain columns [subject_id, SITE, AGE].

Model:
  Y = intercept + f(AGE) + sum_{site!=ref} 1{SITE=site} * beta_site  +  eps
  We fit ridge-regularized OLS in transform space (log1p for DWI; Fisher z for fMRI),
  then subtract the subject's site effect to map all subjects to REF site. No variance scaling.

Usage (DWI example):
  python harmonize_connectomes.py \
    --features /mnt/.../all_features_numeric.csv \
    --covars   /mnt/.../all_covars_aligned.csv \
    --out-dir  /mnt/.../harmonized_full_dwi_location_only \
    --ref-site "ADDecode:1" \
    --modality dwi \
    --nodes 84 \
    --min-per-site 20 \
    --ridge 1e-3 \
    --qc --qc-topk 1500 --qc-folds 5 --qc-min-per-site 5 --qc-panels 6 --qc-pca-sample 500

For fMRI, set --modality fmri (uses Fisher z / tanh).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Optional QC deps
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# -------------------------
# Helpers
# -------------------------

def infer_N_from_m(m: int) -> int:
    N = int((1 + (1 + 8*m)**0.5) // 2)
    if N*(N-1)//2 != m:
        raise ValueError(f"Cannot infer N from m={m} (got N={N}, expected {N*(N-1)//2}).")
    return N

def pre_transform(X: np.ndarray, modality: str) -> np.ndarray:
    if modality == "dwi":
        return np.log1p(np.maximum(X, 0.0))
    elif modality == "fmri":
        Xc = np.clip(X, -0.999999, 0.999999)
        return np.arctanh(Xc)
    else:
        raise ValueError("modality must be 'dwi' or 'fmri'")

def inverse_transform(Y: np.ndarray, modality: str) -> np.ndarray:
    if modality == "dwi":
        X = np.expm1(Y)
        np.clip(X, 0.0, None, out=X)
        return X
    elif modality == "fmri":
        return np.tanh(Y)
    else:
        raise ValueError("modality must be 'dwi' or 'fmri'")

def design_matrix(age: np.ndarray, sites: pd.Series, ref_site: str):
    # Ensure ref_site exists; if not, pick first category and warn upstream
    cats = [ref_site] + sorted([s for s in pd.unique(sites) if s != ref_site])
    sites_cat = pd.Categorical(sites, categories=cats, ordered=True)
    S = pd.get_dummies(sites_cat, drop_first=True)  # site dummies excluding REF
    S_cols = list(S.columns)

    age_mu, age_sd = np.nanmean(age), np.nanstd(age)
    age_z = np.zeros_like(age) if (not np.isfinite(age_sd) or age_sd == 0) else (age - age_mu) / age_sd

    Xdesign = np.column_stack([np.ones(len(age)), age_z, S.to_numpy(dtype=np.float64)])  # n×p
    return Xdesign, S, S_cols

def fit_location_only(Y: np.ndarray, Xdesign: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge-regularized OLS in transform space: returns beta (p×m)."""
    n, m = Y.shape
    p = Xdesign.shape[1]
    XtX = Xdesign.T @ Xdesign
    if ridge > 0:
        XtX = XtX + ridge * np.eye(p)
    XtY = Xdesign.T @ Y
    beta = np.linalg.solve(XtX, XtY)  # p×m
    return beta

def anchor_to_ref(Y: np.ndarray, beta: np.ndarray, S: pd.DataFrame) -> np.ndarray:
    """Subtract site dummy effects to map all to REF in transform space."""
    B_site = beta[2:, :]  # rows correspond to site dummies (excluding intercept, age)
    S_mat = S.to_numpy(dtype=np.float64)
    site_eff = S_mat @ B_site  # n×m
    return Y - site_eff

def write_matrices(subject_ids, Xrows, N: int, out_dir: Path, suffix: str = "_harmonized.csv"):
    r, c = np.triu_indices(N, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid, v in zip(subject_ids, Xrows):
        A = np.zeros((N, N), dtype=np.float64)
        A[r, c] = v
        A[c, r] = v
        np.fill_diagonal(A, 0.0)
        np.savetxt(out_dir / f"{sid}{suffix}", A, fmt="%.6g", delimiter=",")

# ---------- QC ----------

def site_auc_safe(X: np.ndarray, y: np.ndarray, folds=5, min_per_class=None, seed=13):
    """Macro one-vs-rest AUC that tolerates rare classes and empty test classes."""
    y = np.asarray(y)
    vc = pd.Series(y).value_counts()
    if min_per_class is None:
        min_per_class = folds
    kept_labs = vc[vc >= min_per_class].index
    mask = np.isin(y, kept_labs)
    X = X[mask]; y = y[mask]
    if len(y) == 0 or np.unique(y).size < 2:
        return np.nan, np.nan, 0, 0
    n_splits = int(min(folds, pd.Series(y).value_counts().min()))
    if n_splits < 2:
        return np.nan, np.nan, len(kept_labs), n_splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        ss = StandardScaler(with_mean=True, with_std=True).fit(X[tr])
        Xtr, Xte = ss.transform(X[tr]), ss.transform(X[te])
        clf = LogisticRegression(max_iter=500, multi_class='ovr')
        clf.fit(Xtr, y[tr])
        proba = clf.predict_proba(Xte)
        classes = clf.classes_
        per = []
        for j, cls in enumerate(classes):
            y_bin = (y[te] == cls).astype(int)
            if y_bin.min() == y_bin.max():
                continue
            try:
                per.append(roc_auc_score(y_bin, proba[:, j]))
            except ValueError:
                continue
        if per:
            aucs.append(float(np.mean(per)))
    if not aucs:
        return np.nan, np.nan, len(kept_labs), n_splits
    return float(np.mean(aucs)), float(np.std(aucs)), len(kept_labs), n_splits

def pca_plot(X: np.ndarray, y: np.ndarray, path: Path, title: str, n_sample=500, seed=13):
    if len(X) > n_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=n_sample, replace=False)
        X, y = X[idx], y[idx]
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    labs = pd.Categorical(y)
    plt.figure(figsize=(6, 5), dpi=140)
    for lab in labs.categories:
        mask = (labs == lab)
        if mask.sum() == 0:
            continue
        plt.scatter(Z[mask, 0], Z[mask, 1], s=10, alpha=0.7, label=f"{lab} (n={mask.sum()})")
    plt.legend(fontsize=7, markerscale=1.2, frameon=False, ncol=1)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def subject_panels(ids, X_pre, X_post, N, out_dir: Path, n_panels=6, modality="dwi", seed=42):
    r, c = np.triu_indices(N, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_show = min(n_panels, len(ids))
    pick = rng.choice(len(ids), size=n_show, replace=False)
    for j, i in enumerate(pick, 1):
        sid = ids[i]
        A_pre = np.zeros((N, N)); A_pre[r, c] = X_pre[i]; A_pre[c, r] = A_pre[r, c]
        A_post = np.zeros((N, N)); A_post[r, c] = X_post[i]; A_post[c, r] = A_post[r, c]
        if modality == "dwi":
            P = np.log1p(A_pre); Q = np.log1p(A_post); D = Q - P
            vmax = np.percentile(P, 99) or 1.0
            qmax = np.percentile(Q, 99) or 1.0
            dmax = np.max(np.abs(D)) or 1.0
            cm_diff = None
        else:
            P, Q = A_pre, A_post; D = Q - P
            lim = np.percentile(np.abs(P), 99) or 1.0
            lim2 = np.percentile(np.abs(Q), 99) or 1.0
            vmax = lim; qmax = lim2
            dmax = np.percentile(np.abs(D), 99) or 1.0
            cm_diff = "coolwarm"
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), dpi=140)
        im0 = ax[0].imshow(P, vmin=0 if modality=="dwi" else -vmax, vmax=vmax)
        ax[0].set_title(f"{sid} PRE")
        im1 = ax[1].imshow(Q, vmin=0 if modality=="dwi" else -qmax, vmax=qmax)
        ax[1].set_title("POST")
        im2 = ax[2].imshow(D, vmin=-dmax, vmax=dmax, cmap=cm_diff)
        ax[2].set_title("DIFF (post-pre)")
        for a in ax: a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        plt.tight_layout(); plt.savefig(out_dir / f"subject_panel_{j}_{sid}.png"); plt.close()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Location-only anchored harmonization + QC for DWI/fMRI connectomes.")
    ap.add_argument("--features", required=True, help="Wide features CSV: subject_id, e000000, ...")
    ap.add_argument("--covars",   required=True, help="Covariates CSV with columns [subject_id, SITE, AGE]")
    ap.add_argument("--out-dir",  required=True, help="Output directory")
    ap.add_argument("--prefix",   default="", help="Optional filename prefix for outputs")
    ap.add_argument("--nodes",    type=int, default=None, help="Number of nodes (infers from edges if omitted)")
    ap.add_argument("--id-col",   default=None, help="Subject ID column in features (default: first column)")
    ap.add_argument("--site-col", default="SITE", help="Site/batch column in covars")
    ap.add_argument("--age-col",  default="AGE",  help="Age column in covars")
    ap.add_argument("--ref-site", required=True, help="Reference SITE label to anchor to (must exist after merging)")
    ap.add_argument("--modality", choices=["dwi","fmri"], default="dwi", help="DWI uses log1p; fMRI uses Fisher z")
    ap.add_argument("--min-per-site", type=int, default=20, help="Merge sites with <N subjects into OTHER (ref kept)")
    ap.add_argument("--ridge", type=float, default=1e-3, help="Ridge penalty (stabilize OLS)")
    ap.add_argument("--seed", type=int, default=13)

    # QC options
    ap.add_argument("--qc", action="store_true", help="Run QC (AUC, PCA, subject panels)")
    ap.add_argument("--qc-topk", type=int, default=1500, help="Top-K variance edges for AUC/PCA")
    ap.add_argument("--qc-folds", type=int, default=5, help="Stratified folds for AUC")
    ap.add_argument("--qc-min-per-site", type=int, default=5, help="Only evaluate sites with >= this many samples")
    ap.add_argument("--qc-panels", type=int, default=6, help="Number of random subject panels")
    ap.add_argument("--qc-pca-sample", type=int, default=500, help="Max subjects for PCA scatter")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_base = Path(args.out_dir)
    out_mats = out_base / (args.prefix + "_matrices" if args.prefix else "matrices")
    out_qc   = out_base / (args.prefix + "_qc" if args.prefix else "qc")
    out_mats.mkdir(parents=True, exist_ok=True)
    if args.qc:
        out_qc.mkdir(parents=True, exist_ok=True)

    # --- Load features (wide) ---
    F = pd.read_csv(args.features)
    id_col = args.id_col or F.columns[0]
    if id_col not in F.columns:
        raise SystemExit(f"id-col '{args.id_col}' not found in features")
    F[id_col] = F[id_col].astype(str)
    ids_feat = F[id_col].astype(str).values
    X = F.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    m = X.shape[1]
    N = args.nodes or infer_N_from_m(m)
    if m != N*(N-1)//2:
        raise SystemExit(f"Edge count {m} does not match upper-tri for N={N} (expected {N*(N-1)//2}).")
    print(f"[load] subjects={len(ids_feat)}, edges={m}, N={N}")

    # --- Load covars ---
    C = pd.read_csv(args.covars)
    need = {"subject_id", args.site_col, args.age_col}
    if not need.issubset(C.columns):
        raise SystemExit(f"Covars missing columns: {need - set(C.columns)}")
    C["subject_id"] = C["subject_id"].astype(str)
    C[args.site_col] = C[args.site_col].astype(str).str.strip()
    C[args.age_col]  = pd.to_numeric(C[args.age_col], errors="coerce")

    # --- Align IDs (features order) ---
    inter = pd.Index(ids_feat).intersection(pd.Index(C["subject_id"]))
    if inter.empty:
        raise SystemExit("No overlapping subject IDs between features and covars.")
    F = F[F[id_col].isin(inter)].reset_index(drop=True)
    C = C.set_index("subject_id").loc[F[id_col]].reset_index()
    ids = F[id_col].astype(str).values
    X  = F.drop(columns=[id_col]).values.astype(np.float64)
    ok = C[args.age_col].notna() & C[args.site_col].notna()
    if not ok.all():
        drop_n = int((~ok).sum())
        F = F[ok.values].reset_index(drop=True); C = C[ok.values].reset_index(drop=True)
        ids = F[id_col].astype(str).values
        X   = F.drop(columns=[id_col]).values.astype(np.float64)
        print(f"[info] dropped {drop_n} rows w/ missing AGE/SITE")

    sites_raw = C[args.site_col].astype(str).copy()

    # --- Merge tiny sites (keep REF intact) ---
    vc = C[args.site_col].value_counts()
    rare = [s for s, n in vc.items() if (n < args.min_per_site and s != args.ref_site)]
    if rare:
        C[args.site_col] = C[args.site_col].where(~C[args.site_col].isin(rare), other="OTHER")
        print(f"[merge] {len(rare)} sites -> OTHER (<{args.min_per_site})")
    if args.ref_site not in set(C[args.site_col]):
        print(f"[warn] REF '{args.ref_site}' not found after merging; using first present site as REF.")
        args.ref_site = C[args.site_col].iloc[0]

    # --- Transform (modality) ---
    Y = pre_transform(X, args.modality)  # n×m

    # --- Design & fit (location-only) ---
    Xdesign, S, S_cols = design_matrix(C[args.age_col].to_numpy(dtype=np.float64),
                                       C[args.site_col].astype(str), args.ref_site)
    print(f"[design] p={Xdesign.shape[1]} (intercept + age + {len(S_cols)} site dummies), REF='{args.ref_site}'")
    beta = fit_location_only(Y, Xdesign, ridge=args.ridge)
    Y_adj = anchor_to_ref(Y, beta, S)

    # --- Inverse & clean ---
    X_adj = inverse_transform(Y_adj, args.modality)
    X_adj = np.nan_to_num(X_adj, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Write per-subject matrices ---
    write_matrices(ids, X_adj, N, out_mats, suffix="_harmonized.csv")
    print(f"Wrote {len(ids)} harmonized matrices to {out_mats}")

    # --- QC (optional) ---
    if args.qc:
        qc_dir = out_qc
        qc_dir.mkdir(parents=True, exist_ok=True)

        # Top-K variance edges from PRE
        var = X.var(axis=0)
        idx = np.argsort(var)[-args.qc_topk:] if args.qc_topk < m else np.arange(m)
        XpreK  = X[:, idx]
        XpostK = X_adj[:, idx]
        sitesK = C[args.site_col].astype(str).values

        # AUC
        auc_pre_m, auc_pre_s, nlab, ns = site_auc_safe(
            XpreK, sitesK, folds=args.qc_folds, min_per_class=args.qc_min_per_site, seed=args.seed
        )
        auc_post_m, auc_post_s, _, _ = site_auc_safe(
            XpostK, sitesK, folds=args.qc_folds, min_per_class=args.qc_min_per_site, seed=args.seed
        )
        (qc_dir / "auc_site.txt").write_text(
            f"K={args.qc_topk}, evaluated_labels={nlab}, n_splits={ns}\n"
            f"AUC_site_pre : {auc_pre_m:.3f} ± {auc_pre_s:.3f}\n"
            f"AUC_site_post: {auc_post_m:.3f} ± {auc_post_s:.3f}\n"
        )
        print(f"[AUC] pre={auc_pre_m:.3f}±{auc_pre_s:.3f} | post={auc_post_m:.3f}±{auc_post_s:.3f} "
              f"(labels≥{args.qc_min_per_site}, folds={ns}, K={args.qc_topk})")

        # Mean-to-ref shrinkage
        def site_mean(Xm, labs, label):
            msk = (labs == label)
            return Xm[msk].mean(axis=0) if msk.any() else None
        ref = args.ref_site
        mu_ref_pre  = site_mean(X,      sitesK, ref)
        mu_ref_post = site_mean(X_adj,  sitesK, ref)
        rows = []
        for lab in pd.unique(sitesK):
            mu_pre  = site_mean(X,     sitesK, lab)
            mu_post = site_mean(X_adj, sitesK, lab)
            if mu_pre is None or mu_post is None: continue
            d_pre  = float(np.linalg.norm(mu_pre  - mu_ref_pre))
            d_post = float(np.linalg.norm(mu_post - mu_ref_post))
            rows.append((lab, int((sitesK==lab).sum()), d_pre, d_post))
        df = pd.DataFrame(rows, columns=["SITE","n","dist_pre_to_ref","dist_post_to_ref"]).sort_values("n", ascending=False)
        df.to_csv(qc_dir / "mean_to_ref_distances.csv", index=False)

        # PCA
        def pca_plot_local(Xm, labs, path, title):
            Xs = Xm
            if len(Xs) > args.qc_pca_sample:
                ridx = np.random.default_rng(args.seed).choice(len(Xs), size=args.qc_pca_sample, replace=False)
                Xs, labs = Xs[ridx], labs[ridx]
            Xs = StandardScaler().fit_transform(Xs)
            Z = PCA(n_components=2, random_state=args.seed).fit_transform(Xs)
            labs_cat = pd.Categorical(labs)
            plt.figure(figsize=(6, 5), dpi=140)
            for lab in labs_cat.categories:
                msk = (labs_cat == lab)
                if msk.sum() == 0: continue
                plt.scatter(Z[msk,0], Z[msk,1], s=10, alpha=0.7, label=f"{lab} (n={msk.sum()})")
            plt.legend(fontsize=7, markerscale=1.2, frameon=False, ncol=1)
            plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)
            plt.tight_layout(); plt.savefig(path); plt.close()

        pca_plot_local(XpreK, sitesK, qc_dir / "pca_pre.png",  f"PCA PRE (top-{args.qc_topk})")
        pca_plot_local(XpostK, sitesK, qc_dir / "pca_post.png", f"PCA POST (top-{args.qc_topk})")

        # Subject panels
        subject_panels(ids, X, X_adj, N, qc_dir / "panels", n_panels=args.qc_panels, modality=args.modality, seed=args.seed)

if __name__ == "__main__":
    main()
