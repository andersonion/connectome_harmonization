#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fold-wise ComBat / ComBat-GAM with per-subject matrix outputs
+ optional diagnostics (AUC/PCA) and subject comparison plots.

Modes:
  - OOF (no leakage): --cv-folds > 0
  - Full-fit:         --cv-folds 0 (fits once on all, applies to all)

Outputs:
  - Per-subject harmonized N×N matrices (CSV) in:
      {out_dir}/{prefix}_matrices_oof/   (OOF)
      {out_dir}/{prefix}_matrices_full/  (full-fit)
  - Diagnostics (if --diagnostics):
      AUC summary CSV: {out_dir}/{prefix}_cv_auc.csv
      PCA plots/CSVs per fold: {out_dir}/{prefix}_fold{NN}_pca_{pre,post}.{png,csv}
  - Subject comparison plots (if --subject-plots N):
      {out_dir}/{prefix}_subject_plots/{SUBJ}_pre_post_diff.png
"""

from __future__ import annotations
import argparse, logging, sys, math, warnings, os
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

# Harmonization
from neuroHarmonize import harmonizationLearn, harmonizationApply

# CV / diagnostics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Progress (optional)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ----------------------------- UX / Logging ----------------------------- #

def setup_logging(v: int) -> None:
    level = logging.WARNING if v <= 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def banner(msg: str):
    print(f"\n=== {msg} ===", flush=True)

def pbar_iter(it, total=None, desc="", enable=True):
    if enable and tqdm is not None:
        return tqdm(it, total=total, desc=desc, ncols=90)
    return it

# ----------------------- File / Vectorization utils --------------------- #

def _read_noheader_matrix(fp: str) -> np.ndarray:
    try:
        return pd.read_csv(fp, header=None).values
    except Exception:
        return pd.read_csv(fp, sep="\t", header=None).values

def _vectorize_upper(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    if A.ndim == 1 or 1 in A.shape:
        return A.ravel()
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        r, c = np.triu_indices(A.shape[0], k=1)
        return A[r, c]
    return A.ravel()

def read_features(features_path: Path, id_col: str, manifest_vectorize: bool, progress: bool) -> Tuple[pd.DataFrame, Optional[int]]:
    """
    Returns (features_df with id_col + numeric edge columns, inferred_N or None).
    If manifest has 'filepath', vectorizes each file (upper-triangle k=1).
    """
    df = pd.read_csv(features_path)
    if manifest_vectorize and "filepath" in df.columns:
        need = {id_col, "filepath"}
        if not need.issubset(df.columns):
            raise SystemExit(f"Manifest must contain columns {need}. Found: {list(df.columns)}")
        rows, ids = [], []
        first_len = None
        last_A = None
        for rec in pbar_iter(df.itertuples(index=False), total=len(df), desc="Vectorizing", enable=progress):
            A = _read_noheader_matrix(getattr(rec, "filepath"))
            v = _vectorize_upper(A)
            if first_len is None:
                first_len = v.size
            elif v.size != first_len:
                raise SystemExit(f"Vector length mismatch: {v.size} vs {first_len}")
            rows.append(v.astype(np.float32)); ids.append(str(getattr(rec, id_col)))
            last_A = A
        cols = [f"e{i:06d}" for i in range(first_len)]
        out = pd.DataFrame(rows, columns=cols)
        out.insert(0, id_col, ids)
        N_guess = last_A.shape[0] if (last_A is not None and last_A.ndim == 2 and last_A.shape[0] == last_A.shape[1]) else None
        return out, N_guess

    if id_col not in df.columns:
        raise SystemExit(f"Features missing id-col '{id_col}'. Columns: {list(df.columns)[:10]}")

    id_series = df[id_col].astype(str)
    num = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
    if num.isnull().any().any():
        bad = [c for c in num.columns if num[c].isnull().any()]
        logging.warning("Dropping %d non-numeric/NaN feature columns: %s",
                        len(bad), ", ".join(bad[:8]) + ("..." if len(bad) > 8 else ""))
        num = num.drop(columns=bad)
    out = pd.concat([id_series, num], axis=1)
    return out, None

def infer_nodes_from_edges(m: int) -> Optional[int]:
    disc = 1 + 8*m
    rt = int(math.isqrt(disc))
    if rt*rt != disc: return None
    N = (1 + rt) // 2
    return N if N*(N-1)//2 == m else None

def unvectorize_upper(vec: np.ndarray, N: int) -> np.ndarray:
    A = np.zeros((N, N), dtype=float)
    r, c = np.triu_indices(N, k=1)
    if vec.size != r.size:
        raise ValueError(f"Vector length {vec.size} != expected {r.size} for N={N}")
    A[r, c] = vec
    A[c, r] = vec
    np.fill_diagonal(A, 0.0)
    return A

# --------------------- Harmonization plumbing --------------------------- #

def learn_apply(Xtr: np.ndarray, cov_tr: pd.DataFrame,
                Xte: np.ndarray, cov_te: pd.DataFrame,
                site_col: str, age_col: Optional[str],
                mode: str, ref_batch: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Try new API (batch_col=...), fallback to legacy 'SITE'."""
    smooth_terms = [age_col] if (mode == "gam" and age_col and age_col in cov_tr.columns) else []
    try:
        model, Xtr_adj = harmonizationLearn(
            Xtr, cov_tr, smooth_terms=smooth_terms,
            batch_col=site_col, ref_batch=ref_batch
        )
        Xte_adj = harmonizationApply(Xte, cov_te, model)
        return Xtr_adj, Xte_adj
    except TypeError as e:
        if "batch_col" not in str(e):
            raise
    cov_tr_old = cov_tr.rename(columns={site_col: "SITE"})
    cov_te_old = cov_te.rename(columns={site_col: "SITE"})
    model, Xtr_adj = harmonizationLearn(
        Xtr, cov_tr_old, smooth_terms=smooth_terms, ref_batch=ref_batch
    )
    Xte_adj = harmonizationApply(Xte, cov_te_old, model)
    return Xtr_adj, Xte_adj

def augment_test_levels(cov_te: pd.DataFrame, Xte: np.ndarray,
                        site_col: str, missing: List[str],
                        age_col: Optional[str] = None, age_fill: Optional[float] = None) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """Add one dummy row per missing training batch level (legacy apply quirk)."""
    if not missing:
        return cov_te, Xte, 0
    rows = []
    for lvl in missing:
        row = {site_col: str(lvl)}
        if age_col and age_col in cov_te.columns:
            row[age_col] = age_fill if age_fill is not None else float(np.nanmedian(cov_te[age_col].values))
        rows.append(row)
    add_df = pd.DataFrame(rows)
    cov_aug = pd.concat([cov_te, add_df], ignore_index=True)
    base = Xte[0:1] if Xte.shape[0] else np.zeros((1, Xte.shape[1]), dtype=np.float32)
    X_aug = np.vstack([Xte, np.repeat(base, len(missing), axis=0)]).astype(np.float32)
    return cov_aug, X_aug, len(missing)

# ----------------------- Diagnostics helpers (AUC/PCA) ------------------ #

def sanitize_for_auc(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A_tr = np.asarray(Xtr, dtype=np.float32, order="C").copy()
    A_te = np.asarray(Xte, dtype=np.float32, order="C").copy()
    fin_tr = np.isfinite(A_tr); fin_te = np.isfinite(A_te)
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(np.where(fin_tr, A_tr, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    A_tr = np.where(fin_tr, A_tr, col_mean)
    A_te = np.where(fin_te, A_te, col_mean)
    return A_tr, A_te

def subsample_features(Xtr: np.ndarray, Xte: np.ndarray, k: Optional[int], seed: int = 42):
    if k is None or Xtr.shape[1] <= k:
        return Xtr, Xte
    rng = np.random.default_rng(seed)
    cols = rng.choice(Xtr.shape[1], size=k, replace=False)
    return Xtr[:, cols], Xte[:, cols]

def site_auc_on_test(Xtr: np.ndarray, ytr, Xte: np.ndarray, yte) -> float:
    ytr = np.asarray(ytr); yte = np.asarray(yte)
    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    yte_enc = le.transform(yte)
    te_classes = np.unique(yte_enc)
    if len(te_classes) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(Xtr, ytr_enc)
    prob_full = clf.predict_proba(Xte)
    if len(le.classes_) == 2:
        return float(roc_auc_score(yte_enc, prob_full[:, 1]))
    prob_sub = prob_full[:, te_classes]
    return float(roc_auc_score(yte_enc, prob_sub, multi_class="ovr", average="macro"))

def pca_scatter(df: pd.DataFrame, title: str, out_png: Path):
    plt.figure(figsize=(6,5))
    sites = df["SITE"].astype(str).unique().tolist()
    many = len(sites) > 12
    for lab in sites:
        m = df["SITE"].astype(str) == str(lab)
        plt.scatter(df.loc[m,"PC1"], df.loc[m,"PC2"], s=10, alpha=0.7, label=str(lab))
    if not many:
        plt.legend(markerscale=2, bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=0.)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# ---------------------- Subject plot helper (3-panel) ------------------- #

def plot_subject_triptych(A_pre: np.ndarray, A_post: np.ndarray, sid: str, out_png: Path, eps: float = 1e-6):
    vmax = max(eps, float(np.nanmax(A_pre)), float(np.nanmax(A_post)))
    D = A_post - A_pre
    vmaxd = float(np.nanmax(np.abs(D))) if np.isfinite(D).any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(A_pre + eps, norm=LogNorm(vmin=eps, vmax=vmax))
    axes[0].set_title("Pre (log)"); plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(A_post + eps, norm=LogNorm(vmin=eps, vmax=vmax))
    axes[1].set_title("Post (log)"); plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(D, vmin=-vmaxd, vmax=vmaxd, cmap="bwr")
    axes[2].set_title("Post - Pre"); plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(str(sid)); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

# ---------------------------------- Main -------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Harmonize connectomes to per-subject matrices with optional diagnostics.")
    ap.add_argument("--features", required=True, help="Numeric edges CSV (id + edges...) OR manifest (id, filepath).")
    ap.add_argument("--covars", required=True, help="Covariates CSV with SITE and AGE.")
    ap.add_argument("--id-col", default="subject_id")
    ap.add_argument("--site-col", default="SITE")
    ap.add_argument("--age-col", default="AGE")
    ap.add_argument("--mode", choices=["gam","anchored"], default="anchored",
                    help="ComBat-GAM (age spline) or linear/anchored ComBat.")
    ap.add_argument("--ref-batch", default=None, help="Reference batch label in SITE, e.g. 'ADDecode:1' or '1'.")
    ap.add_argument("--nodes", type=int, default=None, help="Number of nodes (e.g., 84). If omitted, inferred from edges.")
    ap.add_argument("--cv-folds", type=int, default=0, help="0=full fit; >0 = OOF CV with N folds (stratified by SITE).")
    ap.add_argument("--cv-handle-rare", choices=["merge","drop","ignore"], default="merge",
                    help="CV only: how to handle tiny SITE classes in CV labels.")
    ap.add_argument("--min-per-site", type=int, default=2, help="CV only: sites with < this many are 'rare'.")
    ap.add_argument("--save-folds", action="store_true", help="Also dump per-fold train/test matrices.")
    # Diagnostics
    ap.add_argument("--diagnostics", action="store_true", help="Run AUC/PCA diagnostics.")
    ap.add_argument("--auc-max-features", type=int, default=None, help="Subsample this many features for AUC only.")
    ap.add_argument("--pca", action="store_true", help="Save PCA (PC1/PC2) per fold, colored by SITE.")
    ap.add_argument("--pca-max-features", type=int, default=None, help="Subsample features for PCA only (defaults to AUC cap).")
    ap.add_argument("--pca-sample", type=int, default=None, help="Randomly subsample this many subjects for PCA plots.")
    ap.add_argument("--diag-folds", type=int, default=3, help="Full-fit only: CV folds for diagnostics.")
    # Subject plots
    ap.add_argument("--subject-plots", type=int, default=0, help="If >0, sample N subjects and save pre/post/diff plots.")
    ap.add_argument("--plots-dir", default=None, help="Output directory for subject plots (default inside out_dir).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for subsampling.")
    # IO / misc
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("-v","--verbose", action="count", default=1)
    args = ap.parse_args()

    setup_logging(args.verbose)
    progress = not args.no_progress

    # Quiet some noisy (non-fatal) warnings
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

    features_path = Path(args.features)
    covars_path   = Path(args.covars)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    # ------------------ Load & check features ------------------ #
    banner("LOAD FEATURES")
    X_df, N_guess = read_features(features_path, args.id_col, manifest_vectorize=True, progress=progress)
    ids_all = X_df[args.id_col].astype(str).values
    X_num = X_df.drop(columns=[args.id_col]).apply(pd.to_numeric, errors="coerce")
    if X_num.isnull().any().any():
        bad = int(X_num.isnull().any(axis=1).sum())
        raise SystemExit(f"{bad} row(s) contain NaNs in features after coercion.")
    m = X_num.shape[1]
    N = args.nodes or N_guess or infer_nodes_from_edges(m)
    if N is None:
        raise SystemExit(f"Cannot infer nodes from {m} edges. Pass --nodes (e.g., --nodes 84).")
    expected_m = N*(N-1)//2
    if m != expected_m:
        raise SystemExit(f"Edge count {m} != expected upper-tri size {expected_m} for N={N}.")
    print(f"INFO: Detected N={N} nodes ({m} edges).", flush=True)

    # ------------------ Load & align covariates ---------------- #
    banner("LOAD COVARIATES")
    C = pd.read_csv(covars_path)
    if args.id_col not in C.columns:
        raise SystemExit(f"Covars missing id-col '{args.id_col}'.")
    C[args.id_col] = C[args.id_col].astype(str)
    C[args.site_col] = C[args.site_col].astype(str).str.strip()
    if args.age_col in C.columns:
        C[args.age_col] = pd.to_numeric(C[args.age_col], errors="coerce")
        if C[args.age_col].isna().any():
            ex = C.loc[C[args.age_col].isna(), [args.id_col, args.age_col]].head(5)
            raise SystemExit(f"AGE non-numeric/missing:\n{ex}")

    keep_ids = sorted(set(ids_all) & set(C[args.id_col]))
    if not keep_ids:
        raise SystemExit("No overlapping subject IDs between features and covars.")
    order = pd.Index(keep_ids)
    X = X_num.set_index(pd.Index(ids_all)).loc[order].reset_index(drop=True).values.astype(np.float32)
    C = C.set_index(args.id_col).loc[order].reset_index()
    ids_all = np.array(keep_ids, dtype=str)

    ref_label = None if args.ref_batch is None else str(args.ref_batch).strip()
    if ref_label is not None and ref_label not in set(C[args.site_col].unique()):
        top = C[args.site_col].value_counts().head(20)
        raise SystemExit(f"--ref-batch '{ref_label}' not found in SITE. Top labels:\n{top}")

    # For subject plots: we’ll need subject -> row index
    id_to_idx: Dict[str, int] = {sid: i for i, sid in enumerate(ids_all)}
    r_idx, c_idx = np.triu_indices(N, k=1)

    # -------------------- FULL-FIT vs OOF ---------------------- #
    if args.cv_folds <= 0:
        banner("FULL FIT")
        cov_use = C[[args.site_col] + ([args.age_col] if args.age_col in C.columns else [])].copy()
        try:
            model, _ = harmonizationLearn(
                X, cov_use,
                smooth_terms=([args.age_col] if args.mode=="gam" and args.age_col in cov_use.columns else []),
                batch_col=args.site_col, ref_batch=ref_label
            )
            X_adj = harmonizationApply(X, cov_use, model)
        except TypeError as e:
            if "batch_col" not in str(e): raise
            cov_old = cov_use.rename(columns={args.site_col: "SITE"})
            model, _ = harmonizationLearn(
                X, cov_old,
                smooth_terms=([args.age_col] if args.mode=="gam" and args.age_col in cov_old.columns else []),
                ref_batch=ref_label
            )
            X_adj = harmonizationApply(X, cov_old, model)

        # Write one matrix per subject
        mat_dir = outdir / f"{args.prefix}_matrices_full"
        mat_dir.mkdir(parents=True, exist_ok=True)
        for sid, v in pbar_iter(zip(ids_all, X_adj), total=len(ids_all), desc="Writing matrices", enable=progress):
            A = np.zeros((N,N), dtype=float); A[r_idx, c_idx] = v; A[c_idx, r_idx] = v; np.fill_diagonal(A, 0.0)
            np.savetxt(mat_dir / f"{sid}_harmonized.csv", A, fmt="%.6g", delimiter=",")

        # Diagnostics (optional): run a CV over pre/post to report AUC/PCA
        if args.diagnostics:
            banner("DIAGNOSTICS (full-fit)")
            n_splits = max(2, int(args.diag_folds))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
            sites = C[args.site_col].astype(str).values
            per_fold = []
            for fold, (tr, te) in enumerate(skf.split(np.arange(len(C)), sites), start=1):
                Xtr, Xte = X[tr], X[te]
                Xtr_post, Xte_post = X_adj[tr], X_adj[te]
                # AUC pre
                Xtr_s, Xte_s = sanitize_for_auc(Xtr, Xte)
                Xtr_s, Xte_s = subsample_features(Xtr_s, Xte_s, args.auc_max_features, seed=args.seed)
                pre_auc = site_auc_on_test(Xtr_s, sites[tr], Xte_s, sites[te])
                # AUC post
                Xtr_s, Xte_s = sanitize_for_auc(Xtr_post, Xte_post)
                Xtr_s, Xte_s = subsample_features(Xtr_s, Xte_s, args.auc_max_features, seed=args.seed)
                post_auc = site_auc_on_test(Xtr_s, sites[tr], Xte_s, sites[te])
                # PCA (optional)
                if args.pca:
                    k = args.pca_max_features if args.pca_max_features is not None else args.auc_max_features
                    # PRE
                    Xtr_v, Xte_v = sanitize_for_auc(Xtr, Xte)
                    Xtr_v, Xte_v = subsample_features(Xtr_v, Xte_v, k, seed=args.seed)
                    Z = PCA(n_components=2, svd_solver="randomized", random_state=args.seed).fit_transform(
                        np.vstack([Xtr_v, Xte_v])
                    )
                    Ztr, Zte = Z[:len(tr)], Z[len(tr):]
                    df_pre = pd.DataFrame({
                        "PC1": np.r_[Ztr[:,0], Zte[:,0]],
                        "PC2": np.r_[Ztr[:,1], Zte[:,1]],
                        "SITE": np.r_[sites[tr], sites[te]],
                        "split": ["train"]*len(tr) + ["test"]*len(te),
                    })
                    if args.pca_sample and len(df_pre) > args.pca_sample:
                        df_pre = df_pre.sample(n=args.pca_sample, random_state=args.seed).reset_index(drop=True)
                    pre_csv = outdir / f"{args.prefix}_fold{fold:02d}_pca_pre.csv"
                    pre_png = outdir / f"{args.prefix}_fold{fold:02d}_pca_pre.png"
                    df_pre.to_csv(pre_csv, index=False)
                    pca_scatter(df_pre, f"PCA pre (fold {fold})", pre_png)
                    # POST
                    Xtr_v, Xte_v = sanitize_for_auc(Xtr_post, Xte_post)
                    Xtr_v, Xte_v = subsample_features(Xtr_v, Xte_v, k, seed=args.seed)
                    Z = PCA(n_components=2, svd_solver="randomized", random_state=args.seed).fit_transform(
                        np.vstack([Xtr_v, Xte_v])
                    )
                    Ztr, Zte = Z[:len(tr)], Z[len(tr):]
                    df_post = pd.DataFrame({
                        "PC1": np.r_[Ztr[:,0], Zte[:,0]],
                        "PC2": np.r_[Ztr[:,1], Zte[:,1]],
                        "SITE": np.r_[sites[tr], sites[te]],
                        "split": ["train"]*len(tr) + ["test"]*len(te),
                    })
                    if args.pca_sample and len(df_post) > args.pca_sample:
                        df_post = df_post.sample(n=args.pca_sample, random_state=args.seed).reset_index(drop=True)
                    post_csv = outdir / f"{args.prefix}_fold{fold:02d}_pca_post.csv"
                    post_png = outdir / f"{args.prefix}_fold{fold:02d}_pca_post.png"
                    df_post.to_csv(post_csv, index=False)
                    pca_scatter(df_post, f"PCA post (fold {fold})", post_png)
                per_fold.append({"fold": fold, "pre_auc": float(pre_auc), "post_auc": float(post_auc)})

            res = pd.DataFrame(per_fold)
            res.to_csv(outdir / f"{args.prefix}_cv_auc.csv", index=False)
            print(f"Wrote AUC summary to {outdir / (args.prefix + '_cv_auc.csv')}", flush=True)

        # Subject triptychs (optional; uses full-fit post vectors)
        if args.subject_plots and args.subject_plots > 0:
            banner("SUBJECT PLOTS")
            rng = np.random.default_rng(args.seed)
            candidates = ids_all.tolist()
            n = min(args.subject_plots, len(candidates))
            sample = rng.choice(candidates, size=n, replace=False)
            plots_dir = Path(args.plots_dir) if args.plots_dir else outdir / f"{args.prefix}_subject_plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            for sid in pbar_iter(sample, total=len(sample), desc="Plot subjects", enable=progress):
                i = id_to_idx[sid]
                A_pre  = unvectorize_upper(X[i], N)
                A_post = unvectorize_upper(X_adj[i], N)
                plot_subject_triptych(A_pre, A_post, sid, plots_dir / f"{sid}_pre_post_diff.png")

        print(f"Wrote {len(ids_all)} matrices to {outdir / f'{args.prefix}_matrices_full'}")
        return

    # -------------------- OOF: CV-based harmonization -------------------- #
    banner("CV OOF")
    sites = C[args.site_col].astype(str).values
    vc = pd.Series(sites).value_counts()
    rare = [lab for lab,cnt in vc.items() if cnt < args.min_per_site and (ref_label is None or lab != ref_label)]
    if args.cv_handle_rare == "drop" and rare:
        keep_mask = ~pd.Series(sites).isin(rare)
        X = X[keep_mask.values]
        C = C.loc[keep_mask.values].reset_index(drop=True)
        sites = C[args.site_col].astype(str).values
        ids_all = ids_all[keep_mask.values]
        id_to_idx = {sid: i for i, sid in enumerate(ids_all)}
        logging.info("Dropped rare sites for CV: %s", rare)
    elif args.cv_handle_rare == "merge" and rare:
        sites = np.where(pd.Series(sites).isin(rare), "OTHER", sites)

    min_class = int(pd.Series(sites).value_counts().min())
    n_splits = max(2, min(int(args.cv_folds), min_class))
    if n_splits != int(args.cv_folds):
        print(f"[warn] Reducing folds from {args.cv_folds} to {n_splits} to satisfy class counts.", flush=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    oof_dir = outdir / f"{args.prefix}_matrices_oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    if args.save_folds:
        folds_root = outdir / f"{args.prefix}_matrices_folds"; folds_root.mkdir(parents=True, exist_ok=True)

    # For diagnostics & subject plots
    per_fold_auc = []
    post_vecs: Dict[str, np.ndarray] = {}

    for fold, (tr, te) in enumerate(skf.split(np.arange(len(C)), sites), start=1):
        Xtr = X[tr].astype(np.float32); Xte = X[te].astype(np.float32)
        cov_tr = C.iloc[tr].copy(); cov_te = C.iloc[te].copy()

        keep_cols = [args.site_col]; 
        if args.age_col in cov_tr.columns: keep_cols.append(args.age_col)
        cov_tr_use = cov_tr[keep_cols].copy()
        cov_te_use = cov_te[keep_cols].copy()
        if args.age_col in cov_tr_use.columns:
            cov_tr_use[args.age_col] = pd.to_numeric(cov_tr_use[args.age_col], errors="coerce")
            cov_te_use[args.age_col] = pd.to_numeric(cov_te_use[args.age_col], errors="coerce")
            if cov_tr_use[args.age_col].isna().any() or cov_te_use[args.age_col].isna().any():
                raise SystemExit("Non-numeric AGE encountered after split.")

        train_levels = sorted(cov_tr_use[args.site_col].astype(str).unique().tolist())
        test_levels  = set(cov_te_use[args.site_col].astype(str))
        missing = [lvl for lvl in train_levels if lvl not in test_levels]
        age_fill = float(np.nanmedian(cov_tr_use[args.age_col].values)) if (args.age_col in cov_tr_use.columns) else None
        cov_te_aug, Xte_aug, n_add = augment_test_levels(cov_te_use, Xte, args.site_col, missing,
                                                         age_col=(args.age_col if args.age_col in cov_tr_use.columns else None),
                                                         age_fill=age_fill)
        if n_add:
            print(f"[info] fold {fold}: added {n_add} dummy test row(s): {missing}", flush=True)

        # Fit/apply
        Xtr_adj, Xte_adj_full = learn_apply(Xtr, cov_tr_use, Xte_aug, cov_te_aug,
                                            site_col=args.site_col, age_col=args.age_col if args.mode=="gam" else None,
                                            mode=args.mode, ref_batch=ref_label)
        Xte_adj = Xte_adj_full[:Xte.shape[0], :]

        # Write OOF matrices for TEST subjects + cache post vectors for plots
        te_ids = ids_all[te]
        for sid, v in pbar_iter(zip(te_ids, Xte_adj), total=len(te_ids), desc=f"Fold {fold}/{n_splits} write TEST", enable=progress):
            A = np.zeros((N,N), dtype=float); A[r_idx, c_idx] = v; A[c_idx, r_idx] = v; np.fill_diagonal(A, 0.0)
            np.savetxt(oof_dir / f"{sid}_harmonized.csv", A, fmt="%.6g", delimiter=",")
            post_vecs[str(sid)] = v.copy()

        # Optional fold dumps (train/test)
        if args.save_folds:
            dtr = folds_root / f"fold{fold:02d}_train"; dtr.mkdir(parents=True, exist_ok=True)
            dte = folds_root / f"fold{fold:02d}_test" ; dte.mkdir(parents=True, exist_ok=True)
            tr_ids = ids_all[tr]
            for sid, v in zip(tr_ids, Xtr_adj):
                A = np.zeros((N,N), dtype=float); A[r_idx, c_idx] = v; A[c_idx, r_idx] = v; np.fill_diagonal(A, 0.0)
                np.savetxt(dtr / f"{sid}_harmonized.csv", A, fmt="%.6g", delimiter=",")
            for sid, v in zip(te_ids, Xte_adj):
                A = np.zeros((N,N), dtype=float); A[r_idx, c_idx] = v; A[c_idx, r_idx] = v; np.fill_diagonal(A, 0.0)
                np.savetxt(dte / f"{sid}_harmonized.csv", A, fmt="%.6g", delimiter=",")

        # Diagnostics per fold (AUC/PCA)
        if args.diagnostics:
            # AUC pre
            Xtr_s, Xte_s = sanitize_for_auc(Xtr, Xte)
            Xtr_s, Xte_s = subsample_features(Xtr_s, Xte_s, args.auc_max_features, seed=args.seed)
            pre_auc = site_auc_on_test(Xtr_s, sites[tr], Xte_s, sites[te])
            # AUC post
            Xtr_s, Xte_s = sanitize_for_auc(Xtr_adj, Xte_adj)
            Xtr_s, Xte_s = subsample_features(Xtr_s, Xte_s, args.auc_max_features, seed=args.seed)
            post_auc = site_auc_on_test(Xtr_s, sites[tr], Xte_s, sites[te])
            per_fold_auc.append({"fold": fold, "pre_auc": float(pre_auc), "post_auc": float(post_auc)})
            # PCA
            if args.pca:
                k = args.pca_max_features if args.pca_max_features is not None else args.auc_max_features
                # PRE
                Xtr_v, Xte_v = sanitize_for_auc(Xtr, Xte)
                Xtr_v, Xte_v = subsample_features(Xtr_v, Xte_v, k, seed=args.seed)
                Z = PCA(n_components=2, svd_solver="randomized", random_state=args.seed).fit_transform(
                    np.vstack([Xtr_v, Xte_v])
                )
                Ztr, Zte = Z[:len(tr)], Z[len(tr):]
                df_pre = pd.DataFrame({
                    "PC1": np.r_[Ztr[:,0], Zte[:,0]], "PC2": np.r_[Ztr[:,1], Zte[:,1]],
                    "SITE": np.r_[sites[tr], sites[te]],
                    "split": ["train"]*len(tr) + ["test"]*len(te),
                })
                if args.pca_sample and len(df_pre) > args.pca_sample:
                    df_pre = df_pre.sample(n=args.pca_sample, random_state=args.seed).reset_index(drop=True)
                pre_csv = outdir / f"{args.prefix}_fold{fold:02d}_pca_pre.csv"
                pre_png = outdir / f"{args.prefix}_fold{fold:02d}_pca_pre.png"
                df_pre.to_csv(pre_csv, index=False); pca_scatter(df_pre, f"PCA pre (fold {fold})", pre_png)
                # POST
                Xtr_v, Xte_v = sanitize_for_auc(Xtr_adj, Xte_adj)
                Xtr_v, Xte_v = subsample_features(Xtr_v, Xte_v, k, seed=args.seed)
                Z = PCA(n_components=2, svd_solver="randomized", random_state=args.seed).fit_transform(
                    np.vstack([Xtr_v, Xte_v])
                )
                Ztr, Zte = Z[:len(tr)], Z[len(tr):]
                df_post = pd.DataFrame({
                    "PC1": np.r_[Ztr[:,0], Zte[:,0]], "PC2": np.r_[Ztr[:,1], Zte[:,1]],
                    "SITE": np.r_[sites[tr], sites[te]],
                    "split": ["train"]*len(tr) + ["test"]*len(te),
                })
                if args.pca_sample and len(df_post) > args.pca_sample:
                    df_post = df_post.sample(n=args.pca_sample, random_state=args.seed).reset_index(drop=True)
                post_csv = outdir / f"{args.prefix}_fold{fold:02d}_pca_post.csv"
                post_png = outdir / f"{args.prefix}_fold{fold:02d}_pca_post.png"
                df_post.to_csv(post_csv, index=False); pca_scatter(df_post, f"PCA post (fold {fold})", post_png)

    # AUC summary (OOF mode)
    if args.diagnostics and len(per_fold_auc):
        res = pd.DataFrame(per_fold_auc)
        res.to_csv(outdir / f"{args.prefix}_cv_auc.csv", index=False)
        print(f"Wrote AUC summary to {outdir / (args.prefix + '_cv_auc.csv')}", flush=True)

    # Subject triptychs (OOF): sample among subjects with post vectors
    if args.subject_plots and args.subject_plots > 0:
        banner("SUBJECT PLOTS")
        rng = np.random.default_rng(args.seed)
        candidates = list(post_vecs.keys())
        if not candidates:
            print("[warn] No harmonized OOF subjects found for plotting.", flush=True)
        else:
            n = min(args.subject_plots, len(candidates))
            sample = rng.choice(candidates, size=n, replace=False)
            plots_dir = Path(args.plots_dir) if args.plots_dir else outdir / f"{args.prefix}_subject_plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            for sid in pbar_iter(sample, total=len(sample), desc="Plot subjects", enable=progress):
                i = id_to_idx[sid]
                A_pre  = unvectorize_upper(X[i], N)
                A_post = unvectorize_upper(post_vecs[sid], N)
                plot_subject_triptych(A_pre, A_post, sid, plots_dir / f"{sid}_pre_post_diff.png")

    # Final count
    oof_dir = outdir / f"{args.prefix}_matrices_oof"
    n_written = sum(1 for _ in oof_dir.glob("*.csv")) if oof_dir.exists() else 0
    if n_written:
        print(f"Wrote OOF matrices for {n_written} subjects to {oof_dir}", flush=True)


if __name__ == "__main__":
    main()
