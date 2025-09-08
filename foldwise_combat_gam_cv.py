#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harmonize connectomes and write one 84x84 matrix per subject.

Modes:
  * Full-fit (default): fit on ALL subjects, apply to ALL, write matrices.
  * CV OOF: --cv-folds N => per-fold fit/train, apply to TEST only, write OOF matrices.

Inputs:
  * --features: either numeric edges table (id + edges...) OR manifest (id, filepath)
  * --covars: covariates with SITE (batch) and AGE; ID column aligns to features

Vectorization assumptions:
  * Manifest matrices are square and vectorized as upper triangle (k=1).
  * If features are already numeric, we assume the same order. Pass --nodes if unsure.
"""

from __future__ import annotations
import argparse, logging, sys, math, warnings
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationLearn, harmonizationApply

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ------------------------- Logging / Progress -------------------------- #

def setup_logging(v: int) -> None:
    level = logging.WARNING if v <= 0 else logging.INFO if v == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def pbar_iter(it, total=None, desc="", enable=True):
    if enable and tqdm is not None:
        return tqdm(it, total=total, desc=desc, ncols=90)
    return it

def banner(msg: str):
    print(f"\n=== {msg} ===", flush=True)

# ------------------------- File / Vectorization ------------------------ #

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
    Returns (features_df with id_col + numeric edge columns, inferred_N or None)
    If manifest_vectorize=True and 'filepath' in columns, reads each file and vectorizes upper triangle.
    """
    df = pd.read_csv(features_path)
    if manifest_vectorize and "filepath" in df.columns:
        need = {id_col, "filepath"}
        if not need.issubset(df.columns):
            raise SystemExit(f"Manifest must contain columns {need}. Found: {list(df.columns)}")
        rows, ids = [], []
        first_len = None
        for rec in pbar_iter(df.itertuples(index=False), total=len(df), desc="Vectorizing", enable=progress):
            A = _read_noheader_matrix(getattr(rec, "filepath"))
            v = _vectorize_upper(A)
            first_len = v.size if first_len is None else first_len
            if v.size != first_len:
                raise SystemExit(f"Vector length mismatch in manifest: {v.size} vs {first_len}")
            rows.append(v.astype(np.float32)); ids.append(str(getattr(rec, id_col)))
        cols = [f"e{i:06d}" for i in range(first_len)]
        out = pd.DataFrame(rows, columns=cols)
        out.insert(0, id_col, ids)
        return out, A.shape[0] if A.ndim == 2 and A.shape[0] == A.shape[1] else None

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
    # Solve m = N*(N-1)/2 -> N^2 - N - 2m = 0
    disc = 1 + 8*m
    rt = int(math.isqrt(disc))
    if rt*rt != disc:
        return None
    N = (1 + rt) // 2
    if N*(N-1)//2 == m:
        return N
    return None

def unvectorize_upper(vec: np.ndarray, N: int) -> np.ndarray:
    A = np.zeros((N, N), dtype=float)
    r, c = np.triu_indices(N, k=1)
    if vec.size != r.size:
        raise ValueError(f"Vector length {vec.size} != expected {r.size} for N={N}")
    A[r, c] = vec
    A[c, r] = vec
    np.fill_diagonal(A, 0.0)
    return A

# ------------------------- Harmonization helpers ----------------------- #

def learn_apply(Xtr: np.ndarray, cov_tr: pd.DataFrame,
                Xte: np.ndarray, cov_te: pd.DataFrame,
                site_col: str, age_col: Optional[str],
                mode: str, ref_batch: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Try new API, fallback to legacy 'SITE' API."""
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
    # legacy path
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

# ---------------------------------- Main -------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Harmonize connectomes and write per-subject N×N matrices.")
    ap.add_argument("--features", required=True, help="Numeric edges CSV (id + edges...) OR manifest (id, filepath).")
    ap.add_argument("--covars", required=True, help="Covariates CSV with SITE and AGE.")
    ap.add_argument("--id-col", default="subject_id")
    ap.add_argument("--site-col", default="SITE")
    ap.add_argument("--age-col", default="AGE")
    ap.add_argument("--mode", choices=["gam","anchored"], default="anchored",
                    help="ComBat-GAM with spline(age) or linear/anchored ComBat.")
    ap.add_argument("--ref-batch", default=None, help="Reference batch label in SITE, e.g. 'ADDecode:1' or '1'.")
    ap.add_argument("--nodes", type=int, default=None, help="Number of nodes (e.g., 84). If omitted, inferred from edges.")
    ap.add_argument("--cv-folds", type=int, default=0, help="0=full fit; >0 = OOF CV with N folds (stratified by SITE).")
    ap.add_argument("--cv-handle-rare", choices=["merge","drop","ignore"], default="merge",
                    help="CV only: how to handle tiny SITE classes.")
    ap.add_argument("--min-per-site", type=int, default=2, help="CV only: sites with < this many are 'rare'.")
    ap.add_argument("--save-folds", action="store_true", help="Also dump per-fold train/test matrices.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("-v","--verbose", action="count", default=1)
    args = ap.parse_args()

    setup_logging(args.verbose)
    progress = not args.no_progress

    # Quiet some noisy warnings that aren't fatal
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

    features_path = Path(args.features)
    covars_path   = Path(args.covars)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)

    banner("LOAD FEATURES")
    X_df, N_from_manifest = read_features(features_path, args.id_col, manifest_vectorize=True, progress=progress)
    ids_all = X_df[args.id_col].astype(str).values
    X_num = X_df.drop(columns=[args.id_col]).apply(pd.to_numeric, errors="coerce")
    if X_num.isnull().any().any():
        bad = int(X_num.isnull().any(axis=1).sum())
        if bad:
            raise SystemExit(f"{bad} row(s) contain NaNs in features after coercion.")

    m = X_num.shape[1]
    N = args.nodes or N_from_manifest or infer_nodes_from_edges(m)
    if N is None:
        raise SystemExit(f"Cannot infer nodes from {m} edges. Pass --nodes (e.g., --nodes 84).")
    expected_m = N*(N-1)//2
    if m != expected_m:
        raise SystemExit(f"Edge count {m} != expected upper-tri size {expected_m} for N={N}.")
    logging.info("Detected N=%d nodes (%d edges).", N, m)

    banner("LOAD COVARIATES")
    C = pd.read_csv(covars_path)
    if args.id_col not in C.columns:
        raise SystemExit(f"Covars missing id-col '{args.id_col}'.")
    # normalize types
    C[args.id_col] = C[args.id_col].astype(str)
    C[args.site_col] = C[args.site_col].astype(str).str.strip()
    if args.age_col in C.columns:
        C[args.age_col] = pd.to_numeric(C[args.age_col], errors="coerce")
        if C[args.age_col].isna().any():
            ex = C.loc[C[args.age_col].isna(), [args.id_col, args.age_col]].head(5)
            raise SystemExit(f"AGE non-numeric/missing:\n{ex}")

    # align IDs (intersection)
    keep = sorted(set(ids_all) & set(C[args.id_col].values))
    if not keep:
        raise SystemExit("No overlapping subject IDs between features and covars.")
    # reorder both by keep order
    order = pd.Index(keep)
    X = X_num.set_index(pd.Index(ids_all)).loc[order].reset_index(drop=True).values.astype(np.float32)
    C = C.set_index(args.id_col).loc[order].reset_index()

    ref_label = None if args.ref_batch is None else str(args.ref_batch).strip()
    if ref_label is not None and ref_label not in set(C[args.site_col].unique()):
        top = C[args.site_col].value_counts().head(20)
        raise SystemExit(f"--ref-batch '{ref_label}' not found in SITE. Top labels:\n{top}")

    # FULL FIT (no CV)
    if args.cv_folds <= 0:
        banner("FULL FIT")
        cov_use = C[[args.site_col] + ([args.age_col] if args.age_col in C.columns else [])].copy()
        # Fit & apply using same covars (apply is used for all rows)
        try:
            model, _ = harmonizationLearn(
                X, cov_use,
                smooth_terms=([args.age_col] if args.mode=="gam" and args.age_col in cov_use.columns else []),
                batch_col=args.site_col, ref_batch=ref_label
            )
            X_adj = harmonizationApply(X, cov_use, model)
        except TypeError as e:
            if "batch_col" not in str(e):
                raise
            cov_old = cov_use.rename(columns={args.site_col: "SITE"})
            model, _ = harmonizationLearn(
                X, cov_old,
                smooth_terms=([args.age_col] if args.mode=="gam" and args.age_col in cov_old.columns else []),
                ref_batch=ref_label
            )
            X_adj = harmonizationApply(X, cov_old, model)

        # write one matrix per subject
        mat_dir = outdir / f"{args.prefix}_matrices_full"
        mat_dir.mkdir(parents=True, exist_ok=True)
        r, c = np.triu_indices(N, k=1)
        for sid, v in pbar_iter(zip(keep, X_adj), total=len(keep), desc="Writing matrices", enable=progress):
            A = np.zeros((N,N), dtype=float)
            A[r, c] = v; A[c, r] = v; np.fill_diagonal(A, 0.0)
            np.savetxt(mat_dir / f"{sid}_harmonized.csv", A, fmt="%.6g", delimiter=",")
        print(f"Wrote {len(keep)} matrices to {mat_dir}")
        return

    # ---------------------- CV OOF (per-subject, no leakage) ------------------ #
    banner("CV OOF")
    from sklearn.model_selection import StratifiedKFold
    sites = C[args.site_col].astype(str).values
    # rare handling for CV labels only
    vc = pd.Series(sites).value_counts()
    rare = [lab for lab,cnt in vc.items() if cnt < args.min_per_site and (ref_label is None or lab != ref_label)]
    if args.cv_handle_rare == "drop" and rare:
        keep_mask = ~pd.Series(sites).isin(rare)
        X = X[keep_mask.values]
        C = C.loc[keep_mask.values].reset_index(drop=True)
        sites = C[args.site_col].astype(str).values
        keep = list(pd.Series(keep)[keep_mask.values])
        logging.info("Dropped rare sites for CV: %s", rare)
    elif args.cv_handle_rare == "merge" and rare:
        sites = np.where(pd.Series(sites).isin(rare), "OTHER", sites)

    # ensure feasible folds
    min_class = int(pd.Series(sites).value_counts().min())
    n_splits = max(2, min(args.cv_folds, min_class))
    if n_splits != args.cv_folds:
        print(f"[warn] Reducing folds from {args.cv_folds} to {n_splits} to satisfy class counts.", flush=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_dir = outdir / f"{args.prefix}_matrices_oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    if args.save_folds:
        fold_root = outdir / f"{args.prefix}_matrices_folds"
        fold_root.mkdir(parents=True, exist_ok=True)

    executed = 0
    idx_all = np.arange(len(C))

    for fold, (tr, te) in enumerate(skf.split(idx_all, sites), start=1):
        Xtr = X[tr].astype(np.float32); Xte = X[te].astype(np.float32)
        cov_tr = C.iloc[tr].copy(); cov_te = C.iloc[te].copy()

        # Covars for harmonization: SITE + AGE
        keep_cols = [args.site_col]; 
        if args.age_col in cov_tr.columns: keep_cols.append(args.age_col)
        cov_tr_use = cov_tr[keep_cols].copy()
        cov_te_use = cov_te[keep_cols].copy()
        if args.age_col in cov_tr_use.columns:
            cov_tr_use[args.age_col] = pd.to_numeric(cov_tr_use[args.age_col], errors="coerce")
            cov_te_use[args.age_col] = pd.to_numeric(cov_te_use[args.age_col], errors="coerce")
            if cov_tr_use[args.age_col].isna().any() or cov_te_use[args.age_col].isna().any():
                raise SystemExit("Non-numeric AGE encountered after split.")

        # Make sure TEST has all TRAIN batches (legacy apply quirk)
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
        Xte_adj = Xte_adj_full[:Xte.shape[0], :]  # drop dummy rows

        # Write OOF matrices for TEST subjects
        r, c = np.triu_indices(N, k=1)
        te_ids = [keep[i] for i in te]
        for sid, v in pbar_iter(zip(te_ids, Xte_adj), total=len(te_ids),
                                desc=f"Fold {fold}/{n_splits} write TEST", enable=progress):
            A = np.zeros((N,N), dtype=float)
            A[r, c] = v; A[c, r] = v; np.fill_diagonal(A, 0.0)
            np.savetxt(oof_dir / f"{sid}_harmonized.csv", A, fmt="%.6g", delimiter=",")

        # Optional fold dumps (train/test)
        if args.save_folds:
            dtr = (outdir / f"{args.prefix}_matrices_folds" / f"fold{fold:02d}_train"); dtr.mkdir(parents=True, exist_ok=True)
            dte = (outdir / f"{args.prefix}_matrices_folds" / f"fold{fold:02d}_test" ); dte.mkdir(parents=True, exist_ok=True)
            tr_ids = [keep[i] for i in tr]
            for sid, v in pbar_iter(zip(tr_ids, Xtr_adj), total=len(tr_ids),
                                    desc=f"Fold {fold}/{n_splits} write TRAIN", enable=progress):
                A = np.zeros((N,N), dtype=float)
                A[r, c] = v; A[c, r] = v; np.fill_diagonal(A, 0.0)
                np.savetxt(dtr / f"{sid}_harmonized.csv", A, fmt="%.6g", delimiter=",")
            # TEST already written to OOF dir; also copy here:
            for sid, v in zip(te_ids, Xte_adj):
                A = np.zeros((N,N), dtype=float)
                A[r, c] = v; A[c, r] = v; np.fill_diagonal(A, 0.0)
                np.savetxt(dte / f"{sid}_harmonized.csv", A, fmt="%.6g", delimiter=",")

        executed += 1

    print(f"Wrote OOF matrices for {len(oof_dir.glob('*.csv'))} subjects to {oof_dir}")
    if executed == 0:
        print("[error] 0 folds executed. Check SITE label counts or --cv-folds.", flush=True)
        sys.exit(5)


if __name__ == "__main__":
    main()
