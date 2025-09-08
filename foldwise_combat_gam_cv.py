#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fold-wise ComBat / ComBat-GAM CV with optional anchoring to a reference batch,
with progress bars and debug banners.

Key features:
- Accepts numeric features OR a manifest (auto-vectorizes per-subject files)
- Keeps harmonization labels intact; CV-only rare-class merging/dropping
- Uses predict_proba for AUC; handles subset classes in test
- Compatible with both neuroHarmonize APIs (batch_col or legacy 'SITE')
- Ensures each test fold has ≥1 ref-batch sample; skips offending folds
- Emits clear DEBUG stats; exits non-zero if 0 folds executed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Fail fast if deps missing
from neuroHarmonize import harmonizationLearn, harmonizationApply

# Optional pretty progress
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def augment_test_for_missing_train_levels(cov_te_df: pd.DataFrame,
                                          Xte_arr: np.ndarray,
                                          site_col: str,
                                          missing_levels: list[str],
                                          age_col: str | None = None,
                                          age_fill: float | None = None):
    """
    Ensure test covars contain at least one row for every training batch level.
    Appends 1 dummy row per missing level (with age filled), and duplicates a
    baseline feature row so shapes match. Returns (cov_te_aug, Xte_aug, n_added).
    """
    if not missing_levels:
        return cov_te_df, Xte_arr, 0

    rows = []
    for lvl in missing_levels:
        row = {site_col: str(lvl)}
        if age_col and age_col in cov_te_df.columns:
            row[age_col] = age_fill if age_fill is not None else float(np.nanmedian(cov_te_df[age_col].values))
        rows.append(row)

    add_df = pd.DataFrame(rows)
    cov_te_aug = pd.concat([cov_te_df, add_df], ignore_index=True)

    n_added = len(missing_levels)
    n_feat = Xte_arr.shape[1]
    base = Xte_arr[0:1] if Xte_arr.shape[0] else np.zeros((1, n_feat), dtype=np.float32)
    add_X = np.repeat(base, n_added, axis=0)
    Xte_aug = np.vstack([Xte_arr, add_X]).astype(np.float32)
    return cov_te_aug, Xte_aug, n_added




# ----------------------------- Logging -------------------------------- #

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def iter_progress(iterable, total=None, desc="", enable=True):
    """Yield iterable with tqdm if available & enabled; else plain iterable."""
    if enable and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, ncols=90)
    return iterable


def banner(msg: str):
    print(f"\n=== {msg} ===", flush=True)


# ------------------------- Feature Loading ----------------------------- #

def _read_numeric_file(fp: str) -> np.ndarray:
    try:
        return pd.read_csv(fp, header=None).values
    except Exception:
        return pd.read_csv(fp, sep="\t", header=None).values


def _vectorize_array(A: np.ndarray, mode: str) -> np.ndarray:
    A = np.asarray(A, float)
    if mode == "auto":
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            mode = "upper"
        else:
            mode = "flatten"
    if mode == "flatten" or A.ndim == 1 or 1 in A.shape:
        return A.ravel()
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        r, c = np.triu_indices_from(A, k=1)
        return A[r, c]
    return A.ravel()


def build_numeric_from_manifest(manifest_df: pd.DataFrame, id_col: str, mode: str = "upper",
                                progress: bool = True) -> pd.DataFrame:
    rows, ids = [], []
    first_len = None
    it = iter_progress(manifest_df.itertuples(index=False), total=len(manifest_df),
                       desc="Vectorizing subjects", enable=progress)
    for rec in it:
        fp = getattr(rec, "filepath")
        sid = getattr(rec, id_col)
        A = _read_numeric_file(fp)
        v = _vectorize_array(A, mode)
        if first_len is None:
            first_len = v.size
        elif v.size != first_len:
            raise SystemExit(f"Feature length mismatch at {fp}: {v.size} vs {first_len}")
        rows.append(v.astype(np.float32))
        ids.append(str(sid))
    cols = [f"e{i:06d}" for i in range(first_len)]
    X = pd.DataFrame(rows, columns=cols)
    X.insert(0, id_col, ids)
    return X


def read_numeric_or_manifest(features_path: Path, id_col: str, vectorize: str, progress: bool) -> pd.DataFrame:
    df = pd.read_csv(features_path)
    if "filepath" in df.columns:
        logging.info("Detected MANIFEST (has 'filepath'); vectorizing per-subject files (mode=%s).", vectorize)
        need = {id_col, "filepath"}
        if not need.issubset(df.columns):
            raise SystemExit(f"Manifest must contain columns {need}. Found: {list(df.columns)}")
        return build_numeric_from_manifest(df[[id_col, "filepath"]].copy(), id_col=id_col, mode=vectorize, progress=progress)

    if id_col not in df.columns:
        raise SystemExit(f"Features table missing id-col '{id_col}'. Columns: {list(df.columns)[:10]}")

    id_series = df[id_col].astype(str)
    num = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
    bad = [c for c in num.columns if num[c].isnull().any()]
    if bad:
        logging.warning("Dropping %d non-numeric/NaN feature columns: %s",
                        len(bad), ", ".join(bad[:8]) + ("..." if len(bad) > 8 else ""))
        num = num.drop(columns=bad)
    out = pd.concat([id_series, num], axis=1)
    if out.shape[1] <= 1:
        raise SystemExit("No numeric feature columns remained after coercion.")
    return out


# ------------------------------ Metrics -------------------------------- #


def sanitize_for_auc(Xtr: np.ndarray, Xte: np.ndarray, *, tag: str = ""):
    """
    Replace NaN/Inf in Xtr/Xte with column means computed on Xtr.
    Columns that are entirely non-finite in Xtr get mean=0.0.
    Returns cleaned float32 arrays.
    """
    A_tr = np.asarray(Xtr, dtype=np.float32, order="C").copy()
    A_te = np.asarray(Xte, dtype=np.float32, order="C").copy()

    fin_tr = np.isfinite(A_tr)
    fin_te = np.isfinite(A_te)

    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(np.where(fin_tr, A_tr, np.nan), axis=0)

    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)

    # broadcast per-column means to bad entries
    A_tr = np.where(fin_tr, A_tr, col_mean)
    A_te = np.where(fin_te, A_te, col_mean)

    # optional: tiny jitter if a column becomes constant
    # (LogReg is usually fine without it; keep simple)
    return A_tr, A_te


def site_auc_on_test(Xtr: np.ndarray, ytr, Xte: np.ndarray, yte) -> float:
    """Train site classifier on (Xtr,ytr); ROC-AUC on (Xte,yte)."""
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
    prob_sub = prob_full[:, te_classes]  # subset to test classes
    return float(roc_auc_score(yte_enc, prob_sub, multi_class="ovr", average="macro"))


# ------------------------------ Main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Fold-wise ComBat / ComBat-GAM CV with optional anchoring to a reference batch."
    )
    ap.add_argument("--features", required=True, help="Features CSV (numeric) or MANIFEST (with 'filepath').")
    ap.add_argument("--covars", required=True, help="Covariates CSV.")
    ap.add_argument("--id-col", default="subject_id", help="ID column (must exist in both tables).")
    ap.add_argument("--site-col", default="SITE", help="Batch/site column in covars.")
    ap.add_argument("--age-col", default="AGE", help="Age column in covars.")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds (stratified by site).")
    ap.add_argument("--mode", choices=["gam", "anchored"], default="gam",
                    help="gam: ComBat-GAM with spline(age); anchored: linear (no spline).")
    ap.add_argument("--ref-batch", default=None,
                    help="Reference batch label (e.g., 'ADDecode:1' or '1'). Must exist in --site-col.")
    ap.add_argument("--age-min", type=float, default=None, help="Optional min age filter.")
    ap.add_argument("--age-max", type=float, default=None, help="Optional max age filter.")
    ap.add_argument("--vectorize", choices=["upper", "flatten", "auto"], default="upper",
                    help="If --features is a MANIFEST, how to vectorize per-subject files.")
    ap.add_argument("--cv-handle-rare", choices=["merge", "drop", "ignore"], default="merge",
                    help="How to handle tiny site classes for CV labels.")
    ap.add_argument("--min-per-site", type=int, default=2, help="Sites with < this many samples are 'rare'.")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-fold results.")
    ap.add_argument("--prefix", required=True, help="Prefix for outputs.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    ap.add_argument("--debug", action="store_true", help="Print detailed shapes/counts and exit codes.")
    ap.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")

    args = ap.parse_args()
    setup_logging(args.verbose)
    progress = not args.no_progress
    
    banner("LOAD")
    features_path = Path(args.features)
    covars_path = Path(args.covars)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    X_df = read_numeric_or_manifest(features_path, id_col=args.id_col, vectorize=args.vectorize, progress=progress)
    C_df = pd.read_csv(covars_path)

    if args.age_min is not None:
        C_df = C_df[C_df[args.age_col] >= args.age_min]
    if args.age_max is not None:
        C_df = C_df[C_df[args.age_col] <= args.age_max]

    X_df[args.id_col] = X_df[args.id_col].astype(str)
    C_df[args.id_col] = C_df[args.id_col].astype(str)

    keep_ids = sorted(set(X_df[args.id_col]) & set(C_df[args.id_col]))
    if not keep_ids:
        print("[error] No overlapping subject IDs between features and covars.", flush=True)
        sys.exit(2)

    X_df = X_df.set_index(args.id_col).loc[keep_ids].reset_index()
    C_df = C_df.set_index(args.id_col).loc[keep_ids].reset_index()

    # Harmonization labels


    C_df[args.site_col] = C_df[args.site_col].astype(str).str.strip()
    # If you want case-insensitive matching, uncomment the next two lines and pass an uppercased --ref-batch:
    C_df[args.site_col] = C_df[args.site_col].str.upper()
    
    # Normalize the ref label the same way
    ref_label = None if args.ref_batch is None else str(args.ref_batch).strip()
    # If you uppercased SITE above, also do:
    ref_label = ref_label.upper() if ref_label is not None else None


    if ref_label is not None and ref_label not in set(C_df[args.site_col].unique()):
        print(f"[error] --ref-batch '{ref_label}' not found in column '{args.site_col}'.", flush=True)
        uniq = C_df[args.site_col].value_counts().head(20)
        print(f"[hint] Top site labels:\n{uniq}", flush=True)
        sys.exit(2)

    # Ensure AGE numeric
    if args.age_col in C_df.columns:
        C_df[args.age_col] = pd.to_numeric(C_df[args.age_col], errors="coerce")
        bad_age = C_df[args.age_col].isna()
        if bad_age.any():
            examples = C_df.loc[bad_age, [args.id_col, args.age_col]].head(5)
            print("[error] AGE has non-numeric/missing values. Examples:\n", examples, flush=True)
            sys.exit(2)

    # Feature matrix
    X = X_df.drop(columns=[args.id_col])
    if not np.issubdtype(X.dtypes.values[0], np.number):
        X = X.apply(pd.to_numeric, errors="coerce")
    if X.isnull().any().any():
        n_bad = int(X.isnull().any(axis=1).sum())
        if n_bad:
            logging.warning("Dropping %d rows with NaNs in features after coercion.", n_bad)
            mask = ~X.isnull().any(axis=1)
            X = X.loc[mask].reset_index(drop=True)
            C_df = C_df.loc[mask].reset_index(drop=True)

    # ---------------- CV labels (only for splitting & AUC) ---------------- #
    banner("CV LABELS")
    sites_raw = C_df[args.site_col].astype(str).values
    sites_series = pd.Series(sites_raw)
    counts = sites_series.value_counts()
    rare = [lab for lab, cnt in counts.items() if cnt < args.min_per_site and lab != ref_label]

    action = "ignore"
    if args.cv_handle_rare == "drop" and rare:
        keep_mask = ~sites_series.isin(rare)
        X = X.loc[keep_mask].reset_index(drop=True)
        C_df = C_df.loc[keep_mask].reset_index(drop=True)
        cv_sites = sites_series[keep_mask].astype(str).values
        action = f"drop {sorted(rare)}"
    elif args.cv_handle_rare == "merge" and rare:
        sites_series.loc[sites_series.isin(rare)] = "OTHER"
        cv_sites = sites_series.astype(str).values
        action = f"merge {sorted(rare)} -> OTHER"
    else:
        cv_sites = sites_series.astype(str).values

    print(f"[debug] subjects: {len(C_df)} | unique sites: {C_df[args.site_col].nunique()} | "
          f"ref='{ref_label}' count: {int((C_df[args.site_col]==ref_label).sum()) if ref_label else 0} | "
          f"rare-handling: {action}", flush=True)

    # ---------------- Build CV splitter (ensure ref in each test fold) ---- #
    banner("CV SPLIT")
    n_splits = int(args.folds)
    if ref_label is not None:
        ref_n = int(pd.Series(cv_sites).value_counts().get(ref_label, 0))
        n_splits = min(n_splits, ref_n)  # each test fold needs ≥1 ref sample
    if n_splits < 2:
        print(f"[error] Not enough '{ref_label}' samples for CV (need ≥2).", flush=True)
        sys.exit(3)
    if n_splits != args.folds:
        print(f"[warn] Reducing folds from {args.folds} to {n_splits} so every test fold has ≥1 '{ref_label}' sample.",
              flush=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    per_fold: List[dict] = []
    fold_iter = skf.split(X.values, cv_sites)
    pbar = tqdm(total=n_splits, desc="CV folds", ncols=90) if (progress and tqdm is not None) else None

    executed = 0
    for fold, (tr_idx, te_idx) in enumerate(fold_iter, start=1):
        if pbar is None:
            print(f"[{fold}/{n_splits}] start", flush=True)

        Xtr = X.iloc[tr_idx].values.astype(np.float32)
        Xte = X.iloc[te_idx].values.astype(np.float32)

        cov_tr = C_df.iloc[tr_idx].copy()
        cov_te = C_df.iloc[te_idx].copy()

        # Skip fold if test set lacks the reference batch
        if ref_label is not None:
            te_has_ref = (cov_te[args.site_col].astype(str) == ref_label).any()
            if not te_has_ref:
                if pbar is None:
                    print(f"[{fold}/{n_splits}] skip: no '{ref_label}' in TEST", flush=True)
                else:
                    pbar.set_postfix_str("skip: no ref in test"); pbar.update(1)
                continue

        # Pre-harmonization AUC
        if pbar is not None: pbar.set_postfix_str("AUC pre")
        Xtr_pre, Xte_pre = sanitize_for_auc(Xtr, Xte, tag="pre")
        pre_auc = site_auc_on_test(Xtr_pre, cv_sites[tr_idx], Xte_pre, cv_sites[te_idx])


        # Harmonization covariates: batch + AGE (exclude SEX to avoid numeric-cast)
        cov_tr0 = cov_tr.copy(); cov_te0 = cov_te.copy()
        cov_tr0[args.site_col] = cov_tr0[args.site_col].astype(str)
        cov_te0[args.site_col] = cov_te0[args.site_col].astype(str)
        keep_cols = [args.site_col]
        if args.age_col in cov_tr0.columns:
            keep_cols.append(args.age_col)
        cov_tr_use = cov_tr0[keep_cols].copy()
        cov_te_use = cov_te0[keep_cols].copy()
        if args.age_col in cov_tr_use.columns:
            cov_tr_use[args.age_col] = pd.to_numeric(cov_tr_use[args.age_col], errors="coerce")
            cov_te_use[args.age_col] = pd.to_numeric(cov_te_use[args.age_col], errors="coerce")
            if cov_tr_use[args.age_col].isna().any() or cov_te_use[args.age_col].isna().any():
                print("[error] Non-numeric AGE encountered after split; check covariates.", flush=True)
                sys.exit(4)

        # Harmonization (new API, else legacy fallback)
        def _fit_apply(Xtr_arr, cov_tr_df, Xte_arr, cov_te_df, smooth_terms, site_col, ref_label_):
            try:
                model, Xtr_adj_ = harmonizationLearn(
                    Xtr_arr, cov_tr_df, smooth_terms=smooth_terms,
                    batch_col=site_col, ref_batch=ref_label_
                )
                Xte_adj_ = harmonizationApply(Xte_arr, cov_te_df, model)
                return Xtr_adj_, Xte_adj_
            except TypeError as e:
                if "batch_col" not in str(e):
                    raise
            cov_tr_old = cov_tr_df.rename(columns={site_col: "SITE"})
            cov_te_old = cov_te_df.rename(columns={site_col: "SITE"})
            model, Xtr_adj_ = harmonizationLearn(
                Xtr_arr, cov_tr_old, smooth_terms=smooth_terms, ref_batch=ref_label_
            )
            Xte_adj_ = harmonizationApply(Xte_arr, cov_te_old, model)
            return Xtr_adj_, Xte_adj_

        if pbar is not None: pbar.set_postfix_str("harmonizing")
        
        # Determine training batch levels and missing ones in TEST
        train_levels = sorted(cov_tr_use[args.site_col].astype(str).unique().tolist())
        test_levels = set(cov_te_use[args.site_col].astype(str))
        missing_levels = [lvl for lvl in train_levels if lvl not in test_levels]
        
        # Fill value for AGE (if present)
        age_fill = None
        if args.age_col in cov_tr_use.columns:
            age_fill = float(np.nanmedian(cov_tr_use[args.age_col].values))
        
        # Augment TEST covars/features to include all training levels
        cov_te_aug, Xte_aug, n_added = augment_test_for_missing_train_levels(
            cov_te_use, Xte, args.site_col, missing_levels,
            age_col=(args.age_col if args.age_col in cov_tr_use.columns else None),
            age_fill=age_fill
        )
        if n_added:
            print(f"[info] Fold {fold}: added {n_added} dummy row(s) to TEST for missing batches: {missing_levels}", flush=True)
        
        # Fit/apply (with possibly augmented TEST)
        smooth_terms = [args.age_col] if (args.mode == "gam" and args.age_col in cov_tr_use.columns) else []
        Xtr_adj, Xte_adj_full = _fit_apply(
            Xtr, cov_tr_use,
            Xte_aug, cov_te_aug,
            smooth_terms=smooth_terms,
            site_col=args.site_col,
            ref_label_=ref_label
        )
        
        # Drop dummy rows from adjusted TEST
        Xte_adj = Xte_adj_full[:Xte.shape[0], :]


        if pbar is not None: pbar.set_postfix_str("AUC post")
        Xtr_post, Xte_post = sanitize_for_auc(Xtr_adj, Xte_adj, tag="post")
        post_auc = site_auc_on_test(Xtr_post, cv_sites[tr_idx], Xte_post, cv_sites[te_idx])


        per_fold.append({
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "pre_auc": float(pre_auc) if pre_auc == pre_auc else None,
            "post_auc": float(post_auc) if post_auc == post_auc else None,
        })
        executed += 1

        if pbar is not None:
            pbar.update(1)
        else:
            print(f"[{fold}/{n_splits}] done: pre_auc={per_fold[-1]['pre_auc']}, "
                  f"post_auc={per_fold[-1]['post_auc']}", flush=True)

    if tqdm is not None and pbar is not None:
        pbar.close()

    banner("RESULTS")
    res = pd.DataFrame(per_fold)
    res_path = Path(args.out_dir) / f"{args.prefix}_cv_auc.csv"
    res.to_csv(res_path, index=False)
    print(f"Wrote {res_path}", flush=True)

    if executed == 0:
        print("[error] 0 folds executed. Likely every test fold lacked the reference batch "
              f"('{ref_label}') or stratification was impossible with current settings.\n"
              "Try: --folds 3  OR  use --site-col/BATCH consistent with --ref-batch  OR  "
              "--cv-handle-rare merge --min-per-site <larger>.", flush=True)
        sys.exit(5)

    pre_mean = np.nanmean(res["pre_auc"].values.astype(float)) if len(res) else float("nan")
    post_mean = np.nanmean(res["post_auc"].values.astype(float)) if len(res) else float("nan")
    print(f"Mean AUC: pre={pre_mean:.4f} post={post_mean:.4f}", flush=True)


if __name__ == "__main__":
    main()
