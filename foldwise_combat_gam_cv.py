#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fold-wise ComBat / ComBat-GAM CV with optional anchoring to a reference batch,
now with progress bars (uses tqdm if installed, else simple prints).

Examples:

# Numeric features (rows=subjects, cols=edges), batches in SITE (ADDecode=1):
python foldwise_combat_gam_cv.py \
  --features /path/features_numeric.csv \
  --covars   /path/covars_all.csv \
  --site-col SITE --age-col AGE --id-col subject_id \
  --folds 5 --mode gam \
  --ref-batch 1 \
  --out-dir harmonized_cv --prefix ADDECODE_ADNI_HABS

# MANIFEST (subject_id,__id__,filepath) – auto-vectorize (upper triangle):
python foldwise_combat_gam_cv.py \
  --features /path/features_manifest.csv \
  --covars   /path/covars_all.csv \
  --site-col BATCH --age-col AGE --id-col subject_id \
  --folds 5 --mode gam \
  --ref-batch ADDecode:1 \
  --vectorize upper \
  --out-dir harmonized_cv --prefix ADDECODE_ADNI_HABS
"""

from __future__ import annotations

import argparse
import logging
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


# ----------------------------- Logging -------------------------------- #

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def iter_progress(iterable, total=None, desc="", enable=True):
    """Yield iterable with tqdm if available & enabled; else plain iterable."""
    if enable and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, ncols=90)
    return iterable


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

def site_auc_on_test(Xtr: np.ndarray, ytr, Xte: np.ndarray, yte) -> float:
    """
    Train a site classifier on (Xtr,ytr) and compute ROC-AUC on (Xte,yte).
    Uses predict_proba. Returns NaN if the test fold has <2 classes.
    Handles test-time subset of training classes.
    """
    ytr = np.asarray(ytr)
    yte = np.asarray(yte)

    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    yte_enc = le.transform(yte)

    te_classes = np.unique(yte_enc)
    if len(te_classes) < 2:
        return float("nan")

    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(Xtr, ytr_enc)

    prob_full = clf.predict_proba(Xte)  # (n_samples, n_train_classes)
    if len(le.classes_) == 2:
        return float(roc_auc_score(yte_enc, prob_full[:, 1]))
    else:
        # subset prob columns to the classes present in test (encoded indices)
        prob_sub = prob_full[:, te_classes]
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
                    help="Reference batch label to anchor to (e.g., 'ADDecode:1' or '1').")
    ap.add_argument("--age-min", type=float, default=None, help="Optional min age filter.")
    ap.add_argument("--age-max", type=float, default=None, help="Optional max age filter.")
    ap.add_argument("--vectorize", choices=["upper", "flatten", "auto"], default="upper",
                    help="If --features is a MANIFEST, how to vectorize per-subject files.")
    ap.add_argument("--cv-handle-rare", choices=["merge", "drop", "ignore"], default="merge",
                    help="How to handle tiny site classes for CV labels (does not touch harmonization labels).")
    ap.add_argument("--min-per-site", type=int, default=2, help="Sites with < this many samples are 'rare'.")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-fold results.")
    ap.add_argument("--prefix", required=True, help="Prefix for outputs.")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars/logging steps.")
    ap.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")

    args = ap.parse_args()
    setup_logging(args.verbose)
    progress = not args.no_progress

    features_path = Path(args.features)
    covars_path = Path(args.covars)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    X_df = read_numeric_or_manifest(features_path, id_col=args.id_col, vectorize=args.vectorize, progress=progress)
    C_df = pd.read_csv(covars_path)

    # Optional age filter
    if args.age_min is not None:
        C_df = C_df[C_df[args.age_col] >= args.age_min]
    if args.age_max is not None:
        C_df = C_df[C_df[args.age_col] <= args.age_max]

    # Align by id
    X_df[args.id_col] = X_df[args.id_col].astype(str)
    C_df[args.id_col] = C_df[args.id_col].astype(str)

    keep_ids = sorted(set(X_df[args.id_col]) & set(C_df[args.id_col]))
    if not keep_ids:
        raise SystemExit("No overlapping subject IDs between features and covars after loading.")

    X_df = X_df.set_index(args.id_col).loc[keep_ids].reset_index()
    C_df = C_df.set_index(args.id_col).loc[keep_ids].reset_index()

    # Harmonization labels (batch) as strings; DO NOT TOUCH for harmonization
    C_df[args.site_col] = C_df[args.site_col].astype(str)
    ref_label = None if args.ref_batch is None else str(args.ref_batch)
    if ref_label is not None and ref_label not in set(C_df[args.site_col].unique()):
        raise SystemExit(f"--ref-batch '{ref_label}' not found in {args.site_col} values.")

    # Ensure AGE is numeric (fail fast with helpful message)
    if args.age_col in C_df.columns:
        C_df[args.age_col] = pd.to_numeric(C_df[args.age_col], errors="coerce")
        bad_age = C_df[args.age_col].isna()
        if bad_age.any():
            n_bad = int(bad_age.sum())
            examples = C_df.loc[bad_age, [args.id_col, args.age_col]].head(5).to_dict(orient="records")
            raise SystemExit(
                f"AGE column contains non-numeric or missing values in {n_bad} row(s). "
                f"Examples: {examples}. Fix covars or choose a correct --age-col."
            )

    # Feature matrix for ML
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
    sites_raw = C_df[args.site_col].astype(str).values
    sites_series = pd.Series(sites_raw)
    counts = sites_series.value_counts()
    rare = [lab for lab, cnt in counts.items() if cnt < args.min_per_site and lab != ref_label]

    if args.cv_handle_rare == "drop" and rare:
        keep_mask = ~sites_series.isin(rare)
        logging.info("Dropping %d subjects from rare CV classes (<%d): %s",
                     int((~keep_mask).sum()), args.min_per_site, sorted(rare))
        X = X.loc[keep_mask].reset_index(drop=True)
        C_df = C_df.loc[keep_mask].reset_index(drop=True)
        cv_sites = sites_series[keep_mask].astype(str).values
    elif args.cv_handle_rare == "merge" and rare:
        logging.info("Merging rare CV classes (<%d) into 'OTHER': %s",
                     args.min_per_site, sorted(rare))
        sites_series.loc[sites_series.isin(rare)] = "OTHER"
        cv_sites = sites_series.astype(str).values
    else:
        cv_sites = sites_series.astype(str).values

    # ---------------- Build CV splitter (ensure ref in each test fold) ---- #
    n_splits = int(args.folds)
    if ref_label is not None:
        ref_n = int(pd.Series(cv_sites).value_counts().get(ref_label, 0))
        n_splits = min(n_splits, ref_n)  # each test fold needs ≥1 ref sample
    if n_splits < 2:
        raise SystemExit(
            f"Not enough '{ref_label}' samples for CV: have {pd.Series(cv_sites).value_counts().get(ref_label,0)}, "
            f"need at least 2 to make ≥2 folds. Reduce --folds or add data."
        )
    if n_splits != args.folds:
        print(f"[warn] Reducing folds from {args.folds} to {n_splits} so every test fold has "
              f"at least one '{ref_label}' sample.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    per_fold: List[dict] = []
    fold_iter = skf.split(X.values, cv_sites)
    # Progress over folds
    pbar = tqdm(total=n_splits, desc="CV folds", ncols=90) if (progress and tqdm is not None) else None

    for fold, (tr_idx, te_idx) in enumerate(fold_iter, start=1):
        if pbar is None:
            print(f"[{fold}/{n_splits}] starting fold...", flush=True)

        Xtr = X.iloc[tr_idx].values.astype(np.float32)
        Xte = X.iloc[te_idx].values.astype(np.float32)

        cov_tr = C_df.iloc[tr_idx].copy()
        cov_te = C_df.iloc[te_idx].copy()

        # Skip fold if test set lacks the reference batch (belt & suspenders)
        if ref_label is not None:
            te_has_ref = (cov_te[args.site_col].astype(str) == ref_label).any()
            if not te_has_ref:
                if pbar is None:
                    print(f"[{fold}/{n_splits}] no '{ref_label}' in TEST; skipping.", flush=True)
                else:
                    pbar.set_postfix_str("skip: no ref in test")
                    pbar.update(1)
                continue

        # Pre-harmonization AUC (site predictability)
        if pbar is not None:
            pbar.set_postfix_str("AUC pre")
        pre_auc = site_auc_on_test(Xtr, cv_sites[tr_idx], Xte, cv_sites[te_idx])

        # --------------- Harmonization (fit on train, apply to test) --------------- #
        if pbar is not None:
            pbar.set_postfix_str("harmonizing")

        cov_tr0 = cov_tr.copy()
        cov_te0 = cov_te.copy()
        cov_tr0[args.site_col] = cov_tr0[args.site_col].astype(str)
        cov_te0[args.site_col] = cov_te0[args.site_col].astype(str)

        # Keep only the needed covariates for harmonization: batch + AGE (exclude SEX to avoid numeric-cast)
        keep_cols = [args.site_col]
        if args.age_col in cov_tr0.columns:
            keep_cols.append(args.age_col)
        cov_tr_use = cov_tr0[keep_cols].copy()
        cov_te_use = cov_te0[keep_cols].copy()
        if args.age_col in cov_tr_use.columns:
            cov_tr_use[args.age_col] = pd.to_numeric(cov_tr_use[args.age_col], errors="coerce")
            cov_te_use[args.age_col] = pd.to_numeric(cov_te_use[args.age_col], errors="coerce")
            if cov_tr_use[args.age_col].isna().any() or cov_te_use[args.age_col].isna().any():
                raise SystemExit("Non-numeric AGE encountered after split; check covariates.")

        def _fit_apply(Xtr_arr, cov_tr_df, Xte_arr, cov_te_df, smooth_terms, site_col, ref_label_):
            # Try NEW API (supports batch_col=...)
            try:
                model, Xtr_adj_ = harmonizationLearn(
                    Xtr_arr,
                    cov_tr_df,
                    smooth_terms=smooth_terms,
                    batch_col=site_col,
                    ref_batch=ref_label_
                )
                Xte_adj_ = harmonizationApply(Xte_arr, cov_te_df, model)
                return Xtr_adj_, Xte_adj_
            except TypeError as e:
                if "batch_col" not in str(e):
                    raise  # some other error; surface it

            # OLD API fallback: rename bat
