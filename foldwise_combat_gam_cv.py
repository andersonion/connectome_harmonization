#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fold-wise ComBat / ComBat-GAM CV with optional anchoring to a reference batch.

Usage (examples):
  # Using numeric features table (rows=subjects, cols=edges), batches in SITE:
  python foldwise_combat_gam_cv.py \
    --features /path/features_numeric.csv \
    --covars   /path/covars_all.csv \
    --site-col SITE --age-col AGE --id-col subject_id \
    --folds 5 --mode gam \
    --ref-batch ADDecode:1 \
    --out-dir harmonized_cv --prefix ADDECODE_ADNI_HABS

  # Using a MANIFEST (subject_id,__id__,filepath) – auto-vectorizes (upper triangle):
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
import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# neuroHarmonize (ComBat-GAM fit/apply)
from neuroHarmonize import harmonizationLearn, harmonizationApply


# ----------------------------- Logging -------------------------------- #

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# ------------------------- Feature Loading ----------------------------- #

def read_numeric_or_manifest(features_path: Path, id_col: str, vectorize: str) -> pd.DataFrame:
    """
    Load features as a numeric table.
    If the file looks like a manifest (has 'filepath'), we load each file and vectorize.

    Returns a DataFrame with: [id_col, <numeric feature columns>]
    """
    df = pd.read_csv(features_path)
    # Manifest?
    if "filepath" in df.columns:
        logging.info("Detected MANIFEST (has 'filepath'); vectorizing per-subject files (mode=%s).", vectorize)
        need = {id_col, "filepath"}
        if not need.issubset(df.columns):
            raise SystemExit(f"Manifest must contain columns {need}. Found: {list(df.columns)}")
        return build_numeric_from_manifest(df[[id_col, "filepath"]].copy(), id_col=id_col, mode=vectorize)

    # Already numeric table: coerce and drop non-numeric columns (except id_col).
    if id_col not in df.columns:
        raise SystemExit(f"Features table missing id-col '{id_col}'. Columns: {list(df.columns)[:10]}")

    id_series = df[id_col].astype(str)
    num = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
    bad = [c for c in num.columns if num[c].isnull().any()]
    if bad:
        logging.warning("Dropping %d non-numeric/NaN columns from features: %s",
                        len(bad), ", ".join(bad[:8]) + ("..." if len(bad) > 8 else ""))
        num = num.drop(columns=bad)

    out = pd.concat([id_series, num], axis=1)
    if out.shape[1] <= 1:
        raise SystemExit("No numeric feature columns remained after coercion.")
    return out


def _read_numeric_file(fp: str) -> np.ndarray:
    # Try CSV then TSV (no header)
    try:
        return pd.read_csv(fp, header=None).values
    except Exception:
        return pd.read_csv(fp, sep="\t", header=None).values


def _vectorize_array(A: np.ndarray, mode: str) -> np.ndarray:
    A = np.asarray(A, float)
    if mode == "flatten" or A.ndim == 1 or 1 in A.shape:
        return A.ravel()
    # default: treat as square adjacency and take upper triangle without diag
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        r, c = np.triu_indices_from(A, k=1)
        return A[r, c]
    return A.ravel()


def build_numeric_from_manifest(manifest_df: pd.DataFrame, id_col: str, mode: str = "upper") -> pd.DataFrame:
    rows, ids = [], []
    first_len = None
    for _, rec in manifest_df.iterrows():
        fp = rec["filepath"]
        A = _read_numeric_file(fp)
        v = _vectorize_array(A, mode)
        if first_len is None:
            first_len = v.size
        elif v.size != first_len:
            raise SystemExit(f"Feature length mismatch at {fp}: {v.size} vs {first_len}")
        rows.append(v.astype(np.float32))
        ids.append(str(rec[id_col]))
    cols = [f"e{i:06d}" for i in range(first_len)]
    X = pd.DataFrame(rows, columns=cols)
    X.insert(0, id_col, ids)
    return X


# ------------------------------ Metrics -------------------------------- #

def site_auc_on_test(Xtr, ytr, Xte, yte) -> float:
    """
    Train a site classifier on (Xtr,ytr) and compute ROC-AUC on (Xte,yte).
    Uses predict_proba. Returns NaN if the test fold has <2 classes.
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    le = LabelEncoder()
    ytr_enc = le.fit_transform(np.asarray(ytr))
    yte_enc = le.transform(np.asarray(yte))

    te_classes = np.unique(yte_enc)
    if len(te_classes) < 2:
        return float("nan")

    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(Xtr, ytr_enc)

    prob_full = clf.predict_proba(Xte)  # shape: (n_samples, n_train_classes)

    if len(le.classes_) == 2:
        # binary: column 1 is the positive class
        return float(roc_auc_score(yte_enc, prob_full[:, 1]))
    else:
        # multiclass: subset prob columns to the classes present in y_test
        # (te_classes are encoded indices matching columns of predict_proba)
        prob_sub = prob_full[:, te_classes]
        return float(roc_auc_score(
            yte_enc,
            prob_sub,
            multi_class="ovr",
            average="macro",
            labels=te_classes
        ))


# ------------------------------ Main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Fold-wise ComBat / ComBat-GAM CV with optional anchoring to a reference batch."
    )
    ap.add_argument("--features", required=True, help="Features CSV (numeric table) or MANIFEST (with 'filepath').")
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
    ap.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")

    args = ap.parse_args()
    setup_logging(args.verbose)

    features_path = Path(args.features)
    covars_path = Path(args.covars)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    X_df = read_numeric_or_manifest(features_path, id_col=args.id_col, vectorize=args.vectorize)
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

    # Matrix for ML
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

    if ref_label is not None:
        ref_n = int(np.sum(cv_sites == ref_label))
        if ref_n < args.folds:
            logging.warning("Reference batch '%s' has only %d sample(s); "
                            "StratifiedKFold(n_splits=%d) may fail.", ref_label, ref_n, args.folds)

    # ---------------- Build CV splitter ---------------- #
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    per_fold = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X.values, cv_sites), start=1):
        Xtr = X.iloc[tr_idx].values.astype(np.float32)
        Xte = X.iloc[te_idx].values.astype(np.float32)

        cov_tr = C_df.iloc[tr_idx].copy()
        cov_te = C_df.iloc[te_idx].copy()

        # Pre-harmonization AUC (site predictability)
        pre_auc = site_auc_on_test(Xtr, cv_sites[tr_idx], Xte, cv_sites[te_idx])

        # --------------- Harmonization (fit on train, apply to test) --------------- #
        # Ensure site col is str for harmonization call
        cov_tr[args.site_col] = cov_tr[args.site_col].astype(str)
        cov_te[args.site_col] = cov_te[args.site_col].astype(str)

        if args.mode == "gam":
            # ComBat-GAM with spline(age)
            model, Xtr_adj = harmonizationLearn(
                Xtr,
                cov_tr,
                smooth_terms=[args.age_col],
                batch_col=args.site_col,
                ref_batch=ref_label
            )
            Xte_adj = harmonizationApply(Xte, cov_te, model)
        else:
            # "anchored" = no spline term; still allow anchoring to ref_batch
            model, Xtr_adj = harmonizationLearn(
                Xtr,
                cov_tr,
                smooth_terms=[],               # no spline(age)
                batch_col=args.site_col,
                ref_batch=ref_label
            )
            Xte_adj = harmonizationApply(Xte, cov_te, model)

        # Post-harmonization AUC
        post_auc = site_auc_on_test(Xtr_adj, cv_sites[tr_idx], Xte_adj, cv_sites[te_idx])

        per_fold.append({
            "fold": fold,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "pre_auc": float(pre_auc) if pre_auc == pre_auc else None,
            "post_auc": float(post_auc) if post_auc == post_auc else None,
        })
        logging.info("Fold %d: pre_auc=%.4f, post_auc=%.4f", fold,
                     per_fold[-1]["pre_auc"] or float("nan"),
                     per_fold[-1]["post_auc"] or float("nan"))

    # ---------------- Save results ---------------- #
    res = pd.DataFrame(per_fold)
    res_path = outdir / f"{args.prefix}_cv_auc.csv"
    res.to_csv(res_path, index=False)
    print(f"Wrote {res_path}")

    # Summary
    pre_mean = np.nanmean(res["pre_auc"].values.astype(float))
    post_mean = np.nanmean(res["post_auc"].values.astype(float))
    print(f"Mean AUC: pre={pre_mean:.4f} post={post_mean:.4f}")


if __name__ == "__main__":
    main()
