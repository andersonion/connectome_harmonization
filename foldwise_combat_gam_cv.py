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
    C_df[args.site_col] = C_df[args.site_col].astype(str)
    ref_label = None if args.ref_batch is None else str(args.ref_batch)
    if ref_label is not None and ref_label not in set(C_df[args.site_col].unique()):
        print(f"[error] --ref-batch '{ref_label}' not found in column '{args.site_col}'.", flush=True)
        uniq = C_df[args.site_col].value_counts().head]()_
