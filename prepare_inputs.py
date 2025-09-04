#!/usr/bin/env python3
"""
Prepare harmonization inputs:
- features.csv  (subjects x edges; index=subject_id)
- covariates.cleaned.csv (subject_id,SITE,AGE,[...])
- folds.csv (subject_id,fold)
- meta.json (basic provenance)

Supported connectome inputs:
  A) --connectomes <DIR>  : per-subject files (.csv/.tsv/.npy), square N×N
     - subject_id is inferred from filename stem unless you give --subjects_csv
     - with --subjects_csv, provide columns: subject_id,file (relative to DIR)

  B) --connectomes <FILE.npy> : 3D array shape (S,N,N); must also give --subjects_csv
     with column subject_id in the order of the first dimension

  C) --connectomes <FILE.csv/tsv> : per-subject adjacency flattened per row (S x N*N)
     must also give --matrix_size N to reshape per row back to N×N

Usage examples:
  python prepare_inputs.py \
      --connectomes data/mats/ \
      --covars data/covariates.csv \
      --outdir build/ \
      --folds 5

  python prepare_inputs.py \
      --connectomes data/all_subjects.npy \
      --subjects_csv data/subjects.csv \
      --covars data/covariates.csv \
      --outdir build/

  python prepare_inputs.py \
      --connectomes data/adj_flat.csv \
      --matrix_size 200 \
      --covars data/covariates.csv \
      --outdir build/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# ----------------------------
# IO utilities
# ----------------------------
# --- ID canonicalization helpers ---
import logging

def build_id_canonicalizer(pattern=None, case_sensitive=False):
    """
    Returns a function that extracts a canonical subject ID from any longer string.
    If `pattern` is provided, it must have exactly one capturing group (group 1).
    Applied to BOTH covariates IDs and feature IDs (from filenames or tables).
    """
    rx_list = []
    if pattern:
        rx_list.append(re.compile(pattern))

    # sensible defaults: BIDS-ish, H####_y# form, fallback to first token
    rx_list += [
        re.compile(r'^(sub-[A-Za-z0-9]+(?:[_-]ses-[A-Za-z0-9]+)?)'),  # sub-XXX[_ses-YY]
        re.compile(r'^([A-Za-z0-9]+[_-]y[0-9]+)'),                   # H1234_y2 or H1234-y2
        re.compile(r'^([A-Za-z0-9]+)')                               # fallback: first token
    ]

    def _canon(x):
        s = "" if x is None else str(x)
        for rx in rx_list:
            m = rx.match(s)
            if m:
                s = m.group(1)
                break
        # normalize whitespace
        s = s.strip()
        if not case_sensitive:
            s = s.lower()
        return s
    return _canon


def load_clean_covars(covars_path, id_col="subject_id", site_col="SITE", age_col="AGE",
                      dropna=True, id_extractor=None, case_sensitive=False):
    import pandas as pd

    df = pd.read_csv(covars_path)
    # Create a canonical ID column that strips any extensions in subject_id
    canon = build_id_canonicalizer(pattern=id_extractor, case_sensitive=case_sensitive)
    df["__id__"] = df[id_col].astype(str).map(canon)

    # optional: enforce dtypes / dropna as you already do
    if dropna:
        df = df.dropna(subset=[site_col, age_col, id_col])

    return df



def _read_matrix(path: Path) -> np.ndarray:
    """Read a single square adjacency matrix from .csv/.tsv/.npy (returns 2D np.ndarray)."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{path} is not a square 2D array")
        return arr
    elif suffix in [".csv", ".tsv"]:
        sep = "," if suffix == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, header=None)
        arr = df.to_numpy()
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{path} is not a square matrix; got {arr.shape}")
        return arr
    else:
        raise ValueError(f"Unsupported file format for matrix: {path.name}")


def _extract_upper_tri(mat: np.ndarray, include_diagonal: bool = False) -> np.ndarray:
    """Vectorize upper triangle of a square matrix."""
    n = mat.shape[0]
    k = 0 if include_diagonal else 1
    idx = np.triu_indices(n, k=k)
    return mat[idx]


def _vectorize_stack(mats: List[np.ndarray], include_diagonal: bool = False) -> Tuple[np.ndarray, int]:
    """Stack feature vectors from list of square matrices. Returns (S x E, N) and N (nodes)."""
    if not mats:
        raise ValueError("No matrices to vectorize.")
    n = mats[0].shape[0]
    for m in mats:
        if m.shape != (n, n):
            raise ValueError(f"Inconsistent matrix shapes: expected {(n,n)} got {m.shape}")
    vecs = [_extract_upper_tri(m, include_diagonal) for m in mats]
    X = np.vstack(vecs)
    return X, n


# ----------------------------
# Loading strategies
# ----------------------------

def load_connectomes_from_dir(
    dirpath: Path,
    subjects_csv: Optional[Path],
    include_diagonal: bool = False,
) -> Tuple[pd.DataFrame, List[str], int]:
    """
    Load per-subject matrices from a directory.
    If subjects_csv is provided, it must have columns: subject_id,file (relative to dir).
    Else, subject_id is inferred from filename stem (strip non-word chars).
    Returns (features_df, ordered_ids, N_nodes).
    """
    files: List[Path] = []
    ids: List[str] = []

    if subjects_csv:
        mapdf = pd.read_csv(subjects_csv)
        if not {"subject_id", "file"}.issubset(mapdf.columns):
            raise ValueError("--subjects_csv must have columns: subject_id,file")
        for _, row in mapdf.iterrows():
            sid = str(row["subject_id"])
            f = dirpath / str(row["file"])
            if not f.exists():
                raise FileNotFoundError(f"Listed file not found: {f}")
            files.append(f)
            ids.append(sid)
    else:
        # Infer subject_id from filename stem
        for f in sorted(dirpath.iterdir()):
            if f.suffix.lower() not in (".npy", ".csv", ".tsv"):
                continue
            sid = re.sub(r"\W+", "_", f.stem).strip("_")
            files.append(f)
            ids.append(sid)

    mats = [_read_matrix(p) for p in files]
    X, n_nodes = _vectorize_stack(mats, include_diagonal)
    # Name features e1..eE
    e = X.shape[1]
    cols = [f"e{i+1}" for i in range(e)]
    feats = pd.DataFrame(X, index=ids, columns=cols)
    feats.index.name = "subject_id"
    return feats, ids, n_nodes


def load_connectomes_from_npy(
    npyfile: Path,
    subjects_csv: Path,
    include_diagonal: bool = False,
) -> Tuple[pd.DataFrame, List[str], int]:
    """Load a 3D npy (S,N,N) and subjects list."""
    arr = np.load(npyfile)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"{npyfile} must be shape (S,N,N); got {arr.shape}")

    sids = pd.read_csv(subjects_csv)["subject_id"].astype(str).tolist()
    if len(sids) != arr.shape[0]:
        raise ValueError("subjects_csv length does not match first dimension of npy")

    mats = [arr[i] for i in range(arr.shape[0])]
    X, n_nodes = _vectorize_stack(mats, include_diagonal)
    cols = [f"e{i+1}" for i in range(X.shape[1])]
    feats = pd.DataFrame(X, index=sids, columns=cols)
    feats.index.name = "subject_id"
    return feats, sids, n_nodes


def load_connectomes_from_flat(
    flatfile: Path,
    matrix_size: int,
    include_diagonal: bool = False,
) -> Tuple[pd.DataFrame, List[str], int]:
    """Load a CSV/TSV with each row = flattened N*N adjacency (with header 'subject_id' + values)."""
    sep = "," if flatfile.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(flatfile, sep=sep)
    if "subject_id" not in df.columns:
        raise ValueError("Flat file must include a 'subject_id' column.")

    sids = df["subject_id"].astype(str).tolist()
    vals = df.drop(columns=["subject_id"]).to_numpy()
    S = vals.shape[0]
    expected = matrix_size * matrix_size
    if vals.shape[1] != expected:
        raise ValueError(f"Expected {expected} values per row (N*N), got {vals.shape[1]}")

    mats = [vals[i].reshape(matrix_size, matrix_size) for i in range(S)]
    X, n_nodes = _vectorize_stack(mats, include_diagonal)
    cols = [f"e{i+1}" for i in range(X.shape[1])]
    feats = pd.DataFrame(X, index=sids, columns=cols)
    feats.index.name = "subject_id"
    return feats, sids, n_nodes


# ----------------------------
# Covariate handling
# ----------------------------

def load_clean_covars(
    covars_path: Path,
    id_col: str = "subject_id",
    site_col: str = "SITE",
    age_col: str = "AGE",
    dropna: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(covars_path)
    # Standardize required columns
    required = {id_col, site_col, age_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required covariate columns: {sorted(missing)}")

    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    # AGE to numeric
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
    # SITE as string (categorical later)
    df[site_col] = df[site_col].astype(str)

    # Optional cleaning: trim spaces from all string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    # Handle NaNs
    if dropna:
        before = len(df)
        df = df.dropna(subset=[id_col, site_col, age_col])
        after = len(df)
        if after < before:
            print(f"[warn] Dropped {before - after} rows with missing {id_col}/{site_col}/{age_col}.", file=sys.stderr)

    # Reindex by subject_id
    df = df.set_index(id_col, drop=False)
    return df


def align_subjects(features: pd.DataFrame, covars: pd.DataFrame, id_col: str = "subject_id") -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Intersect subjects present in both; keep aligned order."""
    common = features.index.intersection(covars.index)
    if len(common) == 0:
        raise ValueError("No overlapping subject IDs between features and covariates.")
    if len(common) < len(features):
        print(f"[warn] {len(features)-len(common)} subjects in features missing in covariates; dropped.", file=sys.stderr)
    if len(common) < len(covars):
        print(f"[warn] {len(covars)-len(common)} subjects in covariates missing in features; dropped.", file=sys.stderr)
    feats2 = features.loc[common].copy()
    cov2 = covars.loc[common].copy()
    # Ensure exact same sorted order
    feats2 = feats2.sort_index()
    cov2 = cov2.sort_index()
    return feats2, cov2, common.tolist()


# ----------------------------
# Folds
# ----------------------------

def make_folds(covars: pd.DataFrame, site_col: str, n_splits: int, random_state: int = 42) -> pd.DataFrame:
    """StratifiedKFold by SITE -> returns DataFrame(subject_id, fold)."""
    sids = covars["subject_id"].astype(str).to_numpy()
    y = covars[site_col].astype(str).to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = np.empty(len(sids), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros_like(y), y), start=1):
        folds[test_idx] = fold_idx
    return pd.DataFrame({"subject_id": sids, "fold": folds})


# ----------------------------
# Main
# ----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Construct harmonization inputs (features, covars, folds).")
    ap.add_argument("--connectomes", required=True, help="Directory with per-subject matrices OR .npy (S,N,N) OR flat CSV/TSV.")
    ap.add_argument("--subjects_csv", default=None, help="CSV mapping for connectomes (subject_id,file) OR ordered subject_id for .npy.")
    ap.add_argument("--matrix_size", type=int, default=None, help="Required if --connectomes is a flat CSV/TSV of N*N per row.")
    ap.add_argument("--covars", required=True, help="Covariates CSV with subject_id,SITE,AGE,[optional columns].")
    ap.add_argument("--outdir", required=True, help="Output directory for features.csv, covariates.cleaned.csv, folds.csv, meta.json.")
    ap.add_argument("--id_col", default="subject_id")
    ap.add_argument("--site_col", default="SITE")
    ap.add_argument("--age_col", default="AGE")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--include_diagonal", action="store_true", help="Include diagonal in feature vector (default off).")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conn = Path(args.connectomes)
    subjects_csv = Path(args.subjects_csv) if args.subjects_csv else None

    # Load features
    if conn.is_dir():
        feats, ids, n_nodes = load_connectomes_from_dir(conn, subjects_csv, include_diagonal=args.include_diagonal)
    elif conn.suffix.lower() == ".npy":
        if subjects_csv is None:
            raise SystemExit("--subjects_csv with subject_id required for .npy input (defines subject order).")
        feats, ids, n_nodes = load_connectomes_from_npy(conn, subjects_csv, include_diagonal=args.include_diagonal)
    elif conn.suffix.lower() in (".csv", ".tsv"):
        if not args.matrix_size:
            raise SystemExit("--matrix_size is required when connectomes is a flat CSV/TSV.")
        feats, ids, n_nodes = load_connectomes_from_flat(conn, args.matrix_size, include_diagonal=args.include_diagonal)
    else:
        raise SystemExit(f"Unsupported --connectomes input: {conn}")

    # Load & clean covars
    cov = load_clean_covars(
        Path(args.covars),
        id_col=args.id_col,
        site_col=args.site_col,
        age_col=args.age_col,
        dropna=True,
        id_extractor=getattr(args, "id_extractor", None),
        case_sensitive=getattr(args, "case_sensitive", False),
    )

    # Align subjects
    feats2, cov2, kept = align_subjects(feats, cov, id_col=args.id_col)

    # Folds
    folds_df = make_folds(cov2, site_col=args.site_col, n_splits=args.folds)

    # Write outputs
    features_path = outdir / "features.csv"
    covars_path = outdir / "covariates.cleaned.csv"
    folds_path = outdir / "folds.csv"
    meta_path = outdir / "meta.json"

    feats2.to_csv(features_path, index=True)
    cov2.to_csv(covars_path, index=False)
    folds_df.to_csv(folds_path, index=False)

    meta: Dict[str, object] = {
        "n_subjects": int(len(kept)),
        "n_nodes": int(n_nodes),
        "n_edges": int(feats2.shape[1]),
        "include_diagonal": bool(args.include_diagonal),
        "id_col": args.id_col,
        "site_col": args.site_col,
        "age_col": args.age_col,
        "folds": int(args.folds),
        "sources": {
            "connectomes": str(conn),
            "covars": str(Path(args.covars)),
            "subjects_csv": str(subjects_csv) if subjects_csv else None,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[ok] Wrote: {features_path}")
    print(f"[ok] Wrote: {covars_path}")
    print(f"[ok] Wrote: {folds_path}")
    print(f"[ok] Wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
