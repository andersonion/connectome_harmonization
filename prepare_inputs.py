#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construct harmonization inputs (features, covars, folds) robustly.

- Canonicalizes subject IDs on BOTH sides using a shared extractor.
- Drops non-overlapping subjects (logs counts) instead of crashing.
- Accepts features as a directory of files OR a table.
- Produces:
    <outdir>/covars_aligned.csv
    <outdir>/features_aligned.csv               (if features is a table)
    <outdir>/features_manifest_aligned.csv      (if features is a directory)
    <outdir>/fold_assignments.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import pandas as pd
import numpy as np


# ------------------------------- Logging ------------------------------------ #

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# --------------------------- ID Canonicalization ----------------------------- #

def build_id_canonicalizer(pattern: str | None = None,
                           case_sensitive: bool = False) -> Callable[[str], str]:
    """
    Returns a function that extracts a canonical subject ID from a longer string.
    If `pattern` is provided, it must have exactly ONE capturing group (group 1).
    Applied uniformly to BOTH covars and features.

    Examples:
      pattern='^([A-Za-z0-9]+_y[0-9]+)'   # turns 'H4567_y2_conn_plain' -> 'H4567_y2'
      pattern='^(sub-[A-Za-z0-9]+(?:[_-]ses-[A-Za-z0-9]+)?)'  # BIDS-ish
    """
    rx_list: List[re.Pattern] = []
    if pattern:
        try:
            rx_list.append(re.compile(pattern))
        except re.error as e:
            raise SystemExit(f"Invalid --id-extractor regex: {e}")

    # Sensible fallbacks if no pattern or no match with the provided pattern
    rx_list += [
        re.compile(r'^(sub-[A-Za-z0-9]+(?:[_-]ses-[A-Za-z0-9]+)?)'),  # sub-xxx[_ses-yy]
        re.compile(r'^([A-Za-z0-9]+[_-]y[0-9]+)'),                    # H1234_y2 or H1234-y2
        re.compile(r'^([A-Za-z0-9]+)'),                               # fallback: first token
    ]

    def _canon(x: str | int | float | None) -> str:
        s = "" if x is None else str(x)
        for rx in rx_list:
            m = rx.match(s)
            if m:
                s = m.group(1)
                break
        s = s.strip()
        if not case_sensitive:
            s = s.lower()
        return s

    return _canon


# ------------------------------ CSV Reading --------------------------------- #

def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Robust CSV read:
    - try pandas default C engine
    - on ParserError, fallback to Python engine
    - strip NULs and CRs on the fly
    """
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        # Attempt to sanitize stream and re-read
        logging.warning("ParserError on %s; retrying with Python engine after sanitization.", path)
        with open(path, "rb") as f:
            data = f.read().replace(b"\x00", b"")
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        from io import StringIO
        return pd.read_csv(StringIO(text), engine="python")


def _read_numeric(fp: str) -> np.ndarray:
    """Read a per-subject file as a numeric array. Tries CSV then TSV."""
    import pandas as pd
    try:
        return pd.read_csv(fp, header=None).values
    except Exception:
        return pd.read_csv(fp, sep="\t", header=None).values

def _vectorize_array(A: np.ndarray, mode: str) -> np.ndarray:
    A = np.asarray(A, float)
    if mode == "flatten" or A.ndim == 1 or 1 in A.shape:
        return A.ravel()
    # assume square adjacency and take upper triangle without diagonal
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        r, c = np.triu_indices_from(A, k=1)
        return A[r, c]
    # fallback: flatten
    return A.ravel()

def build_numeric_from_manifest(manifest_df, mode: str = "upper"):
    """
    manifest_df columns: ['subject_id', '__id__', 'filepath'] (aligned subset).
    Returns a pandas DataFrame with 'subject_id' + numeric feature columns.
    """
    import pandas as pd
    rows, ids = [], []
    first_len = None
    for _, rec in manifest_df.iterrows():
        fp = rec["filepath"]
        A = _read_numeric(fp)
        v = _vectorize_array(A, mode)
        if first_len is None:
            first_len = v.size
        elif v.size != first_len:
            raise SystemExit(f"Feature length mismatch for {fp}: {v.size} vs {first_len}")
        rows.append(v.astype(np.float32))
        ids.append(str(rec["subject_id"]))
    cols = [f"e{i:06d}" for i in range(first_len)]
    import pandas as pd
    X = pd.DataFrame(rows, columns=cols)
    X.insert(0, "subject_id", ids)
    return X


# ------------------------------- Loaders ------------------------------------ #

def load_clean_covars(
    covars_path: Path,
    id_col: str = "subject_id",
    site_col: str = "SITE",
    age_col: str = "AGE",
    dropna: bool = True,
    id_extractor: str | None = None,
    case_sensitive: bool = False,
) -> pd.DataFrame:
    df = read_csv_safely(covars_path)

    if id_col not in df.columns:
        raise SystemExit(f"Covariates file missing id_col '{id_col}'. Columns: {list(df.columns)}")

    canon = build_id_canonicalizer(pattern=id_extractor, case_sensitive=case_sensitive)
    df["__id__"] = df[id_col].astype(str).map(canon)

    if dropna:
        # Only enforce that the core columns exist before dropping
        drop_cols = [c for c in [id_col, site_col, age_col] if c in df.columns]
        if drop_cols:
            df = df.dropna(subset=drop_cols)

    return df


def load_features(
    connectomes: Path,
    id_col: str = "subject_id",
    id_extractor: str | None = None,
    case_sensitive: bool = False,
) -> Tuple[pd.DataFrame, bool]:
    """
    Returns (features_df, is_dir).

    If `connectomes` is a directory:
      - Scans files, extracts canonical IDs from filename stem using id_extractor.
      - Returns DataFrame with columns: ['subject_id', 'filepath', '__id__'].
      - is_dir=True.

    If it's a CSV/TSV:
      - Reads table, canonicalizes id_col to '__id__'.
      - Returns DataFrame with at least [id_col, '__id__', ...].
      - is_dir=False.
    """
    canon = build_id_canonicalizer(pattern=id_extractor, case_sensitive=case_sensitive)

    if connectomes.is_dir():
        rows = []
        for p in sorted(connectomes.iterdir()):
            if not p.is_file():
                continue
            stem = p.stem  # filename without extension
            subj_raw = stem
            subj_canon = canon(subj_raw)
            rows.append({"subject_id": subj_raw, "filepath": str(p), "__id__": subj_canon})

        if not rows:
            raise SystemExit(f"No files found in directory: {connectomes}")

        feats = pd.DataFrame(rows)

        # Deduplicate by canonical ID (keep first, warn on dups)
        dup_counts = feats["__id__"].value_counts()
        dups = dup_counts[dup_counts > 1]
        if len(dups) > 0:
            logging.warning("Duplicate feature IDs after canonicalization (keeping first): %s",
                            ", ".join(f"{k}:{v}" for k, v in dups.head(10).items()))
            feats = feats.sort_values("filepath").drop_duplicates("__id__", keep="first")

        return feats, True

    # Otherwise assume it's a table
    if not connectomes.exists():
        raise SystemExit(f"Features path does not exist: {connectomes}")

    # Try CSV/TSV read
    sep = None
    if connectomes.suffix.lower() == ".tsv":
        sep = "\t"

    feats = pd.read_csv(connectomes, sep=sep)
    if id_col not in feats.columns:
        raise SystemExit(f"Features table missing id_col '{id_col}'. Columns: {list(feats.columns)}")
    feats["__id__"] = feats[id_col].astype(str).map(canon)
    return feats, False


# ----------------------------- Alignment & Folds ----------------------------- #

def align_subjects(
    feats_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    id_col: str = "subject_id",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Align by the already-created '__id__' column (must exist on BOTH frames).
    Drops non-overlapping rows, logs counts, never raises on mismatch.
    """
    for name, df in [("features", feats_df), ("covariates", cov_df)]:
        if "__id__" not in df.columns:
            raise SystemExit(f"{name} does not contain '__id__' column; internal error.")

    f_ids = set(feats_df["__id__"])
    c_ids = set(cov_df["__id__"])
    kept = sorted(f_ids & c_ids)

    dropped_feats = len(f_ids - c_ids)
    dropped_covs = len(c_ids - f_ids)
    logging.info("Alignment: feats=%d, covs=%d, kept=%d, dropped_feats=%d, dropped_covs=%d",
                 len(f_ids), len(c_ids), len(kept), dropped_feats, dropped_covs)

    if not kept:
        logging.warning("No overlapping IDs after canonicalization; exiting cleanly.")
        return feats_df.iloc[0:0].copy(), cov_df.iloc[0:0].copy(), []

    f2 = feats_df[feats_df["__id__"].isin(kept)].copy()
    c2 = cov_df[cov_df["__id__"].isin(kept)].copy()
    return f2, c2, kept


def make_kfold_assignments(ids: Iterable[str], k: int, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic K-fold split over IDs (no sklearn dependency).
    """
    ids = list(ids)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ids))
    rng.shuffle(idx)
    folds = np.mod(idx, k)
    # Map back to id order
    fold_by_id = {ids[i]: int(folds[j]) for j, i in enumerate(idx)}
    return pd.DataFrame({"__id__": ids, "fold": [fold_by_id[i] for i in ids]})


# ---------------------------------- Main ------------------------------------ #

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Construct harmonization inputs (features, covars, folds).")

    ap.add_argument("--connectomes", required=True,
                    help="Path to a directory of feature files OR a CSV/TSV feature table.")
    ap.add_argument("--covars", required=True,
                    help="Path to covariates CSV.")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for aligned covars/features and folds.")

    ap.add_argument("--folds", type=int, default=5,
                    help="Number of folds (default: 5).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for fold assignment (default: 42).")

    ap.add_argument("--id-col", default="subject_id", help="ID column name in tables (default: subject_id).")
    ap.add_argument("--site-col", default="SITE", help="Site column name in covars (default: SITE).")
    ap.add_argument("--age-col", default="AGE", help="Age column name in covars (default: AGE).")
    ap.add_argument(
    "--vectorize",
    choices=["none", "upper", "flatten"],
    default="none",
    help="When --connectomes is a DIRECTORY, emit a numeric features table by "
         "vectorizing each per-subject file: 'upper' = upper triangle (no diag), "
         "'flatten' = full flatten. Default: none (manifest only).")

    # NEW robustness flags
    ap.add_argument(
        "--id-extractor",
        default=None,
        help=("Regex with ONE capturing group that extracts the canonical subject ID "
              "from a longer string, applied to BOTH covars and features.\n"
              "Example: '^([A-Za-z0-9]+_y[0-9]+)' turns 'H4567_y2_conn_plain' -> 'H4567_y2'.")
    )
    ap.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Match IDs case-sensitively (default: case-insensitive).",
    )

    ap.add_argument("-v", "--verbose", action="count", default=1,
                    help="Increase verbosity (-v, -vv).")

    args = ap.parse_args(argv)
    setup_logging(args.verbose)

    connectomes = Path(args.connectomes)
    covars_path = Path(args.covars)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load both sides and canonicalize
    feats, is_dir = load_features(
        connectomes=connectomes,
        id_col=args.id_col,
        id_extractor=args.id_extractor,
        case_sensitive=args.case_sensitive,
    )
    cov = load_clean_covars(
        covars_path=covars_path,
        id_col=args.id_col,
        site_col=args.site_col,
        age_col=args.age_col,
        dropna=True,
        id_extractor=args.id_extractor,
        case_sensitive=args.case_sensitive,
    )

    # Align (drop mismatches instead of crashing)
    feats2, cov2, kept = align_subjects(feats, cov, id_col=args.id_col)
    if len(kept) == 0:
        # Exit cleanly so pipelines can branch, rather than emitting a traceback
        print("No overlapping IDs after cleaning; nothing to do.", file=sys.stderr)
        return 0

    # Write aligned outputs
    cov_out = outdir / "covars_aligned.csv"
    cov2.to_csv(cov_out, index=False)
    logging.info("Wrote %s (%d rows)", cov_out, len(cov2))

    if is_dir:
        man_out = outdir / "features_manifest_aligned.csv"
        manifest_aligned = feats2[["subject_id", "__id__", "filepath"]].copy()
        manifest_aligned.to_csv(man_out, index=False)
        logging.info("Wrote %s (%d rows)", man_out, len(manifest_aligned))
    
        # NEW: also emit numeric features if requested
        if args.vectorize != "none":
            logging.info("Vectorizing per-subject files with mode='%s' ...", args.vectorize)
            Xnum = build_numeric_from_manifest(manifest_aligned, mode=args.vectorize)
            num_out = outdir / "features_numeric_aligned.csv"
            Xnum.to_csv(num_out, index=False)
            logging.info("Wrote %s (%d subjects × %d features)", num_out, Xnum.shape[0], Xnum.shape[1]-1)

    else:
        # For tables, write the aligned table
        feats_out = outdir / "features_aligned.csv"
        feats2.to_csv(feats_out, index=False)
        logging.info("Wrote %s (%d rows)", feats_out, len(feats2))

    # Folds over KEPT IDs
    folds_df = make_kfold_assignments(kept, k=args.folds, seed=args.seed)
    folds_out = outdir / "fold_assignments.csv"
    folds_df.to_csv(folds_out, index=False)
    logging.info("Wrote %s (%d rows)", folds_out, len(folds_df))

    # Summary to stdout (helpful in logs)
    print(
        f"Done. feats_in={len(feats['__id__'].unique())} cov_in={len(cov['__id__'].unique())} "
        f"kept={len(kept)} folds={args.folds}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
