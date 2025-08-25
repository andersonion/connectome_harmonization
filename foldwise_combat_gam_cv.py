#!/usr/bin/env python3
"""
foldwise_combat_gam_cv.py

Fold-aware harmonization for connectome features using ComBat-GAM (neuroHarmonize)
with optional anchored ComBat (spline(age) + neuroCombat w/ ref_batch).

Key idea: fit harmonization ONLY on training subjects in each fold,
then apply the learned model to that fold's held-out test subjects.

Requirements
-----------
pip install neuroHarmonize neuroCombat patsy pandas numpy scikit-learn matplotlib

Inputs
------
--features: CSV of subjects x edge features. Index column = subject IDs.
--covars:   CSV of covariates with columns: subject_id, SITE, AGE, [optional covars]
--folds:    Number of StratifiedKFold splits by SITE (default 5).

Outputs (under --out-dir)
-------------------------
- <prefix>_fold<k>_train_harmonized.csv
- <prefix>_fold<k>_test_harmonized.csv
- <prefix>_fold<k>_prepost_auc.txt (site separability AUC before vs after on TEST set)
- <prefix>_fold<k>_pca_pre.png and _pca_post.png (TEST set PCA colored by SITE)
- <prefix>_alltest_harmonized.csv (concatenated test splits, in original subject order)
- <prefix>_summary.csv (per-fold AUCs and counts)

Notes
-----
- If your features are correlations, Fisher-z transform before harmonization.
- Ensure identical edge ordering across cohorts.
- For anchored mode we approximate GAM with spline(age) in neuroCombat and use
  ref_batch to map into a chosen reference site (e.g., AD-DECODE).

Example
-------
python foldwise_combat_gam_cv.py \
  --features edges.csv \
  --covars covars.csv \
  --site-col SITE --age-col AGE --id-col subject_id \
  --folds 5 --mode gam \
  --out-dir harmonized_cv --prefix ADNI_ADDECODE

python foldwise_combat_gam_cv.py \
  --features edges.csv \
  --covars covars.csv \
  --site-col SITE --age-col AGE --id-col subject_id \
  --folds 5 --mode anchored --ref-batch AD-DECODE \
  --out-dir harmonized_cv --prefix ADNI_to_ADDECODE
"""

import argparse, os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def pca_scatter(X, sites, title, save_path):
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(7, 5))
    # Plot each site with default colors
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], label=str(s), alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def site_auc_on_test(X_train, y_train, X_test, y_test):
    # Project to 10 PCs, train LR on train, evaluate AUC on test (multi-class OVR)
    pca = PCA(n_components=min(10, X_train.shape[1]), random_state=42)
    Ztr = pca.fit_transform(X_train)
    Zte = pca.transform(X_test)
    clf = LogisticRegression(max_iter=500, multi_class='ovr')
    clf.fit(Ztr, y_train)
    # If binary, use decision_function; if multi-class, macro-average OVR AUC
    if len(np.unique(y_test)) == 2:
        scores = clf.decision_function(Zte)
        auc = roc_auc_score(y_test, scores)
    else:
        scores = clf.decision_function(Zte)
        auc = roc_auc_score(y_test, scores, multi_class='ovr')
    return auc

def encode_covars(covars, site_col, age_col):
    cov = covars.copy()
    cov = cov.rename(columns={site_col: 'SITE'})
    keep = ['SITE', age_col]
    other = [c for c in cov.columns if c not in keep]
    if other:
        cov = pd.concat([cov[keep], pd.get_dummies(cov[other], drop_first=True)], axis=1)
    else:
        cov = cov[keep]
    return cov

def fit_apply_gam(Xtr, Xte, cov_tr, cov_te, age_col, smooth_bounds=None):
    from neuroHarmonize import harmonizationLearn, applyHarmonizationModel
    kwargs = dict(smooth_terms=[age_col])
    if smooth_bounds is not None:
        kwargs['smooth_term_bounds'] = smooth_bounds
    model, Xtr_adj = harmonizationLearn(Xtr, cov_tr, **kwargs)
    Xte_adj = applyHarmonizationModel(Xte, cov_te, model)
    return Xtr_adj, Xte_adj, model

def fit_apply_anchored(Xtr, Xte, cov_tr, cov_te, ref_batch, site_col='SITE', age_col='AGE'):
    from patsy import dmatrix
    from neuroCombat import neuroCombat, neuroCombatFromTraining

    # Build spline basis on train, apply to test
    age_tr = cov_tr[age_col].astype(float)
    age_te = cov_te[age_col].astype(float)
    spline_tr = dmatrix("bs(age, df=5, degree=3, include_intercept=False)",
                        {"age": age_tr}, return_type='dataframe')
    spline_te = dmatrix("bs(age, df=5, degree=3, include_intercept=False)",
                        {"age": age_te}, return_type='dataframe')
    spline_tr.columns = [f"AGE_s{i}" for i in range(spline_tr.shape[1])]
    spline_te.columns = spline_tr.columns

    # Design matrices: SITE + spline(age) + other linear covars
    def build_design(cov, spline):
        design = pd.concat([cov[[site_col]], spline], axis=1)
        other = [c for c in cov.columns if c not in [site_col, age_col]]
        if other:
            design = pd.concat([design, pd.get_dummies(cov[other], drop_first=True)], axis=1)
        return design

    dtr = build_design(cov_tr, spline_tr)
    dte = build_design(cov_te, spline_te)

    # Fit on training
    out_tr = neuroCombat(dat=Xtr.T, covars=dtr, batch_col=site_col,
                         continuous_cols=[c for c in dtr.columns if c.startswith('AGE_s')],
                         categorical_cols=[c for c in dtr.columns if c not in (['SITE'] + [c for c in dtr.columns if c.startswith('AGE_s')])],
                         ref_batch=ref_batch, return_s_data=True)
    Xtr_adj = out_tr['data'].T
    params = out_tr['estimates']  # parameters for fromTraining

    # Apply to test
    out_te = neuroCombatFromTraining(dat=Xte.T, covars=dte, batch_col=site_col,
                                     estimates=params, ref_batch=ref_batch, return_s_data=True)
    Xte_adj = out_te['data'].T
    return Xtr_adj, Xte_adj, params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='CSV: subjects x edges, index=subject_id')
    ap.add_argument('--covars', required=True, help='CSV: covariates incl. subject_id, SITE, AGE')
    ap.add_argument('--id-col', default='subject_id')
    ap.add_argument('--site-col', default='SITE')
    ap.add_argument('--age-col', default='AGE')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--mode', choices=['gam', 'anchored'], default='gam')
    ap.add_argument('--ref-batch', default=None, help='Required if mode=anchored (e.g., AD-DECODE)')
    ap.add_argument('--age-min', type=float, default=None)
    ap.add_argument('--age-max', type=float, default=None)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefix', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X = pd.read_csv(args.features, index_col=0)
    C = pd.read_csv(args.covars)
    C = C.set_index(args.id_col)
    # Align subjects
    common = X.index.intersection(C.index)
    X = X.loc[common].copy()
    C = C.loc[common].copy()

    sites = C[args.site_col].astype(str).values
    ages  = C[args.age_col].astype(float).values

    # Summary containers
    records = []
    all_test_adj = []

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X.values, sites), start=1):
        Xtr = X.iloc[tr_idx].values.astype(np.float32)
        Xte = X.iloc[te_idx].values.astype(np.float32)

        Ctr_raw = C.iloc[tr_idx].copy()
        Cte_raw = C.iloc[te_idx].copy()

        # Pre-harmonization test AUC
        pre_auc = site_auc_on_test(Xtr, sites[tr_idx], Xte, sites[te_idx])

        if args.mode == 'gam':
            cov_tr = encode_covars(Ctr_raw[[args.site_col, args.age_col] + [c for c in Ctr_raw.columns if c not in [args.site_col, args.age_col]]],
                                   args.site_col, args.age_col)
            cov_te = encode_covars(Cte_raw[[args.site_col, args.age_col] + [c for c in Cte_raw.columns if c not in [args.site_col, args.age_col]]],
                                   args.site_col, args.age_col)
            smooth_bounds = None
            if args.age_min is not None and args.age_max is not None:
                smooth_bounds = (args.age_min, args.age_max)
            Xtr_adj, Xte_adj, model = fit_apply_gam(Xtr, Xte, cov_tr, cov_te, age_col=args.age_col,
                                                    smooth_bounds=smooth_bounds)
            # Save model
            with open(os.path.join(args.out_dir, f"{args.prefix}_fold{fold}_gam_model.pkl"), 'wb') as f:
                pickle.dump(model, f)
        else:
            if args.ref_batch is None:
                raise SystemExit("--mode anchored requires --ref-batch")
            # For anchored, keep raw covars; function will build spline + design matrices
            Xtr_adj, Xte_adj, params = fit_apply_anchored(Xtr, Xte,
                                                          Ctr_raw[[args.site_col, args.age_col] + [c for c in Ctr_raw.columns if c not in [args.site_col, args.age_col]]],
                                                          Cte_raw[[args.site_col, args.age_col] + [c for c in Cte_raw.columns if c not in [args.site_col, args.age_col]]],
                                                          ref_batch=args.ref_batch,
                                                          site_col=args.site_col, age_col=args.age_col)
            with open(os.path.join(args.out_dir, f"{args.prefix}_fold{fold}_anchored_params.pkl"), 'wb') as f:
                pickle.dump(params, f)

        # Post-harmonization test AUC (site separability should drop)
        post_auc = site_auc_on_test(Xtr_adj, sites[tr_idx], Xte_adj, sites[te_idx])

        # Save dataframes with subject indices
        Xtr_adj_df = pd.DataFrame(Xtr_adj, index=X.index[tr_idx], columns=X.columns)
        Xte_adj_df = pd.DataFrame(Xte_adj, index=X.index[te_idx], columns=X.columns)
        Xtr_adj_df.to_csv(os.path.join(args.out_dir, f"{args.prefix}_fold{fold}_train_harmonized.csv"))
        Xte_adj_df.to_csv(os.path.join(args.out_dir, f"{args.prefix}_fold{fold}_test_harmonized.csv"))

        # PCA plots on TEST set (pre vs post)
        pca_scatter(Xte, sites[te_idx], f"Fold {fold} TEST PCA (pre-harmonization)",
                    os.path.join(args.out_dir, f"{args.prefix}_fold{fold}_pca_pre.png"))
        pca_scatter(Xte_adj, sites[te_idx], f"Fold {fold} TEST PCA (post-harmonization)",
                    os.path.join(args.out_dir, f"{args.prefix}_fold{fold}_pca_post.png"))

        # Write a tiny report
        with open(os.path.join(args.out_dir, f"{args.prefix}_fold{fold}_prepost_auc.txt"), 'w') as f:
            f.write(f"Fold {fold}\n")
            f.write(f"Pre-harmonization TEST AUC:  {pre_auc:.3f}\n")
            f.write(f"Post-harmonization TEST AUC: {post_auc:.3f}\n")
            f.write(f"N_train={len(tr_idx)}, N_test={len(te_idx)}\n")

        # For concatenated test set output
        all_test_adj.append(Xte_adj_df)
        records.append(dict(fold=fold, pre_auc=pre_auc, post_auc=post_auc,
                            n_train=len(tr_idx), n_test=len(te_idx)))

    # Concatenate test splits in original subject order
    all_test_df = pd.concat(all_test_adj).loc[X.index]
    all_test_df.to_csv(os.path.join(args.out_dir, f"{args.prefix}_alltest_harmonized.csv"))

    # Summary CSV
    pd.DataFrame.from_records(records).to_csv(os.path.join(args.out_dir, f"{args.prefix}_summary.csv"), index=False)

if __name__ == "__main__":
    main()
