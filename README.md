Requirements:
Conda (Miniconda/Mambaforge recommended)

# One-time setup
# from the repo root (where environment.yml lives)
bash ./setup_env.sh

# To activate env:
conda activate ./.envs/neuroharmonize-env

# To verify environment has been setup successfully:
bash ./verify_environment.sh

Notes

Modality: --modality dwi uses log1p/expm1; --modality fmri uses Fisher z (arctanh/tanh).

Anchoring: set --ref-site to the exact SITE string present after merging (e.g., ADDecode:1).

Merging: --min-per-site controls rare site merge into OTHER (REF is preserved).

Ridge: --ridge stabilizes the OLS; 1e-3 is a safe default.

QC: add --qc to emit auc_site.txt, mean_to_ref_distances.csv, pca_pre.png, pca_post.png, and a few subject panels under <out-dir>/<prefix>_qc/.
