Requirements:
Conda (Miniconda/Mambaforge recommended)

One-time setup from the repo root (where environment.yml lives) ```bash ./setup_env.sh```

To activate env: ```conda activate ./.envs/neuroharmonize-env```
To verify environment has been setup successfully:
```bash
bash ./verify_environment.sh
```

Notes

Modality: 
```bash
--modality dwi
```
uses log1p/expm1; 

```bash
--modality fmri
```
uses Fisher z (arctanh/tanh).

Anchoring: set
```bash
--ref-site
```
to the exact SITE string present after merging (e.g., ADDecode:1).

Merging:
```bash
--min-per-site
```
controls rare site merge into OTHER (REF is preserved).

Ridge:
```bash
--ridge
```
stabilizes the OLS; 1e-3 is a safe default.

QC: add
```bash
--qc
```
to emit auc_site.txt, mean_to_ref_distances.csv, pca_pre.png, pca_post.png, and a few subject panels under <out-dir>/<prefix>_qc/.

```bash
--ref-site
```
must match a SITE value in your covars after rare-site merging. You can also anchor to something like "UNIVERSITY_NORTH_TEXAS_HS" by changing that flag.

fMRI input: if your fMRI edges are already Fisher-z, convert back to r via tanh(z) before running (the script does Fisher-z internally). I can drop a one-liner if you need it.

Outputs: per-subject CSVs (N×N) under --out-dir/<prefix>_matrices/, plus QC in .../<prefix>_qc/ when --qc is set.

Sample usage calls:
DWI (structural) — quick anchored run:
```
python harmonize_connectomes.py \
  --features /mnt/newStor/paros/paros_WORK/harmonization/all_features_numeric.csv \
  --covars   /mnt/newStor/paros/paros_WORK/harmonization/all_covars_aligned.csv \
  --out-dir  /mnt/newStor/paros/paros_WORK/harmonization/harmonized_full_dwi_location_only \
  --ref-site "ADDecode:1" \
  --modality dwi \
  --nodes 84
```
DWI — with QC + common knobs:
```
python harmonize_connectomes.py \
  --features /mnt/newStor/paros/paros_WORK/harmonization/all_features_numeric.csv \
  --covars   /mnt/newStor/paros/paros_WORK/harmonization/all_covars_aligned.csv \
  --out-dir  /mnt/newStor/paros/paros_WORK/harmonization/harmonized_full_dwi_location_only_qc \
  --prefix   ADDECODE_ADNI_HABS \
  --ref-site "ADDecode:1" \
  --modality dwi \
  --nodes 84 \
  --min-per-site 20 \
  --ridge 1e-3 \
  --qc --qc-topk 1500 --qc-folds 5 --qc-min-per-site 5 --qc-panels 6 --qc-pca-sample 500
```
fMRI (functional) — quick anchored run
⚠️ Assumes your edge values are raw correlations in [-1, 1] (not Fisher-z yet).
```
python harmonize_connectomes.py \
  --features /path/to/fmri_features_numeric.csv \
  --covars   /mnt/newStor/paros/paros_WORK/harmonization/all_covars_aligned.csv \
  --out-dir  /mnt/newStor/paros/paros_WORK/harmonization/harmonized_full_fmri_location_only \
  --ref-site "ADDecode:1" \
  --modality fmri \
  --nodes 84
```
fMRI — with QC + slightly stronger regularization:
```
python harmonize_connectomes.py \
  --features /path/to/fmri_features_numeric.csv \
  --covars   /mnt/newStor/paros/paros_WORK/harmonization/all_covars_aligned.csv \
  --out-dir  /mnt/newStor/paros/paros_WORK/harmonization/harmonized_full_fmri_location_only_qc \
  --prefix   ADDECODE_ADNI_HABS_fMRI \
  --ref-site "ADDecode:1" \
  --modality fmri \
  --nodes 84 \
  --min-per-site 20 \
  --ridge 1e-2 \
  --qc --qc-topk 1500 --qc-folds 5 --qc-min-per-site 5 --qc-panels 6 --qc-pca-sample 500
```
