# ECG Preprocessing Pipeline

This directory contains a deterministic preprocessing pipeline for the MIMIC-IV ECG feature dataset in `../dataset/processed/extracted_ecg_features.csv`.

## What it does

- loads and validates the processed ECG feature dataset
- removes exact duplicate rows
- removes predetermined features from `config.yaml`
- validates datatypes, NaN/Inf, sentinel values, and impossible values
- splits data by `SubjectID`
- fits winsorization, transformations, and scalers on the training split only
- applies fitted preprocessing to validation and test splits
- runs post-processing integrity and redundancy checks
- exports cleaned datasets, metadata, parameters, and a single timestamped log file

## Usage

From this directory:

```bash
python main.py
```

Outputs are written to `outputs/`.

## Main Outputs

- `outputs/datasets/`
  - final reduced-feature datasets used by the preprocessing pipeline
  - these are deduplicated, feature-filtered, winsorized, transformed, and scaled
- `outputs/datasets_all_features/`
  - alternative datasets with all original features retained
  - these are deduplicated and split by `SubjectID`, but no features are removed and no scaling or transformation is applied
- `outputs/metadata/feature_removal_justification.ipynb`
  - the main notebook explaining why removed feature categories were removed
- `outputs/metadata/removed_features_with_reasons.csv`
  - flat file listing each removed feature and its explicit audit reason
- `outputs/metadata/audit_report.md`
  - end-to-end post-run verification of the preprocessing pipeline

## Notes

- Only `../dataset/processed` is used as pipeline input.
- `../dataset/data` is intentionally ignored by this pipeline.
- All configurable preprocessing behavior lives in `config.yaml`.
- For a more detailed directory explanation, see `FOLDER_GUIDE.md`.

