# Folder Guide

This document explains what each major folder in `preprocessing_pipeline/` is doing and which files you should look at depending on your question.

## If Your Question Is "How Do I Run The Pipeline?"

Run:

```bash
python main.py
```

That executes the preprocessing pipeline defined by `config.yaml`.

## If Your Question Is "Do I Need To Run Anything Else?"

For the main preprocessing output, no additional folder needs to be run.

What has already been executed:

- `main.py`
  - produced the final reduced-feature outputs in `outputs/datasets/`
- post-run audit checks
  - produced `outputs/metadata/audit_report.md`
- feature-removal justification notebook
  - the current primary notebook is `outputs/metadata/feature_removal_justification.ipynb`

What is optional:

- opening the notebooks in Jupyter for interactive inspection
- re-running the notebooks if you want live outputs in your own environment

## If Your Question Is "Which File Explains Why Features Were Removed?"

Use these in this order:

1. `outputs/metadata/feature_removal_justification.ipynb`
   - main interactive notebook
   - best place to understand why removed feature categories were removed
   - separates direct notebook evidence from historical project decisions

2. `outputs/metadata/removed_features_with_reasons.csv`
   - one row per removed feature
   - quickest flat file for explicit feature-by-feature reasons

3. `outputs/metadata/feature_removal_audit.ipynb`
   - older notebook copy
   - keep it as a secondary reference, not the primary one

## Folder Overview

### `core/`

Infrastructure utilities used by the pipeline.

- `config.py`
  - loads `config.yaml`
- `logger.py`
  - creates the timestamped pipeline log
- `constants.py`
  - shared filenames and supported options

### `data/`

Data loading, splitting, and exporting.

- `loader.py`
  - reads the processed ECG CSV and validates schema
- `splitter.py`
  - performs subject-level train/validation/test splitting
- `exporter.py`
  - writes datasets, metadata, and parameter artifacts

### `cleaning/`

The actual preprocessing steps.

- `deduplication.py`
  - removes exact duplicate rows
- `feature_removal.py`
  - removes predetermined features from config
- `type_validation.py`
  - checks NaN, Inf, dtype issues, sentinel values, and impossible values
- `outlier_treatment.py`
  - fits and applies winsorization
- `transformation.py`
  - applies `none`, `log1p`, or `Yeo-Johnson`
- `scaling.py`
  - applies `StandardScaler` or `RobustScaler`

### `validation/`

Checks on the processed outputs.

- `integrity_checks.py`
  - verifies no NaN/Inf, no duplicate rows, and no subject leakage across splits
- `redundancy_checks.py`
  - computes post-processing correlation and VIF summaries

### `utils/`

Small helpers shared across modules.

### `outputs/`

Generated artifacts.

#### `outputs/datasets/`

Final reduced-feature modeling datasets.

- feature removal applied
- winsorization applied
- transformations applied
- scaling applied

These are the main outputs for downstream causal modeling.

#### `outputs/datasets_all_features/`

Alternative all-feature datasets.

- exact duplicate rows removed
- split by `SubjectID`
- no feature removal
- no winsorization
- no transformation
- no scaling

Use these if someone wants the original full feature set with the same patient-level split logic.

#### `outputs/metadata/`

Audit and explanation artifacts.

Important files:

- `audit_report.md`
  - pipeline correctness audit
- `removed_features.csv`
  - list of removed features
- `retained_features.csv`
  - list of kept features
- `removed_features_with_reasons.csv`
  - explicit reason per removed feature
- `winsorization_summary.csv`
  - per-feature clipping summary
- `feature_statistics_before.csv`
  - descriptive statistics before winsorization
- `feature_statistics_after.csv`
  - descriptive statistics after winsorization
- `feature_removal_justification.ipynb`
  - main notebook for removal reasoning

#### `outputs/parameters/`

Saved preprocessing parameters:

- winsorization bounds
- transformation objects
- scaler objects

#### `outputs/logs/`

The pipeline execution log.

