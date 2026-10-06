# Post-Run Audit Report

Audit date: August 3, 2026

Scope: verification only. The preprocessing pipeline code was not modified during this audit.

## Overall Result

The exported artifacts are internally consistent with `config.yaml`, the fitted parameter files, and the raw MIMIC-IV feature dataset in `dataset/processed/extracted_ecg_features.csv`.

No preprocessing bug was found in this audit.

## 1. Config-Listed Feature Removal

- `config.yaml` defines removal through two regex rules and no explicit feature list entries.
- Expected removed feature count from config: `64`
- Actual removed feature count in `outputs/metadata/removed_features.csv`: `64`
- Features expected but not removed: `0`
- Unexpected removed features: `0`

Result: every feature targeted by `config.yaml` was removed.

## 2. Unexpected Feature Removal

- Original column count: `160`
- Expected retained columns after removal: `96` including `SubjectID`
- Expected retained model features: `95` excluding `SubjectID`
- `outputs/metadata/retained_features.csv` exactly matches the mathematically expected retained feature list.

Result: no unexpected features were removed.

## 3. Final Feature Count Math

- Raw dataset columns: `160`
- Removed columns: `64`
- Retained columns: `160 - 64 = 96`
- `SubjectID` retained as identifier: `1`
- Final model-ready feature count: `96 - 1 = 95`
- Exported `train.csv`, `validation.csv`, and `test.csv` each contain `96` columns total and `95` non-identifier features.

Result: final feature count is mathematically correct.

## 4. Split Column Identity and Order

- `train.csv`, `validation.csv`, and `test.csv` contain identical columns.
- Column order is identical across all three splits.
- A full reconstruction from raw data plus saved parameters reproduces each export with only floating-point noise at `3.55e-15` max absolute difference.

Result: split schemas are identical and ordered consistently.

## 5. SubjectID Leakage Check

- Overlap between train and validation `SubjectID`: `0`
- Overlap between train and test `SubjectID`: `0`
- Overlap between validation and test `SubjectID`: `0`

Result: no `SubjectID` appears in more than one split.

## 6. Scaling Fit Verification

Scaler parameters were checked against the transformed training split and against the transformed full dataset.

- `global_mean` `StandardScaler`: matches training only, does not match full data
- `amplitude_mean` `RobustScaler`: matches training only, does not match full data
- `iqr` `RobustScaler`: matches training only, does not match full data
- `count` `StandardScaler`: matches training only, does not match full data
- `st_elev_mean` `RobustScaler`: matches training only, does not match full data

Result: scaling was fit only on the training split.

## 7. Impossible `RR_Mean_Global` Values Before and After

These were the only physiologically impossible values detected before preprocessing. All 11 were above the configured physiological maximum of `2500`, and after winsorization they were clipped to the fitted training upper bound `1439.378571428574`.

| Split | Row Index After Dedup | SubjectID | Before | After Winsorization |
|---|---:|---:|---:|---:|
| train | 18363 | 10480744 | 2609.0 | 1439.378571428574 |
| validation | 24844 | 10639500 | 5968.0 | 1439.378571428574 |
| validation | 50870 | 11279168 | 3541.33333333333 | 1439.378571428574 |
| train | 102038 | 12517010 | 5282.0 | 1439.378571428574 |
| train | 166718 | 14085291 | 3959.5 | 1439.378571428574 |
| train | 180454 | 14415578 | 4385.33333333333 | 1439.378571428574 |
| train | 233625 | 15709718 | 2790.5 | 1439.378571428574 |
| train | 249739 | 16099802 | 2874.5 | 1439.378571428574 |
| train | 295527 | 17257484 | 4072.66666666667 | 1439.378571428574 |
| validation | 297683 | 17316016 | 2634.0 | 1439.378571428574 |
| train | 390609 | 19601270 | 3151.33333333333 | 1439.378571428574 |

Result: all 11 impossible `RR_Mean_Global` values were corrected by winsorization.

## 8. Winsorization Behavior

Winsorization was verified directly against `outputs/parameters/winsorization.json`.

- Train values outside fitted bounds: `211,937`
- Train values changed by winsorization: `211,937`
- Validation values outside fitted bounds: `44,622`
- Validation values changed by winsorization: `44,622`
- Test values outside fitted bounds: `46,895`
- Test values changed by winsorization: `46,895`

No in-bound value was changed in any split.

Result: winsorization changed only values outside the configured fitted bounds.

## 9. Exported CSV Hygiene

- `train.csv`: no NaN, no Inf, no duplicated columns
- `validation.csv`: no NaN, no Inf, no duplicated columns
- `test.csv`: no NaN, no Inf, no duplicated columns

Result: all exported CSVs passed the requested hygiene checks.

## 10. Final Audit Conclusion

- Feature removal exactly matches `config.yaml`
- No unexpected features were removed
- Final feature count is correct at `95` model features plus `SubjectID`
- Split columns are identical and ordered identically
- No `SubjectID` leakage exists across splits
- Scaling was fit on training data only
- The 11 impossible `RR_Mean_Global` values were clipped during preprocessing
- Winsorization changed only out-of-bound values
- Exported CSVs contain no NaN, no Inf, and no duplicated columns

Final status: audit passed.
