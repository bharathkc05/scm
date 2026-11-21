# Phase 3 — Causal Dataset Report

**Date:** 2026-11-21 · **Reproduce:** `p3_01_splits` → `p3_02_latents` → `p3_03_build` (project env)
**Files (git-ignored):** `data/roadmap/phase3/layer1.parquet` (430,852 × 37), `layer2.parquet` (193,470 × 312)
**Committed aggregates:** `cohort_flow.csv`, `winsorization_bounds.json`, `p3_0*_*.json`

## 1. Cohort flow

| Step | Admissions | Patients |
|---|---:|---:|
| All admissions (MIMIC-IV v2.2) | 431,231 | 180,733 |
| **Layer 1:** ≥1 diagnosis code | **430,852** | 180,640 |
| ↳ acute MI | 11,222 | 9,459 |
| With any linked ECG | 219,366 | 108,729 |
| Layer 2 candidates: ECG with features | 193,470 | 100,098 |
| ↳ acute MI, after onset | 5,643 | 5,112 |
| ↳ acute MI, uncertain timing (excluded from primary) | 3,780 | |
| ↳ ICD-9 410.x2 follow-up admissions (excluded from controls) | 805 | |
| ↳ no diagnosis codes (excluded from controls) | 77 | |
| ↳ controls (no acute MI) | 183,165 | 95,964 |
| **Layer 2 primary set** | **188,808** | 98,662 |

## 2. Unit and ECG selection

One row per admission (`hadm_id`). Layer 2 uses the **earliest ECG with extracted features** inside the
admission window (ED registration, or admission if no ED visit → discharge). Features come from the raw
full feature file, not the legacy master (Phase 2, G1).

## 3. Splits (`p3_01_splits.json`)

Every patient is in exactly one split, and Layer 1 and Layer 2 share it.
- 99,256 patients keep their legacy split (`split_manifest.csv`).
- 81,477 patients not in the manifest were assigned 70/15/15 with seed 20261121, stratified by
  ever having an acute MI.
- Patients: train 126,513 · validation 27,110 · test 27,110; ever-MI rate 5.2–5.4% in each.
- Layer 2 rows (after-onset MI): train 135,179 (3,922) · validation 29,031 (858) · test 29,260 (863).

## 4. Variables

| Group | Columns | Notes |
|---|---|---|
| IDs | `hadm_id`, `subject_id`, `study_id`, `split` | |
| Exposure / outcome | `acute_mi`, `acute_mi_principal`, `mi_stemi`, `mi_nstemi`, `mi_type2`, `mi_other_unspecified` | Phase 1 definitions |
| Covariates (primary) | `age`, `sex_male`, `diabetes_primary`, `hypertension_primary`, `ckd_primary`, `prior_mi_primary` | prior + same-admission chronic codes |
| Covariates (strict) | `*_strict` | earlier admissions only |
| Look-back | `n_prior_admissions` | 0 = first admission |
| Context | `arrived_via_ed`, `planned_admission`, `admission_type`, `admission_location`, `chest_pain`, `recent_mi_followup`, `future_acute_mi`, `old_mi` | |
| Timing (Layer 2) | `timing_class`, `after_strict`, `hours_from_arrival`, `first_ecg_has_features` | `legacy_label` kept for audit only, never a covariate |
| ECG outcomes, raw | 95 features (53 point, 41 `_iqr`, `QRS_Dur_Global_count`) + 5 ST composites | composites: inferior, anterior, lateral mean; max elevation and depression over non-aVR leads |
| ECG outcomes, winsorized | `w_<feature>` | clipped at training-split 0.5th / 99.5th percentiles |
| Validation outcomes | `machine_qrs_ms`, `machine_rr_ms` | ECG cart measurements; 99.9% coverage |
| Latents | `z_ecg_1` … `z_ecg_64`, `latent_source` | 190,536 stored + 2,934 encoded from WFDB (matches stored within 2.3 × 10⁻⁷) |
| Quality | `beat_detection_ratio`, `quality_ok` | detected / expected beats; ok = 0.8–1.25 |
| Selection (Layer 1) | `has_any_ecg`, `has_feature_ecg` | for the ECG-selection model |

Scaling to control-SD units is done in Phase 5, inside cross-fitting.

## 5. Analysis sets (Layer 2)

Controls everywhere = no acute MI, not a 410.x2 follow-up, at least one diagnosis code.

| Flag | Exposed rows | Total rows | Purpose |
|---|---:|---:|---|
| `set_primary` | 5,643 | 188,808 | after-onset MI vs controls |
| `set_active_comparator` | 5,643 | 26,186 | vs chest-pain controls (selection by design) |
| `set_after_strict` | 3,976 | 187,141 | admission context and legacy troponin agree |
| `set_nstemi` | 4,149 | 187,314 | circularity check (Gate 1 condition) |
| `set_no_type2` | 5,563 | 188,728 | type 2 MI excluded |
| `set_first_ecg_has_features` | 4,787 | 165,135 | first ECG of the admission is the one analysed |
| `set_negative_control` | 6,938 (future MI) | 159,071 | controls with no prior MI; exposure = acute MI on a later admission |

## 6. Findings during the build

- **Beat-detection quality is not a useful filter.** The detected/expected beat ratio has a median of
  0.999 (1st–99th percentile 0.90–1.07), and 99.96% of ECGs pass, so the planned high-quality subset
  would not differ from the primary set. `QRS_Dur_Global_count` stays out of the adjustment sets (D5).
- 77 Layer 2 admissions have no diagnosis codes at all; their MI status is unknown, so they are never
  controls.
- 27 after-onset MI admissions carry both a STEMI and an NSTEMI code and are left out of the NSTEMI set
  (4,149 rather than the 4,176 in the audit).

## 7. Gate check (`p3_03_build.json`)

| Check | Result |
|---|---|
| Layer 1 rows = Phase 2 admissions with a diagnosis code (430,852) | ✅ |
| Layer 2 rows = Phase 2 admissions with a feature ECG (193,470) | ✅ |
| After-onset MI = Phase 2 count (5,643) | ✅ |
| One row per admission; every patient in one split; no missing split | ✅ |
| All 95 features and 64 latents complete | ✅ |
| No forbidden covariate in the dataset (except `legacy_label`, audit-only) | ✅ |
