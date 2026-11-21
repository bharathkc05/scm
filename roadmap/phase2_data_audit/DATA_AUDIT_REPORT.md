# Phase 2 — Data Audit Report

**Date:** 2026-11-21 · **Data:** MIMIC-IV v2.2, MIMIC-IV-ECG v1.0 · **Definitions:** `roadmap/definitions.py` (Phase 1)
**Reproduce:** run `p2_01` → `p2_05` in this folder (project env); each writes a JSON summary quoted below.
Patient-level outputs are in `data/roadmap/phase2/` (git-ignored).

## Summary

| Roadmap check | Result | Verdict |
|---|---|---|
| Chronic conditions: prior + same-admission (primary), prior-only (strict) | Both built for all 431,231 admissions; 64–78% of primary positives are supported by an earlier admission | ✅ Pass, with a first-admission bias to handle |
| MI label is acute-only, covers ICD-9 and ICD-10 | 11,222 admissions (5,376 / 5,846); 410.x2 and old-MI codes excluded | ✅ Pass |
| MI label not decided by reading the ECG | Cannot be verified from coded data; 4,176 after-onset NSTEMI admissions available for the pre-specified circularity check | ⚠️ Conditional |
| Admissions with and without an ECG | 219,471 with any ECG (50.9%), 193,470 with a feature ECG (44.9%) | ✅ Counted; selection is strong |
| ECG timing before / after / uncertain | 5,643 after-onset and 3,780 uncertain MI admissions with a feature ECG; 10,508 before-onset admissions for the negative control | ✅ Pass |
| β-VAE provenance | Stored embeddings reproduce exactly from the checkpoint; HDF5 matches raw WFDB | ✅ Pass — latents eligible |

## 1. Chronic conditions and history (`p2_01_admissions.json`)

| Condition | Primary prevalence | Strict (prior-only) | Primary positives backed by an earlier admission |
|---|---:|---:|---:|
| Diabetes | 26.8% | 18.3% | 68.3% |
| Hypertension | 57.0% | 36.6% | 64.2% |
| CKD | 18.2% | 13.0% | 71.5% |
| Prior MI | 9.7% | 7.6% | 78.1% |

Age and sex are complete for every admission. 379 admissions have no diagnosis code.

**First-admission bias (new finding).** 180,733 admissions (41.9%) are a patient's first, so every
strict history flag is 0 for them. Acute MI is **more common on first admissions (3.27%) than on repeat
admissions (2.12%)**. A strict history flag is therefore mechanically associated with *lower* MI risk,
which likely explains part of the inverse hypertension estimate in the previous attempt. The Phase 1
look-back sensitivity (≥1 earlier admission; number of earlier admissions as covariate) is required,
and same-admission chronic codes in the primary definition reduce the problem.

## 2. MI label (`p2_01`, `p2_03_label_audit.json`)

- Acute MI: 11,222 admissions; 6,640 with MI as principal diagnosis; 389 in planned admissions.
- **Agreement with the legacy master label** (ECG-linked ICD file) on 194,587 shared admissions: 99.6%.
  Every `acute_mi` admission is legacy MI; the 818 legacy-only MI admissions are almost all old-MI
  or 410.x2 follow-up codes (813), which Phase 1 excludes on purpose.
- **"Control_Symptomatic" did not mean troponin-tested.** Troponin testing (items 51002, 51003, 52642,
  linked by patient and time window): acute MI 92.5%, non-MI admissions 28.8%, non-MI admissions with
  a feature ECG 49.5%, **chest-pain non-MI with a feature ECG 93.4%**. The chest-pain comparator gets the
  same work-up as MI patients, which supports it as the active comparator.
- **ECG reading.** ICD codes are assigned from clinical documentation, and STEMI vs NSTEMI depends on the
  ECG, so circularity cannot be ruled out from the data. Mitigation (pre-specified in Phase 1): run Layer 2
  on NSTEMI only. After-onset admissions with a feature ECG: NSTEMI 4,176, STEMI 1,306, type 2 80,
  other 167. Type 2 MI falls mostly in the uncertain class (669), as expected for in-hospital events.

## 3. ECG coverage and selection (`p2_02_ecg_linkage.json`)

ECGs were linked to the admission whose window (ED registration or admission → discharge) contains them.
Of 800,035 ECGs, 464,033 fall inside an admission; 336,002 are ED-only or outpatient and are not used.
Linkage agrees with the legacy `hadm_id` for 99.9% of ECGs.

| Group | Admissions | With any ECG | With a feature ECG |
|---|---:|---:|---:|
| Acute MI | 11,222 | 84.5% | 84.0% |
| No acute MI | 420,009 | 50.0% | 43.8% |
| Chest pain, no MI | 25,973 | 88.4% | 79.4% |

Selection into the ECG cohort is strongly related to MI, as the roadmap warned. The chest-pain
comparator has ECG coverage close to MI admissions, so it handles selection by design.

**24,319 admissions** have a first ECG without extracted features and a later one with features. The
Phase 3 rule is "earliest ECG with features"; a sensitivity restricts to admissions whose first ECG has
features.

## 4. ECG timing relative to MI onset (`p2_02`, `p2_03`)

Rule (admission context, not troponin): *after* = MI is the principal diagnosis and the admission was
unplanned; *uncertain* = MI coded secondary or planned admission; *before* = ECG in an earlier, non-MI
admission of a patient who later has an acute MI.

| Class | ECGs | Admissions with a feature ECG |
|---|---:|---:|
| After | 26,152 | 5,643 MI admissions |
| Uncertain | 18,655 | 3,780 MI admissions |
| Before | 25,496 | 10,508 admissions |
| Control | 393,730 | — |

After-onset first feature ECGs are taken a median of **0.83 h** after arrival (IQR 0.20–7.66 h; 90th
percentile 17.8 h).

**Conflict with the legacy troponin timing.** Of the 5,643 after-onset first ECGs, the legacy label calls
2,353 Acute, 1,623 Post-Incident and **1,667 Pre-Incident** (ECG > 6 h before the troponin-defined onset).
For admissions where MI is the reason for coming in, the first positive troponin (often the second draw,
needed for the rise/fall delta) comes after true onset, which is the roadmap's argument for not drawing the
line at the troponin timestamp. Sensitivity *after-strict*: after-onset admissions where the legacy label
also says Acute or Post-Incident (3,976).

## 5. ECG features (`p2_04_features.json`, `ecg_feature_list.json`)

- The 95 features (53 point, 41 `_iqr`, 1 `_count`) are present in the raw full feature file and
  **complete for all 193,470 candidate admissions**. The list is locked in `ecg_feature_list.json`.
- **The legacy master cannot be the Layer 2 base.** It dropped 31,444 "MI_Post-Incident" ECGs, the
  troponin-defined post-onset ECGs: for 29% of after-onset MI admissions the first feature ECG is missing
  from the master (2,479 MI admissions in total). Phase 3 must build from the raw full feature file plus
  this linkage, applying preprocessing fitted on the training split. This departs from the roadmap's
  "reuse the master" instruction, because the master's ECG selection depends on the troponin timing the
  roadmap rules out.
- **Plausibility against the ECG machine's own measurements** (193,220 ECGs): RR interval agrees closely
  (Spearman 0.988; medians 768 vs 769 ms). **QRS duration does not**: ECGdeli global QRS median 162 ms vs
  machine 93 ms (Spearman 0.69). It is likely a multi-lead earliest-onset to latest-offset span.
  `QRS_Dur_Global` effects must be read with caution, and the machine QRS should be added as a validation
  outcome.
- Beat count (`QRS_Dur_Global_count`): median 13 beats, 1st percentile 8.

## 6. β-VAE provenance (`p2_05_vae_provenance.json`)

| Check | Result |
|---|---|
| Checkpoint | `vae_zdim64_beta4.0_20260821_030245/best_model.pt`, epoch 109 (z = 64, β = 4, seed 42) |
| Config vs split files | Train/validation/test counts match exactly (262,813 / 56,319 / 56,798); embedding IDs match split IDs |
| Re-encode 768 ECGs vs stored embeddings | Max abs difference 2.2 × 10⁻⁶ (float rounding); per-dimension correlation 1.000 |
| HDF5 signals vs raw WFDB (12 ECGs) | Max abs difference 4.8 × 10⁻⁷ after the documented per-lead z-score |
| Labels in training loss | None (code review of `train_vae.py`, `vae_conv1d.py`) |
| Training composition | Train split includes 5,300 acute-MI and 4,179 pre-incident ECGs (unsupervised); post-incident ECGs excluded |

**Decision: the 64 latents are eligible for the Layer 2 latent audit.** Two caveats to report: the encoder
saw acute-MI ECGs during unsupervised training, and latents exist only for the 377,113 ECGs in the HDF5
file. The post-incident first ECGs need encoding from WFDB with the verified preprocessing in Phase 3.

The data loader that produced the embeddings (`src/data/ecg_dataset.py`) was never under version control,
because the `data/` rule in `.gitignore` also matches `src/data/`. It is now added.

## 7. Carried into Phase 3

1. Build Layer 2 from the raw full feature file + `admission_ecg.parquet`, not the legacy master.
2. Reuse `split_manifest.csv` for patients already assigned; assign remaining patients by the same
   patient-level rule and record it.
3. Fit winsorization and scaling on the training split only.
4. Encode latents for first feature ECGs missing from the HDF5 file.
5. Add machine-measured QRS duration and RR as validation outcomes.
6. Keep sensitivity flags: strict history, look-back, after-strict timing, first-ECG-has-features,
   NSTEMI-only, type 2 excluded, chest-pain comparator, before-onset negative control.
