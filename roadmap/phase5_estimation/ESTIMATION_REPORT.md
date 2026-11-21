# Phase 5 — Estimation Report (both layers)

**Date:** 2026-11-21 · **Code:** `roadmap/estimators.py` (shared), `p5_01` → `p5_04` in this folder
**Outputs:** `LAYER1_ESTIMATES.csv`, `LAYER1_DIAGNOSTICS.csv`, `LAYER2_ESTIMATES.csv`, `LAYER2_DIAGNOSTICS.json`,
`LAYER2_TEST_STABILITY.csv`, `p5_0*.json`. Nuisance predictions for Phases 6–7: `data/roadmap/phase5/` (git-ignored).

## 1. Method

- **Estimator:** augmented inverse-probability weighting (doubly robust). Layer 1: risk difference and risk
  ratio. Layer 2: ATT (primary) and ATE (secondary) for every outcome.
- **Cross-fitting:** 5 folds, patients kept whole within a fold, over all splits (seed 20261121).
- **Uncertainty:** 1,000 patient-level bootstrap resamples (a patient's admissions move together; nuisances
  held fixed); percentile 95% intervals.
- **Multiplicity:** Benjamini–Hochberg within each family (53 point features = primary; 41 `_iqr`; 64 latents).
- **Model specification** (`p5_01`, fitted on train, scored on validation): cubic age spline (5 df) with
  age × each covariate and pairwise covariate interactions. It was chosen for both layers. With fewer
  interactions, the CKD and prior-MI propensity models left age unbalanced.
- **Covariates:** Phase 4 adjustment sets with primary flags. Layer 2 outcomes are the winsorized features;
  raw features are a sensitivity (correlation of effects 0.94).

## 2. Data correction found in this phase: aVL and aVF were swapped

The limb-lead identities (aVF = (II+III)/2) failed in the extracted features (r = −0.12 for ST).
MIMIC-IV-ECG files store leads as I, II, III, aVR, **aVF, aVL**, but `src/matlab/extract_ecg_features.m`
labels rows positionally as I, II, III, aVR, **aVL, aVF**. After swapping the two labels, aVF tracks
(II+III)/2 with r = 0.87 (ST), 0.90 (T), 0.90 (R).

- Fixed at read time with `fix_lead_labels()` in `roadmap/definitions.py`; the raw file is unchanged.
- Phase 3 was rebuilt, and a permanent gate check guards the identity.
- Latents and machine measurements come straight from the waveforms and are unaffected.
- **The legacy master dataset, baseline models and feature-importance results carry the swapped labels.**

## 3. Layer 1 — chronic conditions and acute MI at admission

**Primary estimand:** ATE in the overlap region (propensity 0.01–0.99). For diabetes and hypertension
this equals the full-population ATE (no admission outside overlap). For CKD and prior MI the full ATE
fails positivity: 18% and 22% of admissions, mostly young people without comorbidity, have almost no chance
of exposure. There the overlap population is 82% and 78% of admissions. 430,852 admissions; acute MI 2.60%.

| Exposure | Adjusted for | Role | Crude RD (pp) | **Adjusted RD (pp)** | **RR (95% CI)** | ATT RR |
|---|---|---|---:|---:|---:|---:|
| Diabetes | age, sex | bracket (lower set) | 2.49 | 1.44 (1.30–1.58) | **1.67 (1.60–1.75)** | 1.63 |
| Diabetes | age, sex, HTN | bracket (upper set) | 2.49 | 1.20 (1.07–1.35) | **1.54 (1.48–1.62)** | 1.50 |
| Hypertension | age, sex, diabetes | primary | 2.74 | 0.89 (0.74–1.02) | **1.44 (1.35–1.53)** | 1.38 |
| Hypertension | age, sex (roadmap) | sensitivity | 2.74 | 1.22 (1.10–1.34) | 1.68 (1.59–1.78) | 1.61 |
| CKD | age, sex, D, H | primary | 3.25 | 0.96 (0.76–1.16) | **1.34 (1.27–1.42)** | 1.42 |
| CKD | + prior MI | sensitivity | 3.25 | 0.64 (0.46–0.84) | 1.22 (1.16–1.29) | 1.28 |
| Prior MI | age, sex, D, H, CKD | descriptive | 5.59 | 3.81 (3.37–4.21) | **2.35 (2.19–2.50)** | 2.04 |

All contrasts are positive and clinically ordered. The previous attempt's inverse hypertension estimate
does not reappear: it came from ECG-only restriction, strict prior-only flags and the hypertension set.
Diabetes is reported as a bracket (Phase 4: its total effect is not identified under the revised graph).

**Diagnostics** (`LAYER1_DIAGNOSTICS.csv`): weighted max |SMD| ≤ 0.013 for every primary estimand (ATE
in overlap and ATT); effective sample sizes ≥ 17,881 exposed.

## 4. Layer 2 — acute MI and the ECG (ATT, after-onset ECGs)

188,808 admissions (5,643 after-onset MI; 98,662 patients). Propensity among MI admissions 0.0005–0.08,
so overlap is complete on the control side. Largest control odds weight 0.089; effective controls 139,545;
weighted max |SMD| 0.002.

**Primary family (53 point features): 34 significant at FDR 5%.**

| Feature group | Features | Significant | Mean \|ATT\| (SD) | Largest \|ATT\| (SD) |
|---|---:|---:|---:|---:|
| T amplitude | 12 | 10 | 0.192 | 0.335 |
| ST level | 12 | 12 | 0.184 | 0.361 |
| RR interval | 1 | 1 | 0.103 | 0.103 |
| QRS duration | 1 | 1 | 0.093 | 0.093 |
| Q amplitude | 12 | 6 | 0.039 | 0.094 |
| R amplitude | 12 | 4 | 0.029 | 0.089 |
| S amplitude | 3 | 0 | 0.025 | 0.035 |

**Largest effects (control-SD units, 95% CI):** ST in I −0.36 (−0.40, −0.33); ST in aVL −0.34
(−0.38, −0.30); T in V5 −0.34 (−0.37, −0.30); T in V6 −0.31; T in aVR +0.29; T in I −0.29;
ST in aVR +0.24; ST in III +0.23.

**Territory composites:** lateral ST −0.36 SD (−0.016 mV); maximum ST depression −0.23 SD; maximum ST
elevation +0.15 SD; inferior ST +0.11 SD; anterior ST −0.04 SD.

The pattern is lateral ST depression and T-wave flattening with reciprocal aVR elevation, the picture
expected in a cohort that is three-quarters NSTEMI. R, S and Q amplitudes barely move.

**Crude vs adjusted:** effects correlate at 0.93; 44 of 53 shrink after adjustment (median |effect| 0.115 →
0.072 SD); 7 change sign. Phase 6 analyses this.

**Machine-measured validation:**

| Measure | ECGdeli ATT | Machine ATT | Verdict |
|---|---|---|---|
| RR interval | +19.4 ms | +19.5 ms (14.7–24.0) | Validated |
| QRS duration | +2.1 ms (0.09 SD) | +0.4 ms (−0.9 to 1.6) | **Not confirmed** |

The ECGdeli QRS effect should not be reported as an MI effect.

**Secondary families:** `_iqr` 35/41 significant (mean |ATT| 0.048 SD); latents 43/64 significant (mean
0.062, max 0.20 SD), analysed in the Phase 6 latent audit.

**ATE (secondary):** the ATE weights leave age unbalanced (SMD 0.19; 13% of propensities clipped), so the
ATE is reported for completeness only. The ATT is the inferential Layer 2 estimand (Phase 1, D3).

## 5. Held-out checks (`p5_04_heldout.json`, the single use of the test split)

Nuisances were fitted on train + validation and applied to test rows only.

| Pre-declared check | Result | Pass |
|---|---|---|
| H1: age ⊥ sex on test Layer 1 (\|SMD\| < 0.10) | 0.081 | ✅ |
| H2: every Layer 1 test RR on the same side of 1 | 7/7 (test RRs 1.22–2.17) | ✅ |
| H3: sign agreement ≥ 90% among significant features, median diff ≤ 0.05 SD | 34/34 agree; median 0.023 SD; r = 0.98 | ✅ |
