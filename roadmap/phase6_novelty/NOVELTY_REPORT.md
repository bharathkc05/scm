# Phase 6 — Novelty Analysis: the ECG as an effect of MI

**Date:** 2026-11-21 · **Code:** `p6_01` → `p6_05` in this folder · **Figures:** `figures/*.svg`
**Tables:** `SENSITIVITY_ESTIMATES.csv`, `DECONFOUNDED_ECG_EFFECTS.csv`, `FEATURE_GROUP_SUMMARY.csv`,
`LATENT_AUDIT.csv`, `COUNTERFACTUAL_PROFILES.csv`, plus `p6_0*.json`.

All Layer 2 numbers are ATT estimates in control-SD units (Phase 5 estimator, specification and covariates),
in after-onset MI admissions versus controls.

## 1. Gate 1 condition and stability (`p6_01_sensitivity.json`, Figure 2)

| Analysis set | MI cases | Sign agreement, significant features | ST/T magnitude vs primary | Significant (of 53) |
|---|---:|---:|---:|---:|
| Primary | 5,643 | — | 1.00 | 34 |
| **NSTEMI only** (Gate 1 condition) | 4,149 | 91% (ST/T: 95%) | **1.25** | 42 |
| Chest-pain comparator | 5,643 | 94% (ST/T: 100%) | 1.36 | 45 |
| After-strict timing | 3,976 | 100% | 0.98 | 34 |
| Type 2 MI excluded | 5,563 | 100% | 1.00 | 34 |
| First ECG has features | 4,787 | 100% | 1.02 | 32 |

- **The Gate 1 condition is met.** In NSTEMI, which is not defined by ST elevation, the ST/T pattern keeps
  its direction and is stronger (lateral ST −0.50 SD). The only feature that disappears is maximum ST
  elevation (+0.15 → −0.01 SD), as expected, since ST elevation defines STEMI. The ST-depression and T-wave
  results are not an artefact of how STEMI is coded.
- Against chest-pain controls (similar ECG and troponin work-up), effects are larger, not smaller, so ECG
  selection does not create them.

**Negative control** (an ECG from an earlier, non-MI admission of a patient who later has a first MI, vs
patients who never do): ST/T magnitude is **38% of the primary**; 15 of 53 features significant; lateral ST
−0.10 SD. This bounds residual confounding, selection and genuine pre-existing coronary disease together;
about 60% of the ST/T effect is specific to the acute event.

## 2. Result 1 — Deconfounded ECG effects of MI (`DECONFOUNDED_ECG_EFFECTS.csv`, Figure 1)

Classification of the 53 point features (rule fixed in `p6_03`):

| Group | Features | MI-attributable | Partly confounded | Explained by confounding | Revealed by adjustment | Robust* | Mean \|crude\| → \|adjusted\| (SD) |
|---|---:|---:|---:|---:|---:|---:|---|
| ST | 12 | 11 | 1 | 0 | 0 | 7 | 0.237 → 0.184 |
| T | 12 | 10 | 0 | 1 | 0 | 5 | 0.245 → 0.192 |
| Q | 12 | 3 | 1 | 5 | 2 | 2 | 0.074 → 0.039 |
| R | 12 | 1 | 0 | 8 | 3 | 0 | 0.100 → 0.029 |
| S | 3 | 0 | 0 | 2 | 0 | 0 | 0.058 → 0.025 |
| QRS duration | 1 | 0 | 1 | 0 | 0 | 1 | 0.267 → 0.093 |
| RR interval | 1 | 1 | 0 | 0 | 0 | 0 | 0.161 → 0.103 |

\*Robust = significant after adjustment, same sign in NSTEMI, and negative-control ratio < 0.5.

**Findings:**
- **ST and T changes are caused by MI** in this framework: 21 of 24 keep at least half their crude size after
  adjustment.
- **Crude R, Q and S amplitude differences mostly reflect who has an MI** (age, sex, comorbidity), not the MI:
  15 of the 16 "explained by confounding" features are in these groups. A prediction model would use them;
  a causal reading should not.
- **Robust MI signs (14 features):** ST in I, aVL, V4, V5, V6, III, aVR; T in V4, V5, V6, aVR, II; Q in I
  and aVL.
  - QRS duration also passes but is excluded, because the machine QRS did not confirm it (Phase 5, R5).
  - RR is excluded because it is equally present before the MI (negative-control ratio 1.05); it reflects
    the patient, not the event.
- **Robust composites:** lateral ST (−0.36 SD; NSTEMI −0.50; negative control −0.10) and maximum ST
  depression (−0.23 SD).

## 3. Result 2 — Causal audit of the β-VAE latents (`LATENT_AUDIT.csv`, Figure 3)

Cross-fitted R² per latent in the primary set:

| Measure | Value |
|---|---|
| Latents with significant MI effect (BH 5%) | 43 / 64 |
| Median variance explained by baseline factors | 0.60% |
| Median extra variance explained by MI | 0.009% (max 0.12%) |
| Latents where MI explains more than baseline factors | 1 / 64 |
| Share of MI's crude signal retained after adjustment (median) | 62% |
| Largest latent effect | 0.20 SD (z57), vs 0.36 SD for the top handcrafted feature |
| Out-of-fold AUC for MI from baseline-residualized latents / handcrafted point features | 0.659 / 0.653 |

**Findings:**
- The latents encode ECG variation that is almost entirely unrelated to MI or to the measured baseline
  factors.
- The MI signal that exists concentrates in a few dimensions that track T-wave amplitude (z57 and z3 with T
  in aVR; z11 with T in I; z31 with T in aVR), the same physiology as the handcrafted results.
- Beyond baseline factors, the latents carry about the same MI information as the 53 handcrafted features,
  not more.
- **For later work, `z_ecg` should not be treated as an MI-specific node.** Its significant effects come
  from sample size, not from a strong MI signal. Caveat: the encoder saw acute-MI ECGs during unsupervised
  training.

## 4. Result 3 — Model-implied counterfactual ECGs (exploratory; `COUNTERFACTUAL_PROFILES.csv`, Figure 4)

Expected lateral ST level with and without MI from the fitted outcome models (200 patient-bootstrap refits):

| Profile | Without MI (mV) | With MI (mV) | Effect (SD) | 95% CI (mV) |
|---|---:|---:|---:|---|
| 55-year-old man, no comorbidity | 0.015 | 0.008 | −0.17 | −0.013 to −0.001 |
| 65-year-old man, diabetes + HTN | 0.003 | −0.012 | −0.35 | −0.020 to −0.010 |
| 75-year-old woman, HTN + CKD | −0.011 | −0.032 | −0.50 | −0.033 to −0.011 |
| 70-year-old man, D + H + CKD + prior MI | −0.015 | −0.049 | −0.79 | −0.042 to −0.027 |

The model implies a larger lateral ST shift in older, more comorbid patients. These are conditional model
predictions, not observed counterfactuals, and not a tested effect-modification result; subgroups are
tested in Phase 7.

Per patient, only 20–25% of MI admissions deviate by more than 1 control SD from their own model-implied
no-MI ECG in the direction of the effect. The MI shift is small relative to differences between people,
which is why single features detect MI poorly despite clear average effects.
