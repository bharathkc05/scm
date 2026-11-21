# Phase 7 — Stress Tests

**Date:** 2026-11-21 · **Code:** `common.py`, `p7_01` → `p7_06` in this folder (all reuse the Phase 5 estimators)
**Tables:** `EVALUES.csv`, `SELECTION_WEIGHTED_LAYER2.csv`, `BMI_LAYER2_*.csv`, `OUTCOME_CODES_*.csv`,
`SUBGROUPS_*.csv`, `SUBGROUP_INTERACTIONS_LAYER2.csv`, `LAYER1_HISTORY_VARIANTS.csv`, `p7_0*.json`.
Layer 1 estimates are risk ratios (ATE in overlap, 500 patient-bootstrap resamples). Layer 2 estimates are
ATT in control-SD units on the 16 robust measures from Phase 6 (14 point features + lateral ST + maximum ST
depression).

## 1. What survives — summary

| Result | Unmeasured confounding (E-value, CI) | ECG selection | BMI | MI code definitions | Age / sex subgroups | History definition | Verdict |
|---|---|---|---|---|---|---|---|
| Diabetes → MI (RR 1.54–1.67) | 2.3–2.6 | n/a | ✅ | ✅ all (1.37–1.69) | ✅ all | ✅ look-back; ❌ strict flags* | **Survives** |
| Hypertension → MI (RR 1.44) | 2.0 | n/a | ✅ | ✅ all (1.34–1.49) | ✅ all | ✅ look-back; ❌ strict flags (0.72)* | **Survives** |
| CKD → MI (RR 1.34) | 1.9 | n/a | — | ❌ principal-only 0.97; type-1 1.21 | ✅ | ✅ look-back; ❌ strict 1.04 | **Fragile** |
| Prior MI → MI (RR 2.35, descriptive) | 3.8 | n/a | — | ✅ | ✅ | ✅ | Survives (descriptive) |
| MI → ST/T pattern (16 measures) | 1.5–2.0 (ST/T); 1.2–1.3 (Q) | ✅ 16/16 | ✅ 16/16 sign, 15/16 CI | ✅ broad 16/16 | ✅ 16/16 in each subgroup | n/a | **Survives** (negative-control bound 38%, Phase 6) |

\*Strict flags are 0 by construction on a patient's first admission, where MI is more common (Section 6).

## 2. Unmeasured confounding (`EVALUES.csv`)

E-value = the minimum risk-ratio association an unmeasured confounder would need with both exposure and
outcome to explain the estimate away (CI column: to move the CI to include the null).

| Layer 1 | RR | E-value | E-value (CI) |
|---|---:|---:|---:|
| Diabetes \| age, sex | 1.67 | 2.73 | 2.58 |
| Diabetes \| age, sex, HTN | 1.54 | 2.46 | 2.31 |
| Hypertension \| age, sex, diabetes | 1.44 | 2.24 | 2.03 |
| CKD \| age, sex, D, H | 1.34 | 2.02 | 1.86 |
| Prior MI (descriptive) | 2.35 | 4.12 | 3.81 |

**Layer 2** (d converted to RR ≈ exp(0.91·d)): lateral ST 2.13 (CI 2.03); ST in I 2.12; ST in aVL 2.07;
T in V5 2.05; T in V6 1.97; T in aVR 1.93; down to 1.34–1.37 for Q in I and aVL.

For comparison, the strongest measured risk factor after adjustment, diabetes, has RR 1.67. A hidden
confounder of the ST/T effects would need to be stronger than that with both MI and the ECG. Smoking or
lipids at that strength are possible, so this is a bound, not a proof.

## 3. ECG selection (`p7_02_selection.json`)

- **Setup:** 425,146 admissions would enter the primary set if they had a feature ECG; 188,808 (44.4%) did.
  The selection model used age, sex, history flags, acute MI, ED arrival, planned admission, admission type
  and look-back.
- **Weights:** stabilized weights had a 99th percentile of 2.8 and maximum 10.1; effective sample 152,063.
- **Result:** all 16 robust measures keep their sign with CIs excluding 0; ST/T magnitude 0.99× primary;
  lateral ST −0.36 SD (−0.40, −0.32).
- **Limit:** valid only if ECG recording is ignorable given these variables. The chest-pain comparator
  (Phase 6, 1.36×) addresses the same threat by design.

## 4. BMI (`p7_03_bmi.json`)

- **Linkage:** pre-admission OMR BMI (within 3 years) for 55% of admissions (44% of MI admissions).
- **Method:** estimates without and with BMI categories on the same admissions.
- **Layer 1:** diabetes 1.91 → 1.94; diabetes \| HTN 1.73 → 1.75; hypertension 1.67 → 1.70.
- **Layer 2:** ST/T magnitude 0.999× (16/16 signs; lateral ST −0.44 SD both ways).
- **Conclusion:** BMI does not confound either layer measurably. BMI-available admissions show somewhat
  larger effects than the full cohort (a different, more frequently followed population).

## 5. MI code definitions (`OUTCOME_CODES_*.csv`)

| Layer 1 outcome | MI events | Diabetes | Diabetes \| HTN | Hypertension | CKD | Prior MI |
|---|---:|---:|---:|---:|---:|---:|
| Primary (acute, any position) | 11,222 | 1.67 | 1.54 | 1.44 | 1.34 | 2.35 |
| Broad (+ 410.x2 follow-ups) | 12,187 | 1.69 | 1.55 | 1.49 | 1.35 | 2.94 |
| Type 1 only (I21.A1 excluded) | 9,757 | 1.59 | 1.48 | 1.40 | 1.21 | 2.34 |
| Principal diagnosis only | 6,640 | 1.44 | 1.37 | 1.34 | **0.97** (0.91–1.05) | 1.99 |

**CKD is the only fragile Layer 1 result.** Its association disappears when only MIs that are the reason for
admission count, and it shrinks without type 2 MI. CKD is linked to MI coded as a secondary diagnosis
(in-hospital or demand-related), not to MI presentations.

Layer 2 with the broad exposure (6,448 cases incl. follow-ups): 16/16 robust, magnitude 0.97×.

## 6. History definitions (`LAYER1_HISTORY_VARIANTS.csv`)

| Variant | Diabetes | Diabetes \| HTN | Hypertension | CKD | Prior MI |
|---|---:|---:|---:|---:|---:|
| Primary (prior + same-admission codes) | 1.67 | 1.54 | 1.44 | 1.34 | 2.35 |
| Strict flags (earlier admissions only) | 1.15 | 1.03 | **0.72** | 1.04 | 2.29 |
| Restricted to admissions with look-back | 2.03 | 1.79 | 2.24 | 1.77 | 3.72 |
| Number of earlier admissions as covariate | 1.90 | 1.71 | 1.86 | 1.65 | 2.71 |

- The strict-flag analysis reproduces the earlier "protective hypertension" result (0.72) and shows where it
  comes from. Strict flags are 0 on every first admission, and MI is more common on first admissions (3.27%
  vs 2.12%, Phase 2). Restricting to admissions with look-back, or adjusting for it, makes every association
  *larger* than the primary estimate.
- **The primary estimates are conservative.** Strict prior-only flags should not be used without handling
  look-back.

## 7. Subgroups (`SUBGROUPS_*.csv`, `SUBGROUP_INTERACTIONS_LAYER2.csv`)

**Layer 1:** every RR exceeds 1 in every subgroup.
- Diabetes, hypertension and prior MI are relatively stronger under 65 (e.g. prior MI 3.32 vs 1.95).
- Diabetes and CKD are relatively stronger in women (diabetes 1.92 vs 1.53).

**Layer 2:** all 16 robust measures keep their sign with CIs excluding 0 in each of the four subgroups.
Nine interactions pass FDR 5% across the 16 measures × 2 modifiers:

| Modifier | Measure | Effect (SD) | Reading |
|---|---|---|---|
| Age | ST in V5 / V6 | < 65: −0.11 / −0.12 vs ≥ 65: −0.29 / −0.29 | Lateral precordial ST depression larger in older patients |
| Age | Lateral ST composite | −0.29 vs −0.40 | Same |
| Age | ST in aVR | +0.16 vs +0.27 | Larger reciprocal aVR elevation in older patients |
| Age | ST in III | +0.36 vs +0.17 | Inferior ST change larger in younger patients |
| Age | ST in aVL | −0.44 vs −0.30 | High-lateral ST depression larger in younger patients |
| Sex | T in aVR | women +0.37 vs men +0.24 | T-wave changes larger in women |
| Sex | T in V5 | −0.39 vs −0.30 | Same |
| Sex | Max ST depression | −0.29 vs −0.19 | ST depression larger in women |

The pattern is the same everywhere; the size varies. Older patients show more diffuse lateral and aVR
changes; younger patients show more inferior / high-lateral changes (closer to STEMI territory patterns).
These are descriptive modifications under the same assumptions as the main estimates.
