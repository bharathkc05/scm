# Phase 1 — Estimand Note

**Project:** Causal Model of Acute MI (roadmap of 2026-11-14)
**Data:** MIMIC-IV v2.2 (`hosp`), MIMIC-IV-ECG v1.0, existing master / splits / longitudinal files
**Date:** 2026-11-21 · **Status:** draft for review — locked once Gate 1 passes
**Code lists:** [`roadmap/definitions.py`](../definitions.py) (single source; later phases import it)

Neither question involves a treatment, a discharge time zero or follow-up. The unit is the
**admission** (`hadm_id`); patients stay whole within one split; inference is clustered by patient.

---

## 1. Shared definitions

| Variable | Definition | Admissions (of 430,852 with any code) |
|---|---|---:|
| **Acute MI at this admission** (`acute_mi`) | ICD-9 410.x0 / 410.x1, ICD-10 I21.\*, I22.\*, any diagnosis position. **Excludes ICD-9 410.x2** ("subsequent episode of care": follow-up admission for an earlier MI). | 11,222 (ICD-9 5,376 · ICD-10 5,846) |
| ↳ STEMI | ICD-9 410.0–410.6, 410.8; ICD-10 I21.0–I21.3 | 1,917 |
| ↳ NSTEMI | ICD-9 410.7; ICD-10 I21.4 | 7,611 |
| ↳ Type 2 MI | ICD-10 I21.A1 (exists only in ICD-10 years) | 1,465 |
| ↳ Other / unspecified | ICD-9 410.9; ICD-10 I21.9, I21.A9, I22 | 370 |
| Old (historical) MI code | ICD-9 412; ICD-10 I25.2 — never an outcome | 24,061 |
| Diabetes | ICD-9 250; ICD-10 E08–E11, E13 | 103,903 |
| Hypertension | ICD-9 401–405; ICD-10 I10–I13, I15, I16 | 216,447 |
| CKD | ICD-9 585; ICD-10 N18 | 62,558 |
| Chest pain (comparator) | ICD-9 786.5; ICD-10 R07 | 26,093 (25,973 without acute MI) |
| Age | `anchor_age + (year(admittime) − anchor_year)`; all admissions are adults | — |
| Sex | `gender` | — |

**Chronic-condition timing.**
- *Primary:* a code on any **earlier** admission of the same patient **or** a code at the current admission
  (chronic conditions are coded when present, so a same-admission code is treated as pre-existing).
- *Sensitivity (strict):* earlier admissions only. This is what the existing master flags use
  (`src/features/extract_comorbidities.py`, `admittime < current`).

**Prior MI.** Acute MI code on an earlier admission, **or** an old-MI code (412 / I25.2) on the
current or an earlier admission. Treated as a descriptive marker of coronary disease, not as a cause.

---

## 2. Layer 1 — Do chronic conditions relate to MI at admission?

**Population.** All adult MIMIC-IV admissions with at least one diagnosis code (430,852 admissions).
*Not* restricted to admissions with an ECG, because Layer 1 does not use the ECG and restricting it
would add ECG selection for no gain.

**Outcome.** `acute_mi` at the current admission.

**Estimand.** For each exposure X, the adjusted risk difference standardized to the study population:

  ψ₁(X) = E_L[ P(Y=1 | X=1, L) − P(Y=1 | X=0, L) ]

and the matching risk ratio. Interpretation: **risk-factor contrast among hospitalized adults**,
not the effect of an intervention. Age and sex are reported as standardized risk by age band and sex
(structural contributions), not as effects.

| Exposure | Adjust for (L) | Do not adjust for | Planned sensitivity |
|---|---|---|---|
| Diabetes | age, sex | CKD (downstream) | add hypertension (alternative DM/HTN ordering) |
| Hypertension | age, sex | CKD (downstream) | add diabetes (alternative ordering) |
| CKD | age, sex, diabetes, hypertension | prior MI | add prior MI (reverse ordering) |
| Prior MI | age, sex, diabetes, hypertension, CKD | anything recorded after the prior MI | descriptive only |

**Additional Layer 1 sensitivities (added after review).**
- *Look-back:* 41.9% of admissions (180,733 of 431,231) are a patient's first admission, so their
  prior-history flags are 0 by construction. Repeat with admissions that have ≥1 earlier admission
  (250,498), and with number of earlier admissions as a covariate.
- *Hospital-population selection:* everyone is hospitalized for some reason, so conditions that cause
  many non-MI admissions can look protective (Berkson-type bias). Report the ECG-restricted and
  chest-pain-restricted Layer 1 estimates for comparison, and state the bias in the limitations.

---

## 3. Layer 2 — What does MI do to the ECG?

**Population.** Adult admissions with at least one ECG recorded between ED registration (or admission
time if there was no ED visit) and discharge. **One ECG per admission: the earliest in that window.**
Primary analysis uses only ECGs classified **after onset** (rule set in Phase 2 from admission context,
not from troponin).

**Exposure (M).** `acute_mi` at the current admission.

**Comparators.**
- *Primary:* admissions without an acute MI code (prior MI enters as a covariate, per the roadmap).
- *Active comparator (sensitivity):* chest-pain admissions without an acute MI code. These patients are
  nearly all sent for an ECG, so this controls ECG selection by design rather than by weighting.

**Outcomes (F_j).**
- *Primary family:* the **53 point features** of the 95 handcrafted features (ST, Q, R, T, QRS, RR and
  other morphology) — false-discovery control within this family.
- *Secondary:* the 41 `_iqr` features (beat-to-beat spread); territory-level ST composites
  (inferior II/III/aVF, anterior V1–V4, lateral I/aVL/V5–V6) derived from the point features.
- *Conditional:* the 64 β-VAE latents, only if Phase 2 confirms provenance (checkpoint → stored
  embeddings reproduce).
- `QRS_Dur_Global_count` (beats detected) is a **quality flag, not an adjustment covariate**: MI can change
  heart rate and rhythm, so adjusting for it would adjust for a consequence of the exposure. It is used
  to define a high-quality-ECG sensitivity subset.

**Adjustment set.** age, sex, diabetes, hypertension, CKD, prior MI (Section 1 definitions).

**Estimand.** For each feature j, in control-SD units:

- *Primary — effect among MI admissions (ATT):*
  ψ₂ⱼ = E[ F_j(1) − F_j(0) | M=1 ] = E[F_j | M=1] − E_{L|M=1}[ E(F_j | M=0, L) ]
- *Secondary — population average (ATE).*

The ATT is primary because the question is what MI did to the ECGs of people who had one, and with
~4% exposure prevalence it needs overlap only on the control side (every MI profile has comparable
controls), which avoids the heavy propensity clipping seen in the previous attempt.

**Effect modifier.** Hours from ED registration (or admission) to the ECG, available for every
admission. `index_mi_time` is not used (troponin-based, missing for ~40% of acute ECGs).

**Pre-specified subgroup and control analyses** (run in Phases 6–7):
- *STEMI vs NSTEMI:* STEMI is diagnosed from ST elevation, so its ST "effects" are partly built into the
  label. NSTEMI (7,611 admissions) is the circularity-free test of the ST/T pattern.
- *Type 1 vs type 2 MI:* exclude I21.A1 as a sensitivity (different mechanism).
- *Negative control:* before-onset ECGs (from an earlier admission of a patient who later has an acute MI)
  vs. ECGs of patients who never do. A non-zero adjusted difference measures residual confounding and
  selection **plus** genuine pre-existing coronary disease, so it is read as an upper bound on bias.
- *Model-implied counterfactual ECGs:* exploratory appendix only.

---

## 4. Time order assumed by each question

| Layer | Order | How Phase 2 checks it |
|---|---|---|
| 1 | age, sex → chronic condition (earlier admission or chronic code) → acute MI coded at current admission | share of exposures supported by an earlier admission; strict vs primary flags |
| 2 | baseline factors → MI onset → ECG | timing classes: *before* (ECG from an earlier admission), *after* (MI admission arriving through the ED / emergency, ECG soon after arrival), *uncertain* (elective, surgical or transfer admissions where MI may start in hospital; late ECGs) |

---

## 5. Variables never used as covariates

`Label_Troponin`, `is_MI`, troponin values, `index_mi_time`, `time_diff_hours`, and every in-admission
treatment or support variable (`*_during`, `*_after`, vasopressors, mechanical ventilation) — these are
post-onset. Latents are never in a Layer 1 adjustment set. (Enforced list: `FORBIDDEN_COVARIATES`.)

---

## 6. Inherited-data notes for Phase 2

1. **Master `prior_mi` flag** counts earlier acute codes (410/I21/I22) and ICD-9 412, but **omits ICD-10
   I25.2**, and includes 410.x2 follow-ups. Rebuild prior MI from `diagnoses_all` with the Section 1 rule.
2. **Master chronic flags are strict prior-only.** Keep them as the strict sensitivity version; build the
   primary version (prior + same-admission chronic codes) from `diagnoses_all`.
3. **Master MI label** (`Label_Troponin`: `MI_Acute_Presentation`, `MI_Pre-Incident`, `Control_Symptomatic`)
   came from the ECG-linked ICD file plus troponin timing, not from the admission's `diagnoses_icd`.
   Phase 2 compares it with `acute_mi` and confirms how controls were chosen ("Symptomatic") and that the
   label was not read from the ECG.
4. The previous attempt counted 12,187 acute-MI admissions; the 965-admission difference is the 410.x2
   follow-up admissions now excluded.
