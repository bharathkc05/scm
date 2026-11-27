# Phase 10 report: individual counterfactual ECGs

**Question.** Can the study answer counterfactual and interventional questions about **one patient**? For example:
"What would this patient's ECG look like without their MI?", "What would a healthy patient's ECG look like with an
anterior STEMI?" and "Was this patient's diabetes necessary for their MI?"

**Answer.** Yes, at the level of the 16 robust ECG measures and for profile-specific clinical effects, but
**imprecisely**.
- On real held-out patients the best individual model beats both simple baselines, so it does use what is specific
  to the patient.
- About 0.7–0.9 SD of error per ST/T measure remains, because two ECGs of the same patient differ that much without
  any MI.
- Every answer therefore comes as a range with its validation error.
- Waveform counterfactuals were built and look physiologically right, but they were not validated and stay local.

Plan: [ROADMAP.md](ROADMAP.md). Decisions and every gate: [GATE_DECISION.md](GATE_DECISION.md).

---

## 1. Data

| Step | Output |
|---|---|
| **CF1 cohort** (DuckDB over MIMIC-IV v2.2, MIMIC-IV-ECG, ECG-Ext-ICD) | 11,222 acute-MI admissions (9,459 patients). ECG index of 271,341 ECGs. **6,205 within-patient pairs** (5,159 patients): one ECG during the MI and a baseline ECG more than 24 h before arrival or more than 30 days after discharge |
| **Split** | Pair patients are held out at 80/20: `pair_test` (4,128 patients), `pair_val` (1,031). Everyone else keeps their Phase 3 split. No pair patient is in any training set |
| **CF2 signal store** (legacy ECGdeli chain in MATLAB) | 117,625 ECGs: every pair ECG, every during-MI ECG, the other ECGs of non-pair MI patients, 35,000 sampled controls and PTB-XL. 12-lead median beats, 95.4% passing quality control (MIMIC). Leads are reordered by header name, which fixes the Phase 5 aVL/aVF swap. Features match the legacy feature file: 98% of values identical, r = 0.97–1.00 except where a few impossible ECGdeli ST values (up to 21 mV) pull V5 and V6 down to 0.78–0.82; those are set to missing |
| **Primary test pairs** | After-onset MI, before-MI baseline not after an earlier MI, both ECGs passing QC: **1,378 pairs from 1,313 patients** (NSTEMI 1,084, STEMI 243) |

## 2. Causal framing

An individual counterfactual takes three steps (Pearl):
1. **Abduction:** infer what is specific to the patient.
2. **Action:** set MI on or off, and choose its type.
3. **Prediction:** recompute the ECG while keeping what is specific to the patient.

The within-patient pairs are the **answer key**. "Remove MI" from the during-MI ECG should land near the same
patient's real baseline ECG, and "add MI" to the baseline should land near their real during-MI ECG.

| ID | Assumption | Status |
|---|---|---|
| A2–A4 | Inherited from Phases 1–9: after-onset timing, no unmeasured MI–ECG confounder, ignorable ECG recording | Phase 7 negative control bounds residual bias at about 38% of the ST/T effect |
| CF-A2 | What is specific to the patient does not change with MI | Track A: the share ρ is estimated, not assumed (§4.1). Track B: z adds no MI information beyond the covariates (§4.4) |
| CF-A3 | The MI effect depends only on the covariates and MI type | Not identifiable; bounded by the pair error |
| CF-A4 | Another visit's ECG stands in for "this patient without MI" | Ageing, drugs and disease between visits count as error; reported by time gap |

## 3. Methods

**Track A (CF3), feature-level.**
- **Measures:** the 16 measures that passed every Phase 5–7 check: ST level in I, aVL, V4–V6, III and aVR; T
  amplitude in II, aVR and V4–V6; Q amplitude in I and aVL; lateral ST mean; maximum ST depression.
- **Structural model:** OLS fitted on the Phase 10 training split only, with the Phase 5 design (age spline, sex,
  diabetes, hypertension, CKD, prior MI, interactions):
  - m0 (no MI) on 23,152 controls;
  - m1 (MI) on 2,092 MI episodes, with MI-type dummies (NSTEMI, anterior / inferior / other STEMI, type 2, other).
- **Counterfactual:** `y_cf = m_target(L) + ρ · (y_obs − m_source(L))`.
  - **A1** (pre-specified): ρ = 1.
  - **A2**: ρ per measure and direction, fitted on `pair_val` only.
- **Baselines:** "no change" (copy the source ECG) and "population mean" (`m_target(L)`, which is what Phase 9's
  `explain_ecg` gives).
- **Intervals:** split conformal at 90%, calibrated on `pair_val`.

**Track B (CF4), waveform-level.**
- A conditional β-VAE (7.6M parameters) on 12-lead median beats.
- The condition is MI state, covariates and heart rate. Heart rate is held fixed, so counterfactuals change beat
  shape, not rhythm.
- A gradient-reversal adversary pushes MI information out of the latent z.
- The counterfactual beat is the observed beat plus the decoder's change.

**Clinical do() (CF6).** These come from the Phase 5 Layer 1 outcome models, refitted on the Phase 3 training split:
- the profile-specific risk ratio under do(diabetes / hypertension / CKD), with an interval from 200 draws of the
  patient-clustered sandwich covariance;
- the probability of necessity, PN = 1 − 1/RR. It assumes the condition never prevents MI (monotonicity) and no
  unmeasured confounding.

## 4. Results

### 4.1 Individual counterfactuals beat both baselines, but only after shrinkage (claims C21, C22)

Mean absolute error over the 14 ST/T measures, in control-SD units, on the primary test pairs:

| Direction | **A2 (shrunk)** | A1 (additive, pre-specified) | Population mean | No change |
|---|---|---|---|---|
| Add MI (before ECG → during-MI ECG) | **0.94** | 1.01 | 1.02 | 1.01 |
| Remove MI (during-MI ECG → before ECG) | **0.68** | 1.01 | 0.73 | 1.01 |

- **A2 vs the baselines** (95% patient-bootstrap CIs):
  - Adding MI: −0.075 [−0.093, −0.056] vs no change, and −0.079 [−0.090, −0.068] vs the population mean.
  - Removing MI: −0.33 vs no change, and −0.044 [−0.055, −0.033] vs the population mean.
- **Per measure**, A2 is never worse than either baseline. It beats both on 14 of 14 ST/T measures when adding MI,
  and on 14 and 12 of 14 when removing it.
- **Gate CF-G1:** passed by A2, failed by A1. The sensitivity sets (all before baselines, 3,299 pairs; all after
  baselines, 3,000 pairs) show the same pattern.
- **Why A1 fails:** only 20–76% of a patient's deviation from the expected ECG persists between visits (ρ on
  `pair_val`). Copying all of it carries about 1 SD of visit-to-visit noise into the counterfactual. A2 is the
  regression-to-the-mean correction.

### 4.2 Intervals are honest overall, too narrow for STEMI (claim C23)

- **Coverage:** 90% conformal intervals covered 89% of real values when adding MI and 92% when removing it.
- **Width:** the median half-width is 1.9 SD (add MI) and 1.6 SD (remove MI).
- **STEMI:**
  - Errors are larger.
  - **Anterior STEMI coverage is only 77%** (87 pairs).
  - `pair_val` has only 38 anterior STEMI pairs, too few to calibrate STEMI separately. Every STEMI answer from the
    tools carries this warning.
- **By time gap** (1 day to 10 years between ECGs) the pattern is stable. A2 helps most for short gaps.

### 4.3 Consistent with the average effect

Averaged over the primary MI episodes, Track A's patient-level effects have the **same sign as the Phase 5 ATT on 16
of 16 measures**, at similar size. For example, lateral ST: −0.43 SD (Track A) vs −0.36 SD (ATT). Model uncertainty in
one patient's effect is about 0.2 SD, small next to the visit-to-visit variation.

### 4.4 Waveform counterfactuals: plausible, not validated (claim C24)

- **Reconstruction:** 0.068 of the variance left unexplained, against 0.278 for the study's earlier full-strip VAE.
- **Reproducibility:** identical weights from the same seed.
- **MI leakage into z:**
  - z alone predicts MI at AUROC 0.59, which fails the roadmap's ≤ 0.55 criterion as first written.
  - That information is mostly duplicated age. **Given the covariates, z adds +0.011 AUROC (95% CI −0.004 to
    0.024).** The conditional criterion was accepted after the results were seen, and is labelled as such.
- **Counterfactual direction is territory-specific:**
  - removing an anterior STEMI lowers ST in V2–V4;
  - removing an inferior STEMI lowers ST in II, III and aVF, with reciprocal changes;
  - removing an NSTEMI lifts the lateral and precordial ST depression.
- **Counterfactual size:** the average shift is about **half** the Phase 5 ATT.
- **Not validated:** CF5 was skipped because you did not need waveforms, so by the roadmap's rule (gate CF-G2)
  Track B is **not served** and stays a local research artifact.

### 4.5 Clinical what-ifs for one profile (claim C25)

The Layer 1 models give profile-specific effects. For example, for a 70-year-old woman with diabetes and
hypertension:
- **Diabetes risk ratio:** 1.96 (1.70–2.21), against a study average of 1.67;
- **Probability that diabetes was necessary for her MI:** 0.49.

Diabetes is reported with both Phase 5 adjustment sets, as a bracket (the alternative set gives 1.84), because its
total effect is not identified. These numbers cannot be checked per patient, since nobody's diabetes switches off.

## 5. What can and cannot be asked about one patient

| Question | Possible? | Answered by |
|---|---|---|
| This MI patient's ECG measures without the MI | Yes, validated (§4.1) | `counterfactual_ecg` (typed-in features); `patient_explorer.py` (real patients, local) |
| A no-MI ECG's measures with an NSTEMI / anterior / inferior STEMI | Yes, validated (§4.1); STEMI intervals optimistic | Same |
| Effect of MI on the ECG for patients like this one | Yes | Track A τ(L, type); `explain_ecg` profile mode |
| Risk ratio under do(diabetes / hypertension / CKD) for this profile | Yes, under A1, A5 and A6 | `clinical_what_if` |
| Was the condition necessary for this patient's MI? | As a probability (PN), under monotonicity | `clinical_what_if` |
| The whole beat with or without MI | Built, plausible, not validated | `patient_explorer.py` only (local) |
| This individual's absolute probability of MI | Not served (server rule) | The local explorer shows it for research |
| "If this ECG were normal, would the MI not have happened?" | No: ECG changes are effects of MI, not causes | — |
| "If this patient were 20 years younger…" | No: age is not an intervention | — |

## 6. How to use it

**Real patients: local only.** The explorer prints only the file path. Its reports go to the git-ignored
`data/roadmap/phase10/explorer/` and must not pass through the MCP server or any cloud service.

```
python -X utf8 -m roadmap.phase10_counterfactual_ecg.patient_explorer --random --kind pair      # checked against the real other ECG
python -X utf8 -m roadmap.phase10_counterfactual_ecg.patient_explorer --random --kind mi        # without MI
python -X utf8 -m roadmap.phase10_counterfactual_ecg.patient_explorer --random --kind control   # with NSTEMI / anterior / inferior STEMI
python -X utf8 -m roadmap.phase10_counterfactual_ecg.patient_explorer --subject <subject_id>
```

Random selection draws from held-out patients. Each report flags in-sample patients.

**Any MCP client:** `counterfactual_ecg`, `clinical_what_if`, `counterfactual_validation` and
`render_figure(cf_pair_validation)`. The server now has 15 tools; see [local_mcp/README.md](../../local_mcp/README.md)
and the demo script in [local_mcp/DESIGN.md](../../local_mcp/DESIGN.md) §7.

## 7. Limitations

1. **The answer key is imperfect.** A baseline ECG from another visit includes ageing, drugs and new disease (CF-A4).
   Part of the 0.7–0.9 SD error is that, not model error.
2. **The improvement over the population mean is modest**: 6–8% of the error. Most of a patient's visit-to-visit
   ECG variation is unpredictable from these data.
3. **STEMI, especially anterior STEMI,** has larger errors and intervals that are too narrow.
4. **Inherited assumptions:** the Phase 1–9 assumptions, and the 38% negative-control bound on residual bias.
5. **PN is a model-based probability.** It rests on monotonicity and no unmeasured confounding, and cannot be
   validated for an individual.
6. **Track B** shifts are about half the average effect, and it was not validated against pairs.
7. **Hospitalized adults in one health system (MIMIC-IV).** No external validation of the individual counterfactuals.
   PTB-XL beats were extracted but not used, since CF5 was skipped.

## 8. Deviations from the roadmap (details in GATE_DECISION.md)

| Where | Change |
|---|---|
| CF2 | Legacy ECGdeli (MATLAB) instead of neurokit2. Stage 3 reduced to what Track B can use (64,688 of 240,203 ECGs). The median-beat vs ECGdeli ST correlation target of r ≥ 0.9 was not met, because of ECGdeli outliers (Spearman 0.82–0.92) |
| CF3 | A2 (shrunk) added before any test result; A1 reported as failed. Conformal intervals instead of bootstrap. MI type includes STEMI territory |
| CF4 | λ = 200 added to the grid after a smoke test. The adversary criterion was restated as conditional, after the results were seen |
| CF5 | Skipped; replaced by the local patient explorer |
| CF6 | No waveform image tool (Track B not served). `clinical_what_if` added, with no absolute probability. Claims C21–C25 added in CF7 |

## 9. Reproduce

```
python -X utf8 -m roadmap.phase10_counterfactual_ecg.cf1_01_cohort
python -X utf8 -m roadmap.phase10_counterfactual_ecg.cf2_01_extract --write-tasks          # task lists (stage 3: --write-stage3-reduced); MATLAB cf2_extract_ecgdeli per stage; --check; --merge
python -X utf8 -m roadmap.phase10_counterfactual_ecg.cf3_01_track_a
python -X utf8 -m roadmap.phase10_counterfactual_ecg.cf4_01_train                    # GPU, about 75 min
python -X utf8 -m roadmap.phase10_counterfactual_ecg.cf4_02_leakage
python -X utf8 -m roadmap.phase10_counterfactual_ecg.cf6_01_export
python -X utf8 local_mcp/build_manifest.py
python -X utf8 -m unittest discover -s roadmap/phase10_counterfactual_ecg -p "test_*.py" -t .
python -X utf8 -m unittest discover -s local_mcp/tests -t .
```
