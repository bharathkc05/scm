# Phase 10, CF0–CF1: gate decision

**Date:** 2026-11-26 · **Gate CF-G0: PASS** · **Accepted by the user 2026-11-26** (all three proposals). 4,128 pair patients are held out in `pair_test`; the minimum was 3,000.

The plan is in [ROADMAP.md](ROADMAP.md). The numbers come from
[cf1_01_cohort.json](cf1_01_cohort.json), and the checks are in
[test_cf0_cf1.py](test_cf0_cf1.py) (10 tests, all passing).

## CF0. Set-up and locked definitions

- **Paths.** `roadmap/config.py` gains the raw waveform folder, the Ext-ICD labels, PTB-XL, PTB-XL+ and
  `PHASE_DIRS[10]`. `python -m roadmap.config` shows every path present.
- **Definitions.** `roadmap/definitions.py` gains a Phase 10 section:
  - MI kind priority: STEMI > NSTEMI > type 2 > other;
  - territory maps for ICD-9 410.x and ICD-10 I21/I22 (detail, plus a coarse class used for modelling). ICD-9
    410.x2 follow-ups stay excluded, which the tests check against the Phase 1 `acute_mi` flag;
  - pair windows: before = more than 24 h before ED arrival; after = more than 30 days after discharge and more than
    1 day before any later MI;
  - troponin anchor: first troponin above the lab's upper reference limit, from 12 h before arrival to discharge;
  - revascularisation codes;
  - the Phase 10 split rule.

## CF1. Cohort, pairs and labels

The script runs in one DuckDB pass (about 12 s). It reuses the Phase 2 admissions file and the read-only MIMIC DuckDB
copy, and writes patient-level files to `data/roadmap/phase10/` (git-ignored).

| Output | Rows |
|---|---|
| `episodes.parquet`: acute-MI admissions | 11,222 (9,459 patients) |
| `ecg_index.parquet`: ECGs of MI patients plus one per never-MI patient | 271,341 |
| `pairs.parquet`: MI episodes with a during-MI ECG and at least one baseline ECG | **6,205 episodes, 5,159 patients** |
| `cf_splits.parquet`: Phase 10 split per patient | 180,733 |

**Episodes.**
- **By kind:** NSTEMI 7,581, STEMI 1,917, type 2 1,443, other or unspecified 281.
- **STEMI by territory:** inferior 915, anterior 698, unspecified 147, other site 77, lateral 51, posterior 29.
- **Timing:** 6,481 are in the after-onset context (principal diagnosis, unplanned).
- **Troponin:** tested in 92.8% of episodes; a rise was found in 89.4%, a median 1.1 h after arrival (IQR 0.4–4.9).
- **Treatment:** 43.4% were revascularised.

**Pairs.**
- **Baselines:** 4,506 have a before-MI baseline and 4,055 an after-MI baseline; 2,356 have both.
- **By kind:** NSTEMI 4,499, STEMI 1,017 (inferior 482, anterior 375, lateral 25, posterior 21), type 2 525.
- **Gaps:** before baselines a median 122 days earlier (IQR 31–455); after baselines a median 98 days later
  (IQR 55–232).
- **Subsets for the primary analysis:** 3,563 are in the after-onset context; 4,984 are the patient's first MI.

**Split.** Pair patients are held out at 80/20: 4,128 in `pair_test` and 1,031 in `pair_val`, stratified by MI kind.
Everyone else keeps their Phase 3 split (train 122,942 / validation 26,349 / test 26,283). The split is frozen by
SHA-256 `6f0fc44f…`.

## Deviations from the plan and things found

1. **The "during" window starts at ED arrival, not hospital admission.** This counts 44,810 during-MI ECGs, against
   35,841 in the audit, which used admission time. ED ECGs taken during the MI presentation belong to the MI.
2. **Pair counts differ slightly from the audit:** 6,205 episodes against 6,186 (+0.3%), and 5,159 patients against
   5,034 (+2.5%). The causes are the ED-arrival window and the stricter baseline rules: baselines are never inside
   another MI stay, and after baselines must come before the next MI. Patients are above the planned ±1% band
   because of the wider window. The definitions are locked and tested.
3. **5 ECGs fell into two MI admissions that share one ED arrival** (discharge and readmission in the same visit).
   Each is now assigned to the earlier stay.
4. **824 before baselines come after an earlier MI.** They are flagged (`before_follows_earlier_mi`) so CF3 can keep
   them out of the primary pair test, because those ECGs may already carry old-MI changes.
5. **3,571 pair patients were in the Phase 3 training split.** Models from Phases 5–6 saw them, so Phase 10 refits its
   own models on the Phase 10 `train` split. The Phase 5 nuisance models must not be reused. This is recorded in
   `definitions.py`.
6. **The preprocessed HDF5 file covers only half of what is needed:** 7,477 of the 14,332 pair ECGs, and 101,812 of
   the 271,341 indexed ECGs. CF2 has to extract about 169,500 ECGs from the raw waveforms. That is unattended work,
   about 15–20 h on CPU, with checkpoints.
7. **False positive in the leak scanner.** The fixed random seed (20261121) has the shape of a MIMIC identifier. The
   committed JSON contains no patient identifiers.

## Decisions for you before CF2

1. **Accept CF0–CF1** as above?
2. **Primary pair test set:** I propose after-onset-context episodes whose before baseline does not follow an earlier
   MI. The other pairs become sensitivity analyses.
3. **Scope** (ROADMAP.md §8): the full path with Tracks A and B, or Track A only? The answer changes CF2. Track A
   needs median beats only for the 14,332 pair ECGs and their training counterparts; Track B needs all 271k.

## Decisions recorded (2026-11-26)

1. **CF0–CF1 accepted.**
2. **Primary pair test set:** after-onset-context episodes whose before baseline does not follow an earlier MI. All
   other pairs are sensitivity analyses.
3. **Scope.** CF2 serves both tracks. It extracts in priority order, so Track A can start before the full set needed
   by Track B is done:
   1. the pair ECGs;
   2. the remaining during-MI ECGs;
   3. other ECGs of MI patients;
   4. the control ECGs.

---

# Phase 10, CF2: extraction (all three stages done)

**Gate: PASS. Accepted by the user 2026-11-27.** Extraction stops here; the remaining 175,515 ECGs are not needed (see the stage 3 notes).

Against the roadmap targets:
- **Pair ECGs passing QC:** 95.8% (target 95%). Met.
- **Lead-order check:** met in 100% of ECGs.
- **Median-beat ST vs ECGdeli ST at r ≥ 0.9:** not met as written. Spearman is 0.82–0.92 and Pearson with |ST| < 2 mV is 0.69–0.89, with a median difference of 0.01 mV. The two measure ST differently: ECGdeli averages it beat by beat at each beat's own J point, while the median beat uses one median J. ECGdeli also has a few impossible values. Accepted as adequate.

**Date:** 2026-11-26. The method is the legacy ECGdeli chain, run in MATLAB by `cf2_extract_ecgdeli.m` with two changes:
leads are reordered by the names in the `.hea` header, and a median beat is added. Checks are in
[cf2_00_pilot.json](cf2_00_pilot.json) (300 ECGs) and [cf2_02_check_stage1.json](cf2_02_check_stage1.json) , [cf2_02_check_stage2.json](cf2_02_check_stage2.json) and [cf2_02_check_stage3.json](cf2_02_check_stage3.json).

| Stage | ECGs | Status |
|---|---|---|
| 1. Pair ECGs | 14,332 | **done** in 98 min (2.4 ECG/s, 6 workers); merged into `beats.h5` |
| 2. Other during-MI ECGs | 38,605 | **done** in 278 min (2.3 ECG/s); merged with stage 1 (52,937 ECGs, 476 MB) |
| 3. Reduced (see below): non-pair MI patients' other ECGs, sampled controls, PTB-XL | 64,688 | **done** in 392 min (2.7 ECG/s); all stages merged: 117,625 ECGs, 1.06 GB |

**Stage 1 results.**
- **Status:** 14,029 ok; 281 had too few clean beats for a median beat; 22 errors inside ECGdeli (0.15%).
- **Quality control:** 95.8% pass (at least 3 beats, template correlation ≥ 0.80, heart rate 30–180, no sample above 5 mV).
- **Lead order:** correct in 100% of ECGs (aVF ≈ (II+III)/2 at r = 0.999; the swapped-order check gives −0.50).
- **Same as the legacy features:** ST_Elev r = 0.993–0.999 and T_Amp r = 0.994–0.996 in 8,516 ECGs present in both.
- **Median-beat ST vs ECGdeli ST:** Spearman 0.82–0.89, median difference 0.01 mV. Pearson is low in some leads
  because 1–15 ECGs per lead have impossible ECGdeli beat-wise ST values, up to 21 mV. The median beats stay below
  4 mV. These outliers are in the legacy feature file too, so **CF3 sets |ST_Elev| > 2 mV to missing**.

**Pairs after QC.**
- **During-MI ECG passes:** 5,858 of 6,205 episodes; 5,751 also have a passing baseline (4,118 before, 3,763 after).
- **Primary test set** (after-onset context, before baseline not after an earlier MI, both ECGs pass): 1,736 episodes,
  1,315 `pair_test` and 340 `pair_val` patients. By kind: NSTEMI 1,374, STEMI 302.

**Stage 2 results.**
- **Status:** 37,353 ok; 1,105 had no median beat; 147 errors inside ECGdeli (0.4%).
- **Quality control:** 93.8% pass. Across stages 1 and 2 it is 94.4%.
- **Lead order:** correct in 100% of ECGs (r = 0.999).
- **Same as the legacy features** (37,345 ECGs):
  - r = 0.98–0.997 in every lead except V5 (0.78) and V6 (0.82).
  - The V5 and V6 gap is outliers. Over both stages, 98.2% of values are identical to the legacy file, and with
    |ST| < 2 mV, r = 0.99 in V4, V5 and V6.
- **Median-beat ST vs ECGdeli ST:** Spearman 0.85–0.92, median difference 0.01 mV. Pearson with |ST| < 2 mV is
  0.75–0.89. 1–38 ECGs per lead have impossible ECGdeli ST values (up to 21 mV), which CF3 sets to missing.

**Stage 3 reduced to what Track B can use** (`--write-stage3-reduced`, 64,688 of 240,203 ECGs). Skipped:
- the other ECGs of pair patients (59,721): these patients are held out, and their pairs are the test;
- controls with no Phase 3 split (31,890);
- controls beyond a random sample of 25,000 train, 5,000 validation and 5,000 test (83,904 more).

**Stage 3 results.**
- **Status:** 63,658 ok; 961 had no median beat; 69 errors.
- **Quality control:** 96.6% pass in MIMIC and 97.0% in PTB-XL.
- **Lead order:** correct in 100% of ECGs.
- **Same as the legacy features:** r = 0.97–1.00 in 19,012 ECGs.

**Training material for Track B** (QC-passing median beats in the Phase 10 `train` split): 9,113 during-MI ECGs,
5,261 other ECGs of MI patients and 24,256 controls. The validation and test splits each have about 4,800 controls,
2,000 during-MI ECGs and 1,100 others. There are also 21,144 QC-passing PTB-XL ECGs for the external test.

---

# Phase 10, CF3: Track A, feature-level individual counterfactuals

**Date:** 2026-11-27 · **Gate CF-G1: PASS for A2 (shrunk), FAIL for A1 (additive, pre-specified)** ·
**Accepted by the user 2026-11-27.**

Code: [cf3_01_track_a.py](cf3_01_track_a.py) (about 40 s). Results: [cf3_01_track_a.json](cf3_01_track_a.json),
[CF3_PAIR_VALIDATION.csv](CF3_PAIR_VALIDATION.csv) (per measure) and [CF3_SUBGROUPS.csv](CF3_SUBGROUPS.csv).
Tests: [test_cf3.py](test_cf3.py) (8 tests, all passing).

## Model

- **Measures:** the 16 robust measures (Phases 6–7), taken from the CF2 ECGdeli features with correct lead labels.
  |ST_Elev| > 2 mV is set to missing, then the values are winsorized with the Phase 3 bounds.
- **Structural model:** the Phase 5 design (5-df age spline, sex, diabetes, hypertension, CKD, prior MI, with
  interactions), fitted on the Phase 10 `train` split only:
  - `m0` (no MI): OLS on 23,152 training controls, one ECG per never-MI patient;
  - `m1` (MI): OLS on 2,092 training MI episodes (the earliest QC-passing during-MI ECG of each), with MI-type
    dummies. By type: NSTEMI 1,464, STEMI inferior 218, STEMI anterior 142, type 2 137, other STEMI 73, other 58.
- **Counterfactual** (abduction, action, prediction): the patient's own deviation is `u = y_obs − m_obs(L)`, and
  `y_cf = m_cf(L) + ρ·u`.
  - **A1, additive** (pre-specified in the roadmap): ρ = 1, which is `y_cf = y_obs ∓ τ(L, type)`.
  - **A2, shrunk** (added at CF3 design, before any test-set result): ρ per measure and direction, fitted by least
    squares on `pair_val` only.
- **Baselines:** "no change" (copy the other ECG) and "population mean" (`m_cf(L)`, what `explain_ecg` gives today).
- **Intervals:** split conformal at 90%, calibrated on `pair_val` residuals separately for before and after
  baselines.

## Results on the primary test set (1,378 pairs, 1,313 patients)

Mean absolute error over the 14 ST/T measures, in control-SD units:

| Direction | A1 additive | **A2 shrunk** | No change | Population mean | 90% coverage (A1 / A2) |
|---|---|---|---|---|---|
| Add MI (before ECG → during-MI ECG) | 1.009 | **0.937** | 1.012 | 1.016 | 0.897 / 0.889 |
| Remove MI (during-MI ECG → before ECG) | 1.009 | **0.682** | 1.012 | 0.725 | 0.897 / 0.916 |

- **A2 vs the baselines** (95% patient-bootstrap CIs):
  - Add MI: −0.075 [−0.093, −0.056] vs no change; −0.079 [−0.090, −0.068] vs population mean.
  - Remove MI: −0.331 vs no change; −0.044 [−0.055, −0.033] vs population mean.
  - Per measure, A2 beats both baselines on 14 of 14 ST/T measures when adding MI. When removing MI it beats no
    change on 14 of 14 and the population mean on 12 of 14. It is never worse.
- **A1 is no better than either baseline when adding MI**: −0.003 [−0.016, 0.010] vs no change. When removing MI it
  is clearly worse than the population mean (+0.28).
- **Sensitivity sets** (all before baselines, 3,299 pairs; all after baselines, 3,000 pairs): the same pattern. A2
  beats both baselines in both directions.
- **Interval width:** the median 90% half-width is 1.9 SD (add MI) and 1.6 SD (remove MI).

## What this means

1. **Only 30–60% of a patient's own deviation persists between visits** (ρ on `pair_val`: 0.31–0.76 for adding MI,
   0.20–0.60 for removing it). Two ECGs months apart differ by about 1 SD on each ST/T measure even without MI.
   Copying the whole deviation (A1) carries that noise into the counterfactual. Shrinking it (A2) is the
   regression-to-the-mean correction.
2. **Individual counterfactuals are possible, but imprecise.** The best model cuts the error by 7–8% when adding MI
   and 6% (vs the population mean) when removing it. The remaining error, about 0.9 SD, is mostly visit-to-visit
   variation that no feature-level model can predict. The 90% intervals are honest (coverage 0.89–0.92), but wide.
3. **Removing MI is easier than adding it** (0.68 vs 0.94 SD). Baseline ECGs vary less, and MI changes differ a lot
   between patients in ways the covariates and MI type do not capture.
4. **Consistency with Phases 5–7:** Track A's average effect over the primary MI episodes has the same sign as the
   Phase 5 ATT on 16 of 16 measures, at similar size (for example, lateral ST −0.43 vs −0.36 SD). Model uncertainty
   in τ for one patient is about 0.2 SD.

**Subgroups** (primary set, A2):
- **NSTEMI** (1,084 pairs): beats both baselines in both directions.
- **STEMI** (243 pairs): adding MI, 1.13 vs 1.26 (no change) and 1.20 (population mean). Removing MI, it ties the
  population mean (0.66 vs 0.64).
- **Anterior STEMI** (87 pairs): the largest errors (1.38 SD when adding MI) and **under-coverage (77%)**. The
  intervals are too narrow there.
- **Time gap** (quartiles from 1–55 days to 2–10 years): the pattern is stable. A2 helps most for short gaps.

## Deviations from the roadmap

1. **A2 was added; it was not in the roadmap.** It was written before any test-set evaluation, its only tuned
   parameters (ρ) come from `pair_val`, and `pair_test` was used once. The pre-specified A1 fails the gate, and that
   is reported as such.
2. **Intervals are split conformal on `pair_val`, not bootstrap.** A bootstrap refit only measures model uncertainty
   in τ (about 0.2 SD). The real error is dominated by visit-to-visit variation (about 1 SD), so bootstrap intervals
   would cover far below 90%. The bootstrap is still run and reported as model uncertainty.
3. **MI type includes the STEMI territory** (anterior, inferior, other), because ST changes are territory-specific.

## Decisions for you before CF4

1. **Accept CF3**, with Gate CF-G1 passed by A2 and failed by A1, as above?
2. **Track A model for CF6 (MCP):** I propose **A2** with its conformal intervals. The tool would always report the
   interval and the validation error, and would say that about 0.9 SD of error remains, so it is not a precise
   individual prediction.
3. **Anterior STEMI under-coverage:** I propose a warning in CF6 for STEMI, and especially anterior STEMI, that the
   intervals are too narrow there. `pair_val` has only 98 STEMI pairs with a before baseline (38 anterior), too few for type-specific calibration.
   Track B (CF4–CF5) is the place to try to do better on STEMI morphology.

## Decisions recorded (2026-11-27)

1. **CF3 accepted.** Gate CF-G1 is passed by A2 and failed by the pre-specified A1, and both results are reported.
2. **Track A model for CF6:** A2 with its split-conformal 90% intervals. The tool always reports the interval and
   the validation error, and does not present the result as a precise individual prediction.
3. **STEMI warning:** CF6 warns that intervals are too narrow for STEMI, especially anterior STEMI.

---

# Phase 10, CF4: Track B, conditional β-VAE on median beats

**Date:** 2026-11-27 · **CF4 acceptance: 2 of 3 criteria pass as written; the adversary criterion fails (AUROC 0.59
against ≤ 0.55)** ·
**Accepted by the user 2026-11-27 with option (a).**

Code: [cf4_model.py](cf4_model.py) (data and model), [cf4_01_train.py](cf4_01_train.py) (grid, selection, checks)
and [cf4_02_leakage.py](cf4_02_leakage.py) (where the MI information in z comes from). Results:
[CF4_SWEEP.csv](CF4_SWEEP.csv), [cf4_01_train.json](cf4_01_train.json) and [cf4_02_leakage.json](cf4_02_leakage.json).
Tests: [test_cf4.py](test_cf4.py) (10 tests, all passing). Checkpoints are in `models/cf_vae/` (git-ignored).

## Model and data

- **Input:** the CF2 median beat, 12 leads × 400 samples (800 ms, QRS onset at 200 ms), in mV divided by the training
  SD (0.178 mV).
- **Condition c** (given to the encoder and to every decoder layer through FiLM):
  - MI state: none, NSTEMI, STEMI anterior, STEMI inferior, other STEMI, type 2, other;
  - age, sex, diabetes, hypertension, CKD;
  - prior MI (a prior-MI code, or any MI episode discharged before the ECG);
  - heart rate.
- **Latent z:** 32 dimensions. A gradient-reversal adversary predicts the MI state from z.
- **Size:** 7.6M parameters, about 6 s per epoch on the RTX 4050.
- **Data:** 38,630 training and 8,082 validation median beats; no pair patient, and the test split is untouched.
  Training beats by MI state: none 29,517, NSTEMI 6,140, STEMI inferior 982, STEMI anterior 765, type 2 675, other
  STEMI 330, other 221. Sampling is weighted by 1/√count of the MI state.
- **Grid:** β ∈ {0.5, 1, 2} × λ ∈ {10, 50, 200}, seed fixed, early stopping on validation loss. λ = 200 was added
  after a 3-epoch smoke test, before the full runs.
- **Selection rule (fixed in advance):** among runs with adversary AUROC ≤ 0.55, the lowest validation reconstruction
  error. No run passed, so the rule fell back to the lowest AUROC: **β = 1, λ = 10**.

## Results against the roadmap criteria

| Criterion | Result | |
|---|---|---|
| Reconstruction at or below the current VAE's | Per-lead FVU **0.068** vs **0.278** (global FVU 0.032) | **Pass** |
| Adversary on z: AUROC ≤ 0.55 | **0.591** (boosting probe; logistic 0.562). Best in the grid: 0.590–0.600 | **Fail** |
| Reproducible from a fixed seed | Two runs: weights identical (max difference 0.0) | **Pass** |

Raising the adversary weight made leakage **worse**, not better (λ = 200: AUROC 0.66–0.72). Gradient-reversal
training becomes unstable at high weights: the in-training adversary is fooled, but a fresh probe still finds the
information.

## Where the leaked information comes from

The roadmap criterion tests z **alone**. Assumption CF-A2 needs z to carry no MI information **beyond the covariates
in c**, because the decoder already receives those covariates.

| Probe predicting MI (validation) | AUROC |
|---|---|
| Covariates only (age, sex, comorbidities, heart rate) | 0.830 |
| z only | 0.591 |
| Covariates + z | 0.841 |
| **What z adds to the covariates** | **+0.011 (95% CI −0.004 to +0.024)** |

The z-only probe score correlates most with age (r = 0.15). z keeps some age-related beat shape, and age predicts MI
strongly, so z alone looks informative. Given the covariates, z adds nothing statistically detectable. **The leak is
mostly covariate information duplicated in z, not MI-specific morphology.**

## Early look at the counterfactuals (validation MI beats, MI state set to "none")

ST change at J + 60 ms, in mV:
- **Anterior STEMI:** ST falls in V2–V4 (−0.065, −0.090, −0.074), with small reciprocal rises in III and aVR.
- **Inferior STEMI:** ST falls in II, III and aVF (−0.038, −0.059, −0.049), with reciprocal rises in I, aVL and V2.
- **NSTEMI:** ST rises slightly in I, II and V2–V6 (+0.01 to +0.02); the ST depression is removed.

These are the right leads and the right directions for each territory. **The size may be too small:** the mean
lateral ST shift is +0.0075 mV, about half the Phase 5 ATT (0.0156 mV). CF5 will test this formally, measured the same
way as the ATT.

## Decisions for you before CF5

1. **Adversary criterion.** Two options:
   - **(a) Recommended:** accept CF4, with the adversary criterion restated as conditional, which is what CF-A2
     requires: **"z adds no MI information beyond the covariates: the CI of ΔAUROC includes 0, and its upper bound
     is ≤ 0.03."** The selected model passes this (+0.011, CI −0.004 to +0.024). The marginal result (0.59) stays
     reported as a fail of the criterion as first written. This change is made after seeing the results, and it is
     labelled as such.
   - **(b)** Keep the original criterion and try to remove the age information from z too, for example with a
     second adversary on the covariates. That makes the model blurrier, and age-related shape is not what CF-A2
     protects. I expect it to cost reconstruction quality for no causal gain.
2. **The model for CF5** is `cvae_b1.0_l10`: the best reconstruction among the least-leaking runs.

## Decisions recorded (2026-11-27)

1. **CF4 accepted with option (a).** The adversary criterion is restated as conditional: z adds no MI information
   beyond the covariates (the ΔAUROC CI includes 0, and its upper bound is ≤ 0.03). The selected model passes this
   (+0.011, CI −0.004 to +0.024). The criterion as first written (z alone, AUROC ≤ 0.55) is reported as failed
   (0.591). The restatement was made after seeing the results.
2. **Model for CF5:** `cvae_b1.0_l10`.

---

# Phase 10: patient explorer (replaces CF5, decided 2026-11-27)

The user does not need waveform reconstruction validated as a product, so **CF5 is skipped**. Track B stays a local
research artifact. In its place, [patient_explorer.py](patient_explorer.py) writes a **local-only** HTML report for a
random or chosen real patient:
- the timeline of ECGs and MI admissions;
- Track A (A2) counterfactual measures with 90% intervals;
- Track B median-beat overlays (observed beat plus the decoder's change);
- for pair patients, the comparison with the patient's real other ECG;
- clinical do() risks and the probability of necessity.

Random selection draws from held-out patients. Reports go to the git-ignored `data/roadmap/phase10/explorer/` and
must never pass through the MCP server or a cloud service. Tests: [test_patient_explorer.py](test_patient_explorer.py)
(8 tests). They include an exact match with the CF3 validation numbers, and they never print patient values.

---

# Phase 10, CF6: MCP integration

**Date:** 2026-11-27 · **Acceptance: PASS** (109 MCP tests; smoke test 15 of 15 tools from all five client configs) ·
awaiting your review.

## What the server gained (12 → 15 tools)

| Tool | What it answers | Inputs | Never returns |
|---|---|---|---|
| `counterfactual_ecg` | One ECG's 16 robust measures with MI **removed** (ECG recorded during MI) or **added** (choose NSTEMI, anterior / inferior / other STEMI, type 2, other). Track A model A2, 90% interval, and for each measure the error this model made on real held-out pairs. | Patient facts; per-lead ECGdeli features (12 ST levels, 5 T amplitudes, 2 Q amplitudes) **or** a synthetic profile ECG; which pairs calibrate the interval (before-MI or after-MI baselines) | Diagnosis, probability of MI, waveforms |
| `clinical_what_if` | For one profile: risk ratio of do(diabetes / hypertension / CKD = yes) vs no, with a 95% interval, beside the study average. For an MI patient with the condition, the **probability of necessity**. Diabetes returns both adjustment sets as a bracket, as in Phase 5. | Patient facts, condition, whether the patient had an MI | **Absolute probabilities of MI** (server rule; they are computed internally for the ratio only) |
| `counterfactual_validation` | How accurate individual counterfactuals are: composite and per-measure errors against both baselines, coverage, subgroups by MI type and time gap. | View, direction, pair set | Cells under 11 pairs |

`render_figure` gains `cf_pair_validation`.

## How it is built

- **Export** ([cf6_01_export.py](cf6_01_export.py)) writes three T3 assets with aggregate parameters only:
  - `cf_track_a.json`: the design, coefficients, ρ, conformal half-widths, input rules and validation metrics;
  - `cf_clinical.json`: the Layer 1 do() models fitted on the Phase 3 training split, with 200 draws per arm from
    the patient-clustered sandwich covariance;
  - the figure.

  The CF3 validation CSVs become two new DuckDB views (T2). The manifest grows from 69 to 78 artifacts.
- **Portable arithmetic** (`local_mcp/tools/cf_math.py`) needs no numpy or sklearn. It matches the fitted Track A
  model and the patient explorer's Layer 1 models **to 1e-9** on random profiles and every MI type.
- **Guards:**
  - Inputs are enums and bounded numbers. Unknown feature names are rejected, so a planted `subject_id` is refused,
    and ST beyond 2 mV is rejected as an artefact.
  - Patient facts and features are redacted in the audit log.
  - All outputs pass the egress guard. The by-measure validation view nests methods inside each measure to stay
    under the 60-item list cap and the 32 KB payload cap.
- **Wording written by the server:**
  - every counterfactual comes with its interval, the validation error and the STEMI warning (accepted at CF3);
  - the clinical tool states that it gives no absolute probability, and lists the assumptions behind the
    probability of necessity.

## Deviations from the roadmap

1. **No waveform image tool.** CF5 was skipped, so by the roadmap's own rule (gate CF-G2) Track B stays a local
   research artifact. The roadmap's `counterfactual_ecg` image output and the synthetic exemplar bank are dropped.
   `counterfactual_ecg` is the Track A feature-level tool instead, and `explain_ecg` is unchanged.
2. **`clinical_what_if` is new** (agreed 2026-11-27, when you asked about individual reasoning). To keep the
   server's standing rule (no individual probability of MI), it returns ratios and the probability of necessity
   only. The local patient explorer still shows absolute risks.
3. **`trace_claim` claims C21 onwards are not added yet.** They belong with the paper text, in CF7.
4. **Eval sets:** G01 and F13 now expect 78 verified artifacts, and three golden questions (G16–G18) cover the new
   tools. Their values were checked against the live server. They have **not** been run in Claude Code, because
   that run is paid; that is part of CF7.

## Decisions for you

1. **Accept CF6?**
2. **CF7** (documentation, evaluation and demo): run the golden set in Claude Code (about $2) including G16–G18,
   add the Phase 10 claims, and write the Phase 10 report?
