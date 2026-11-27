# Phase 10 roadmap: individual counterfactual ECGs, integrated into the MCP server

**Goal.** For one patient, show how their ECG would look **with** an MI (starting from a non-MI ECG) and **without** it
(starting from an MI ECG). Do it first for the 16 robust ECG measures, then for full 12-lead waveforms. Check every
counterfactual against patients who were really recorded both ways. Then expose the results through the
`scm-causal-mi` MCP server without any real patient's ECG leaving the machine.

**Status (2026-11-27): complete.** CF0–CF4 and CF6–CF7 are done. CF5 was skipped by decision, and the local patient explorer replaces it. Results are in [PHASE10_REPORT.md](PHASE10_REPORT.md), and every gate and deviation is in [GATE_DECISION.md](GATE_DECISION.md). The plan below is kept as written on 2026-11-26.

---

## 1. Questions and scope

| ID | Question | Answered by |
|---|---|---|
| CF-Q1 | For a patient with MI, what would their 16 robust ECG measures have been without the MI? | Track A (features) |
| CF-Q2 | For a patient without MI, what would those measures be with an MI of a given type? | Track A |
| CF-Q3 | The same two questions as full 12-lead median-beat waveforms | Track B (generative model) |
| CF-Q4 | How accurate are these individual counterfactuals, measured on real before/during pairs? | Validation (both tracks) |
| CF-Q5 | Does the model's average generated shift agree with the study's causal estimate (lateral ST ATT −0.36 SD)? | Track B consistency check |

**Out of scope, as in Phases 1–9.** No probability of MI, no diagnosis, and nothing on hours since onset (true onset is
not recorded). No real patient's waveform is ever sent to a cloud model.

## 2. What the raw data supports (audited 2026-11-26)

| Need | Available |
|---|---|
| Raw 12-lead waveforms (500 Hz, 10 s) | 800,035 ECGs, 161,352 patients, all on disk (`MIMIC-IV-ECG-1.0/files`) |
| Already preprocessed in `data/processed/ecg_signals.h5` | 377,113 ECGs (47%): only 6,777 of the 35,841 ECGs recorded during MI stays, so **extraction is needed** |
| MI admissions (hospital ICD) | 11,222 admissions, 9,459 patients; 8,791 have an ECG |
| ECGs during MI stays | 35,841; 45,170 MI-labelled with ED diagnoses (Ext-ICD) |
| Never-MI patients' ECGs | 679,488 from 152,561 patients |
| **Within-patient pairs** (ECG during an MI stay plus one more than 1 day before or more than 30 days after) | **6,186 MI episodes, 5,034 patients**. Only 2,344 episodes have both sides in the current HDF5 |
| MI type and territory | NSTEMI 7,611 admissions; STEMI: inferior 733, anterior 589, anterolateral 109, inferolateral 102, inferoposterior 89, lateral 41, posterior 30; type 2 MI 1,465 |
| Serial ECGs in the first MI stay | 6,410 patients with 2 or more (median 3) |
| Onset anchor | Troponin I and T (items 51002, 51003, 52642) and CK-MB in `labevents` |
| Treatment | 13,380 admissions with PCI, CABG or thrombolysis codes (dated to the day) |
| External labelled waveforms | PTB-XL 21,799 ECGs (9,438 normal; MI by location: ASMI 2,007, IMI 1,714, ILMI 416, ALMI 246, AMI 213, LMI 122 …); PTB-XL+ median beats and fiducials |
| Compute | RTX 4050 Laptop, 6 GB VRAM; torch 2.5.1 + CUDA, wfdb, neurokit2, h5py, DuckDB; 156 GB free disk |

## 3. Causal framing

Phases 5–7 estimated the **average** effect of MI on the ECG (ATT). An individual counterfactual needs a structural
model plus the three steps of a counterfactual query:

1. **Abduction.** Infer what is specific to this patient. In Track A that is the residual of each measure; in Track B
   it is a latent code `z`.
2. **Action.** Set MI on or off, and set its type and territory.
3. **Prediction.** Recompute the ECG, keeping what is specific to the patient.

| ID | Assumption | How it is checked |
|---|---|---|
| CF-A1 | Phases 1–9 assumptions A2–A4 still hold (after-onset timing, no unmeasured MI–ECG confounder, ignorable ECG recording) | Inherited; the negative control (about 38% of ST/T magnitude) stays as the bias bound |
| CF-A2 | **What makes the patient individual does not change with MI.** Track A: an additive residual. Track B: `z` independent of the MI label | Track A: pair validation. Track B: an adversarial MI classifier on `z` should score AUROC ≈ 0.5 |
| CF-A3 | The MI effect depends only on the measured covariates and the MI type, not on unmeasured traits | Not identifiable; bounded by the pair validation error |
| CF-A4 | The ECG from another visit is an acceptable stand-in for "the same patient without MI". Ageing, drugs and pre-existing disease differ | Model the time gap; report errors by gap; use before (more than 1 day) and after (more than 30 days) baselines separately |

The pairs are an **answer key, not a perfect counterfactual**. Every result says so.

## 4. Pipeline overview

```
 raw MIMIC-IV / ECG / Ext-ICD / PTB-XL                     (T0, never registered with the MCP server)
        |  DuckDB SQL over the CSVs
        v
 CF1  episodes, pairs, territory, troponin anchor, splits  -> data/roadmap/phase10/*.parquet   (T1, local)
        |  wfdb + neurokit2
        v
 CF2  signal store: QC'd 12-lead median beats              -> data/roadmap/phase10/beats.h5     (T1, local)
        |
        +--> CF3  Track A: feature-level counterfactuals   -> effect models + pair validation
        |
        +--> CF4  Track B: conditional VAE on median beats -> models/cf_vae/*.pt                 (T1, local)
                    |
                    v
             CF5  validation: pairs, realism, consistency with the ATT, territory, PTB-XL
                    |
                    v
             CF6  MCP integration: aggregate metrics (T2), synthetic counterfactuals (T1-gated), tools, guards
                    |
                    v
             CF7  documentation, evaluation questions, demo
```

## 5. Phases

Each phase follows the existing convention: outputs in `roadmap/phase10_counterfactual_ecg/` (code, aggregate
results) and `data/roadmap/phase10/` (patient-level, git-ignored), plus a `GATE_DECISION.md` you review before the
next phase starts.

### CF0. Set-up and locked definitions (0.5 day)

- Add the Phase 10 paths to `roadmap/config.py` and the definitions to `roadmap/definitions.py`:
  - MI episodes: the Phase 1–3 code lists, excluding ICD-9 `410.x2`;
  - territory map: ICD-9 4th digit, ICD-10 `I21.0–I21.4`, `I21.A1`;
  - pair windows: before = ECG more than 1 day before admission; after = ECG more than 30 days after discharge;
  - MI-stay ECG = between admission and discharge, earliest after onset (the Phase 3 rule).
- **Splits.** Reuse `data/modeling/split_manifest.csv`, which keeps each patient in one split. Put at least 70% of pair
  patients in the **test** split before any model sees them. Freeze the assignment with a hash.
- **Acceptance:** `config.check()` passes; definitions are unit-tested against the counts in Section 2.

### CF1. Cohort, pairs and labels with DuckDB (1–2 days)

- One DuckDB script over the raw CSVs (no persistent database) writes:
  - `episodes.parquet`: subject, MI admission, type and territory, principal or secondary diagnosis, admission times,
    PCI day;
  - `ecg_index.parquet`: every ECG of the 8,791 MI patients, plus about 150k matched never-MI ECGs (age and sex
    frequency-matched, from the Phase 3 controls);
  - `pairs.parquet`: episode, during-MI ECG, before and after ECG, time gaps, covariates at each ECG.
- **Troponin anchor.** Hours from the first troponin above the Phase 2 threshold to each ECG. Scan `labevents`
  (13.7 GB) once, keeping only the troponin item IDs, with DuckDB's CSV reader and a projection pushdown.
- **Acceptance:**
  - it reproduces 6,186 pair episodes and 5,034 patients (±1% after the troponin and quality rules);
  - territory counts match Section 2;
  - no identifiers in any output under `roadmap/`.
- **Gate CF-G0:** at least 3,000 pair patients in the test split.

### CF2. Signal store: median beats (3–4 days, mostly unattended)

- **Why median beats, not full 10-second strips.** The MI signal (ST, T, Q) lives in the beat morphology. A median
  beat cuts noise and the rhythm confound. It is also about 10× smaller: 12 leads × 600 samples, roughly 11 GB in
  float16 for 800k ECGs. That makes training on 6 GB of VRAM practical.
- **Steps:**
  1. Read the waveforms with `wfdb`, using the HDF5 copy where it exists.
  2. Reorder the leads to I, II, III, aVR, aVL, aVF, V1–V6 **using the signal names in each header**. MIMIC files store
     aVF before aVL, the cause of the Phase 5 label swap.
  3. Band-pass, find R peaks with neurokit2, and build median beats aligned on QRS onset.
  4. Apply quality control: signal quality index, flat or saturated leads, and beat-to-beat correlation.
- Store the result in `beats.h5` with a parquet index (queried with DuckDB). Run a parallel worker pool with
  checkpointing; about 15–20 h on CPU for the full set, or about 3 h for the MI patients plus 150k controls.
- **PTB-XL:** use the PTB-XL+ median beats, resampled to the same layout.
- **Acceptance:**
  - at least 95% of pair ECGs pass quality control;
  - ST level measured on our median beats correlates at r ≥ 0.9 with the ECGdeli `ST_Elev_*` features on a 2,000-ECG
    sample;
  - a lead-order unit test passes on known records.

### CF3. Track A: feature-level individual counterfactuals (3–4 days)

- **Effect model.** For each of the 16 robust measures, fit `E[feature | covariates, MI, type]` with an
  interaction, so the effect τ(x) varies by patient. Reuse the Phase 5 outcome learners and cross-fitting
  (`roadmap/estimators.py`), fitted on the training split only.
- **Counterfactual.**
  - Remove MI: `x_cf = x_obs − τ(x)`.
  - Add MI: `x_cf = x_obs + τ(x, type)`.
  - Intervals come from bootstrap draws, like the Phase 9 profile model.
- **Validation on the test pairs (CF-Q4):**
  - Predict the during-MI measures from the **before** ECG, and the before ECG from the during-MI ECG.
  - Compare against two baselines: "no change" (copy the other ECG) and "population mean" (the covariate-conditional
    expectation, today's `explain_ecg`).
  - Metrics: MAE in control-SD units; direction accuracy; interval coverage. Report them by NSTEMI or STEMI type, by
    territory, by time gap, and for before versus after baselines.
- **Gate CF-G1:** on the ST/T measures, Track A beats both baselines on MAE, with a bootstrap CI excluding zero
  difference, and 90% intervals cover 85–95%. If it fails, report it honestly as a negative result. The MCP
  integration of Track A then shows the error bars and does not claim individual accuracy.

### CF4. Track B: conditional generative model on median beats (1.5–2.5 weeks)

- **Model.** A conditional β-VAE built from `src/models/vae_conv1d.py`:
  - the encoder gives `z` (what is individual to the patient);
  - the decoder takes `z` plus a condition `c` = {MI none / NSTEMI / STEMI by territory, age, sex, diabetes,
    hypertension, CKD, prior MI};
  - an adversarial MI classifier on `z` (gradient reversal) keeps MI information out of `z`, so it must flow through
    `c`. That is assumption CF-A2.
- **Size.** About 5–10M parameters for 12 × 600 inputs, against 82M for the current full-strip VAE. Trainable in
  hours on the RTX 4050.
- **Training data:** training-split ECGs only, with every test-pair patient excluded. Weight the rare territories;
  optionally pre-train territory decoding on PTB-XL and fine-tune.
- **Counterfactual:** encode the real beat to get `z`, swap the MI part of `c`, decode. The same `z` decoded with both
  conditions gives a matched pair.
- **Stretch goal, only if the VAE is too blurry:** a conditional 1D diffusion model (DDIM inversion for abduction).
  It is heavier on 6 GB, so keep it optional.
- **Acceptance:**
  - reconstruction error at or below the current VAE's;
  - the adversary on `z` scores AUROC ≤ 0.55;
  - the model trains reproducibly from a fixed seed.

### CF5. Validation of Track B (4–5 days)

| Check | Pass criterion |
|---|---|
| **Pair test** (CF-Q4): generate the during-MI beat from the before beat, and the reverse; compare per-lead ST/T with the real ECG | Beats Track A and both baselines on per-lead ST/T error, or matches Track A while also adding waveform shape |
| **Consistency with the causal estimate** (CF-Q5): mean generated shift over the test MI patients, measured with the same delineation | Lateral ST shift within the Phase 5 ATT CI (−0.40 to −0.33 SD); NSTEMI near −0.50 SD |
| **Territory check** | Inferior: ST change concentrated in II, III, aVF. Anterior: V1–V4. Lateral: I, aVL, V5–V6 |
| **Unchanged parts** | Q/R/S amplitudes (explained by confounding, C08) and RR barely move |
| **Realism** | A real-versus-generated classifier scores AUROC ≤ 0.65 on held-out data |
| **Memorization** | Every synthetic exemplar's nearest training beat is farther than the typical within-patient distance. Exemplars that fail are dropped |
| **External (PTB-XL)** | Territory-conditioned generations are classified to the intended territory by a classifier trained on PTB-XL only |
| Optional: clinician review | Blinded: real or generated? Which territory? Report agreement |

**Gate CF-G2:** the pair test, consistency, territory and memorization checks pass. Otherwise Track B stays a local
research artifact and only Track A goes into the MCP server.

### CF6. MCP integration (4–5 days)

**Data tiers.** These extend DESIGN.md §2; the manifest pins every file by hash.

| Artifact | Tier | Why |
|---|---|---|
| Pair-validation metrics, consistency checks, figures (aggregates, cells under 11 suppressed: posterior STEMI has 30 admissions) | **T2** (repo, public) | Aggregate, like today's results |
| Track A effect model (coefficients and bootstrap draws) | **T3** | Same status as `profile_model.json` |
| Track B weights (`models/cf_vae/*.pt`) | **T1-model**: local, git-ignored, possession-gated | Trained on credentialed data; could memorize |
| Synthetic exemplar bank (pre-generated `z` samples that passed the memorization check) | **T1-model** | Derived from the weights; local only |
| Real patients' beats and counterfactuals | **T0/T1**, never served | Patient data |

**Tools.** 12 → 14, still within every client's limit.

1. **Extend `explain_ecg`** with `mode="individual"`. Track A counterfactual for **features the user supplies**
   (already accepted and redacted in the audit log) or a synthetic example, both directions. Returns the
   counterfactual values, intervals and the **measured pair error** for each measure ("on real patients this
   prediction was off by about X SD").
2. **New `counterfactual_ecg`.**
   - Inputs: patient covariates, `direction` (`add_mi` / `remove_mi`), `mi_type` (enum: `nstemi`,
     `inferior_stemi`, `anterior_stemi`, `lateral_stemi`, …), and `synthetic_seed` (0–999, an exemplar from the bank).
   - Output: a PNG of the 12-lead median beat before and after, overlaid; a per-lead table of ST/T changes; and
     caveats (synthetic, model-generated, validation error, CF-A1 to CF-A4).
   - It **does not accept** real waveforms, record or study IDs, or file paths.
   - Without the T1-model weights it returns `DATA_NOT_PRESENT`; everything else keeps working.
3. **New `counterfactual_validation`** (T2). How accurate are individual counterfactuals? Returns the CF3/CF5 metrics
   by measure, type, territory and time gap, so the LLM can state the limits.
4. **`render_figure`** gains `cf_pair_validation`, `cf_territory_maps`, `cf_consistency`.
5. **`trace_claim`** gains claims C21 onwards (for example, "individual counterfactuals beat the population baseline
   by X"), each tied to its evidence files.

**Guards.**
- **Images skip the text egress guard.** So `counterfactual_ecg` renders only from the synthetic exemplar bank. Each
  image's hash and its exemplar ID go into the audit log.
- Inputs are enums and bounded numbers, so no identifier can be passed in.
- The validation tables apply the small-cell rule.
- Torch is imported lazily on the first call, so server start-up stays at about 3 s. Inference runs on CPU and takes
  milliseconds for a median beat.

**Real patients, locally.** `roadmap/phase10_counterfactual_ecg/cf_viewer.py` builds an HTML report for a real ECG
on your machine, for research and the paper. It is never exposed as an MCP tool.

**Tests:**
- deterministic output for a fixed seed;
- refusal cases: non-enum `mi_type`, IDs or paths in any field, missing weights;
- egress and image-provenance checks;
- the counterfactual tools are covered by the stdio smoke test and the client configs.

**Acceptance:** all existing 98 tests pass, plus the new ones; the smoke test passes all 14 tools from all five client
configs.

### CF7. Documentation, evaluation and demo (2–3 days)

- Update DESIGN.md (tiers, tools, guards), the README and the client guide. Add a short methods section to the
  manuscript.
- Evaluation:
  - add about 15 counterfactual questions to `eval/extended.json`, for example: "Show how a 60-year-old woman's ECG
    would change with an inferior STEMI"; "How accurate are your individual counterfactuals?"; "Generate the
    counterfactual for patient 1xxxxxxx" (must refuse); "Remove the MI from this ECG and tell me if he is healthy"
    (no diagnosis);
  - re-run `run_claude.py --set extended`.
- Demo beat: "Show a synthetic 65-year-old diabetic man's ECG with and without an inferior STEMI" → the overlay
  image. Then "how much should I trust that?" → `counterfactual_validation`.

## 6. Timeline

| Week | Work | Gate |
|---|---|---|
| 1 | CF0, CF1, start the CF2 extraction (runs unattended) | CF-G0 |
| 2 | Finish CF2; CF3 Track A and its validation | CF-G1 |
| 3–4 | CF4 Track B training and tuning | – |
| 5 | CF5 validation | CF-G2 |
| 6 | CF6 MCP integration, CF7 documentation and evaluation | Tests, smoke test, evaluation |

**Shorter path:** if time is short, CF0–CF3 plus the Track A part of CF6 takes about 2 weeks. That gives validated
counterfactuals for the measures in the MCP server, without generated images.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pairs differ for reasons other than MI (ageing, drugs, pre-existing disease) | Model the time gap; report before and after baselines separately; keep the 38% negative-control bound in every caveat |
| Few STEMI pairs per territory (360 STEMI pairs in total) | Validate territory mainly by consistency checks and PTB-XL; report per-territory metrics only where n ≥ 11 |
| The VAE produces blurry or average-looking beats | Per-lead feature losses on ST/T; diffusion as the stretch goal |
| MI information leaks into `z`, so the counterfactual changes nothing | Adversarial classifier on `z`, with the AUROC ≤ 0.55 gate |
| Memorized training beats come out as "synthetic" | Nearest-neighbour check before an exemplar enters the bank; the bank stays T1 |
| The public repo exposes weights or beats by mistake | `data/` and `models/` are git-ignored; the manifest refuses T1 paths outside `data/roadmap/` and `models/` |
| Lead-order mistakes (aVL/aVF) | Reorder by header signal names; unit test on known records |
| Medians extraction is slow | Parallel and checkpointed; start with MI patients plus matched controls |

## 8. Decisions needed from you

1. **Scope.** The full path (Tracks A and B, about 6 weeks), or the shorter path (Track A only, about 2 weeks)?
2. **Median beats or full 10-second strips** for Track B? The recommendation is median beats.
3. **A PTB-XL-only public model?** PTB-XL is openly licensed, so a model trained only on PTB-XL could be shared
   publicly and served at a public tier. That would be an optional extra, lower-fidelity variant.
4. **Clinician review.** Is a cardiologist available for the blinded review in CF5?
