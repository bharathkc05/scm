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
