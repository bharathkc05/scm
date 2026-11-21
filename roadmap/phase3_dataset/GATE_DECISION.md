# Phase 3 — Gate Decision

**Gate criterion (roadmap):** counts match the audit.

**Result: PASS** (pending reviewer sign-off). Layer 1 (430,852), Layer 2 (193,470) and after-onset MI
(5,643) equal the Phase 2 counts; one row per admission; patients never split; features and latents
complete. Evidence: `DATASET_REPORT.md`, `p3_03_build.json`, `cohort_flow.csv`.

## Decisions for review

| # | Decision | Reason |
|---|---|---|
| P1 | ICD-9 410.x2 follow-up admissions (805) are excluded from Layer 2 controls; in Layer 1 they stay as non-MI | A follow-up admission within weeks of an MI shows that MI on the ECG; as a control it would bias contrasts toward zero |
| P2 | Admissions without any diagnosis code (77) are never controls | MI status unknown |
| P3 | 81,477 patients outside the legacy manifest assigned 70/15/15 by seed, stratified by ever-MI; legacy splits unchanged | Keeps the roadmap's existing patient-level splits and covers all patients |
| P4 | Primary Layer 2 outcomes = winsorized features (training-split 0.5/99.5 percentiles); raw features as sensitivity | Limits the influence of delineation outliers; bounds never see validation/test |
| P5 | Negative-control set excludes patients with any prior MI | So a "later MI" is a first MI and the earlier ECG cannot show an old infarct |
| P6 | Drop the planned high-quality-ECG sensitivity | 99.96% of ECGs pass the beat-detection check, so it would repeat the primary analysis |
