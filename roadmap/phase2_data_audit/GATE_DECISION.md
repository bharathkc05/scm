# Phase 2 — Gate 1 Decision: can the graph be tested?

**Roadmap criteria:** history precedes admission, ECG follows MI onset, key variables are complete.

**Result: PASS, with one condition** (pending reviewer sign-off). Evidence: `DATA_AUDIT_REPORT.md`.

| Criterion | Evidence | Status |
|---|---|---|
| History precedes admission | 64–78% of primary chronic flags backed by an earlier admission; strict prior-only version built; first-admission bias identified and handled by the look-back sensitivity | ✅ |
| ECG follows MI onset | Admission-context rule gives 5,643 after-onset MI admissions with a feature ECG (median 0.83 h after arrival); uncertain and before classes separated | ✅ |
| Key variables complete | Age, sex, flags and all 95 features complete for 193,470 candidate admissions; label acute-only across ICD-9/10 | ✅ |
| Label not read from the ECG | Not verifiable from coded data | ⚠️ Condition: NSTEMI-only Layer 2 (n = 4,176) must agree in direction with the primary ST/T results before any causal wording |

## Decisions for review

| # | Decision | Reason |
|---|---|---|
| G1 | Build Layer 2 from the raw full feature file, not the legacy master | Master dropped post-onset ECGs using troponin timing (2,479 MI admissions lost) |
| G2 | Latents are eligible for the latent audit | Provenance fully reproduced |
| G3 | `QRS_Dur_Global` reported with a caution; machine QRS added as a validation outcome | Systematic +65 ms offset, Spearman 0.69 vs machine |
| G4 | Primary ECG = earliest with features; sensitivity = admissions whose first ECG has features | 24,319 admissions differ |
| G5 | Add *after-strict* timing sensitivity (admission context and legacy troponin agree) | Two timing rules conflict for 1,667 admissions |
| G6 | Track `src/data/ecg_dataset.py` in git | Code that produced the embeddings was untracked |
