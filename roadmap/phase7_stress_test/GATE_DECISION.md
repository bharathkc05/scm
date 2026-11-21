# Phase 7 — Gate Decision

**Gate criterion (roadmap):** which results survive is stated openly.

**Result: PASS** (pending reviewer sign-off). `STRESS_TEST_REPORT.md` Section 1 lists the survival of every
result against every stress test, including the failures (CKD; strict-flag Layer 1; fragile Q-amplitude
magnitudes).

## Decisions for review

| # | Decision | Reason |
|---|---|---|
| T1 | CKD → MI is reported as fragile: an association with MI coded as a secondary diagnosis, not a risk-factor effect on MI presentation | Null for principal-diagnosis MI (RR 0.97); weaker for type 1 (1.21) |
| T2 | Primary Layer 1 estimates are described as conservative; strict prior-only flags are never used without look-back handling | Strict flags reverse hypertension (0.72) through first-admission structural zeros; look-back analyses give larger RRs |
| T3 | Q-amplitude features (Q in I, aVL) are claimed for direction only; the paper's main ECG claims rest on ST and T | E-values 1.3–1.4; effects 0.07–0.08 SD |
| T4 | Age/sex differences in the ST/T pattern are reported as descriptive effect modification | Same pattern in every subgroup; magnitudes differ |
| T5 | Phase 9 demo eligible measures = the 14 point features + lateral ST + maximum ST depression; no latents, QRS or RR | Passed Gate 2, Phase 6 robustness and every Phase 7 test |
| T6 | BMI is not added to the primary adjustment sets | No measurable confounding; would drop 45% of admissions |
