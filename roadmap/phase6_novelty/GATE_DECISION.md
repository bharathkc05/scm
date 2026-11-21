# Phase 6 — Gate Decision

**Gate criterion (roadmap):** results are stable across the sensitivity checks.

**Result: PASS** (pending reviewer sign-off). Every pre-specified analysis set keeps ≥ 91% sign agreement
with the primary significant features (ST/T ≥ 95%), and ST/T magnitude stays at 0.98–1.36× the primary.
The Gate 1 condition (NSTEMI-only agreement) is met. Evidence: `NOVELTY_REPORT.md`.

## Decisions for review

| # | Decision | Reason |
|---|---|---|
| S1 | Layer 2 ST/T results may be described as effects of MI under the Phase 4 assumptions, always with the negative-control bound (≈ 38% of magnitude) | Gate 1 NSTEMI condition met; stable across all sets |
| S2 | The claimed robust MI signs are 14 point features + lateral ST and maximum ST depression; QRS duration and RR are excluded from claims | QRS not confirmed by machine measurement; RR equally present before MI |
| S3 | `z_ecg` latents are reported as a negative result and not used as an MI-specific node in later work or the demo | MI explains < 0.2% of each latent's variance; no MI information beyond handcrafted features |
| S4 | Counterfactual profiles stay an exploratory appendix | Model-implied, not observed; heterogeneity untested until Phase 7 |
| S5 | Chest-pain comparator results are reported next to the primary results | Same work-up as MI patients; effects larger, which argues against selection artefacts |
