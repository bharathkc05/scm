# Phase 5 — Gate 2 Decision: are the estimates usable?

**Roadmap criteria:** overlap, balance and held-out checks pass.

**Result: PASS** (pending reviewer sign-off). Evidence: `ESTIMATION_REPORT.md`.

| Criterion | Layer 1 (primary estimand) | Layer 2 (ATT) |
|---|---|---|
| Overlap | ATE restricted to propensity 0.01–0.99 (100% of admissions for D, H; 82% CKD; 78% prior MI) | Exposed propensity ≤ 0.08; max control weight 0.089 |
| Balance (max weighted \|SMD\| < 0.10) | ≤ 0.013 | 0.002 |
| Held-out (test split) | 7/7 RRs on the same side of 1 | 34/34 significant features keep sign; median diff 0.023 SD |

Gate 2 clears the estimates as statistically usable. Causal wording for Layer 2 still requires the Gate 1
condition: the NSTEMI-only analysis must agree in direction with the ST/T pattern (Phase 6).

## Decisions for review

| # | Decision | Reason |
|---|---|---|
| R1 | Layer 1 primary estimand = ATE in the overlap region; ATT alongside | Full ATE fails positivity for CKD and prior MI; identical to full ATE for D and H |
| R2 | Nuisance specification = 5-df age spline with age × covariate and covariate × covariate interactions | Chosen on validation for both layers; simpler models left CKD/prior-MI age unbalanced |
| R3 | Layer 2 ATE reported but not interpreted | ATE weights leave age unbalanced (SMD 0.19) |
| R4 | aVL/aVF label fix applied at read time; Phase 3 rebuilt with a permanent identity check | MATLAB extraction assumed the wrong lead order |
| R5 | `QRS_Dur_Global` effect flagged as not validated; RR validated | Machine QRS shows no effect; machine RR matches |
