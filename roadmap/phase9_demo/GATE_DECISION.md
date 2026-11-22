# Phase 9 — Gate Decision

**Gate criterion (roadmap):** Gate 2 passed and every acceptance check passes.

**Result: PASS** (pending reviewer sign-off). Gate 2 passed in Phase 5; all six roadmap acceptance checks
pass (`ACCEPTANCE.md`, 10 automated tests).

## Decisions for review

| # | Decision | Reason |
|---|---|---|
| W1 | The demo shows a per-measure comparison and a per-group summary only, never a score | Roadmap "what it must never do" |
| W2 | Inputs far outside the training range are rejected; mildly outside are flagged and clipped | Clear rejection without discarding plausible ECGs |
| W3 | A "raw ECGdeli file" option swaps aVL/aVF, on by default for loaded files | Files from the project pipeline carry the swapped labels |
| W4 | The demo stays a local HTML file; it is not published to a hosted page. **Updated 2026-11-22: the repository owner chose to publish the demo files (`demo_params.json`, `ecg_explainer.html`) in the public GitHub repository.** | They contain only aggregate parameters (model coefficients, percentiles, effects); no patient rows. Sharing remains the owner's responsibility under the PhysioNet data-use agreement |
| W5 | The example input uses training medians, not a real patient | No patient-level data in the demo |
