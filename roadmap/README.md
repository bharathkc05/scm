# Causal Model of Acute MI — Roadmap Restart (2026-11-21)

Follows *Causal Model of Acute MI: Roadmap and Novelty* (Nov 14, 2026).
Data: MIMIC-IV v2.2 + MIMIC-IV-ECG v1.0. The previous attempt and the closed SBP / statin
branches are archived at `C:\Users\Acer\Desktop\Capstone\_archive_pre_roadmap\` (reference only).

## Rules
- All paths come from `config.py` (`python -m roadmap.config` checks them).
- Reuse the splits and longitudinal files; do not retrain models. Exception (Phase 2, G1): Layer 2 is
  built from the raw full feature file, because the legacy master dropped post-onset ECGs by troponin timing.
- Patient-level outputs go to `data/roadmap/` (git-ignored); only aggregates and code are committed.
- Each phase writes its outputs into its own folder plus a `GATE_DECISION.md`.
- Lock estimands and adjustment sets (commit them) before running Phase 5.
- Never use troponin, `Label_Troponin` or `index_mi_time` as covariates.

## Phases
| # | Folder | Output | Gate |
|---|---|---|---|
| 1 | `phase1_questions/` | Estimand note (Layer 1, Layer 2) | No treatment / discharge time zero |
| 2 | `phase2_data_audit/` | Data audit report | **Gate 1** |
| 3 | `phase3_dataset/` | One row per admission, causal dataset | Counts match audit |
| 4 | `phase4_graph/` | Graph, adjustment table, independence tests (judge by effect size) | Tests don't contradict graph |
| 5 | `phase5_estimation/` | Cross-fitted doubly robust effects, patient bootstrap, FDR | **Gate 2** |
| 6 | `phase6_novelty/` | Deconfounded ECG effects, latent audit, model-implied ECGs | Stable across sensitivity |
| 7 | `phase7_stress_test/` | E-values, ECG-selection weights, alt. MI codes, subgroups | Survivors stated openly |
| 8 | `phase8_paper/` | Manuscript | Every claim tied to an assumption |
| 9 | `phase9_demo/` | ECG explainer (only if Gate 2 passes) | Acceptance checks |

## Improvements to build in (from review of the previous attempt)
- **Phase 1:** STEMI/NSTEMI split (circularity check); chest-pain-ruled-out comparison group; Layer 1 on all admissions, not only ECG admissions.
- **Phase 2:** classify ECG timing by admission context (ED arrival, emergency vs elective), not troponin; verify β-VAE provenance by re-encoding a sample from `ecg_signals.h5`.
- **Phase 3:** 53 point ECG features as the primary family; `_iqr` / `_count` as quality covariates; territory-level ST features.
- **Phase 5:** add overlap-weighted estimates alongside clipped AIPW.
- **Phase 6:** negative control on before-onset ECGs (non-zero = bias + pre-existing CAD, an upper bound).
