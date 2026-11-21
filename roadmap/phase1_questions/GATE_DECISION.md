# Phase 1 — Gate Decision

**Gate criterion (roadmap):** both questions are stated without a treatment or a discharge time zero.

**Result: PASS (pending reviewer sign-off on the decisions below).**
Layer 1 (chronic conditions → acute MI coded at admission) and Layer 2 (acute MI → ECG features) are
defined at the admission level with no treatment, no time zero and no follow-up. See `ESTIMANDS.md`.

## Decisions that go beyond or sharpen the roadmap — confirm before Phase 2

| # | Decision | Reason |
|---|---|---|
| D1 | ICD-9 410.x2 is not acute MI | It codes a follow-up admission for an earlier MI (965 admissions affected) |
| D2 | Layer 1 uses all 430,852 admissions, not only ECG admissions | Layer 1 does not need the ECG; restriction only adds selection |
| D3 | Layer 2 primary estimand is the ATT; ATE secondary | Natural question; overlap needed only on the control side |
| D4 | Primary ECG family = 53 point features; `_iqr` secondary | `_iqr` describes beat-to-beat spread / signal quality |
| D5 | `QRS_Dur_Global_count` is a quality flag, not a covariate | It can be changed by MI (rate/rhythm); adjusting would block part of the effect |
| D6 | Timing modifier = hours from ED registration/admission to ECG | Available for all; `index_mi_time` is troponin-based and often missing |
| D7 | Chest-pain active comparator, STEMI/NSTEMI split, type-2 sensitivity, negative control are pre-specified | Address ECG selection and label circularity by design |
| D8 | Type 2 MI stays in the primary exposure | Roadmap defines MI as I21/I22; excluding it is a sensitivity |

## Carried to Phase 2 (Gate 1)
- Chronic flags: build primary (prior + same-admission) and strict versions; check coverage.
- MI label: compare master `Label_Troponin` with `acute_mi`; confirm label was not read from the ECG.
- Count admissions with/without an ECG; classify every ECG before / after / uncertain by admission context.
- β-VAE provenance: re-encode a sample with `best_model.pt` and compare with stored embeddings.
