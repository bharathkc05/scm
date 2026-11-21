# Phase 4 — Gate Decision

**Gate criterion (roadmap):** tests do not contradict the graph, or the graph is revised.

**Result: PASS after revision** (pending reviewer sign-off). The draft's implication D ⊥ H | age, sex was
contradicted by effect size (OR 5.77; strict 6.20 with look-back). The graph is revised to diabetes →
hypertension plus a shared metabolic cause. The revised graph's remaining testable implication
(age ⊥ sex) holds. Evidence: `GRAPH.md`.

## Decisions for review

| # | Decision | Reason |
|---|---|---|
| Q1 | Primary graph = revised graph (D → H + unmeasured shared metabolic cause) | Draft contradicted; direction from clinical knowledge |
| Q2 | Hypertension primary set = age, sex, **diabetes**; roadmap set (age, sex) becomes a sensitivity | Valid under the revised graph and all variants except H → D |
| Q3 | Diabetes reported as a bracket: {age, sex} and {age, sex, HTN}, labelled not identified under the revised graph | Neither set is valid when D → H and they share a cause |
| Q4 | CKD and prior-MI sets unchanged; CKD + prior MI as the "run both ways" sensitivity | Robust across graphs except prior MI → CKD |
| Q5 | Layer 2 set unchanged; add a BMI sensitivity (pre-admission OMR BMI, where available) in Phase 7 | Unmeasured cause of MI and ECG is the main threat to Layer 2 |
| Q6 | Look-back stays a sensitivity (restriction / covariate), not in primary sets | Adjusting can open collider paths; recording bias handled by primary flags |
| Q7 | ECG = one multivariate node; no feature-pair independence tests | Shared waveform structure, not a causal claim |
| Q8 | Time-to-ECG analyses are descriptive | Time to ECG is a consequence of MI |
