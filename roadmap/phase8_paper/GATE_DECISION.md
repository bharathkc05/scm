# Phase 8 — Gate Decision

**Gate criterion (roadmap):** every claim is tied to an assumption.

**Result: PASS for draft v1** (pending reviewer sign-off). `CLAIM_EVIDENCE_MAP.csv` lists 20 claims, each
with its evidence files, the assumptions it rests on (A1–A6, Section 3 of the manuscript) and its strength.
All numbers in the tables are generated from phase outputs by `p8_01_tables.py`.

## Outstanding before submission

| # | Item | Why |
|---|---|---|
| U1 | **Partly done (2026-11-21):** targeted literature review in `LITERATURE_REVIEW.md` (23 searches); 18 references verified (✅), 6 located with details to complete (◐); no study found estimating causal effects of MI on ECG features. **Remaining:** systematic PubMed/Embase search with a screening log, and completing the ◐ entries | The roadmap requires it before any novelty claim |
| U2 | Authors, affiliations, target journal and its format/word limits | Not specified |
| U3 | Clinical co-author review of the graph and the ECG interpretation | The roadmap's draft graph was "for clinical review" |
| U4 | MIMIC data-use and PhysioNet credentialing statements | Required by the data licence |
| U5 | Decide whether to merge Figures 1–4 into fewer panels for the journal | Format-dependent |

## Decisions for review

| # | Decision | Reason |
|---|---|---|
| V1 | Title kept as the roadmap's suggestion | Matches the results; may be shortened for the target journal |
| V2 | Layer 2 framed as causal under stated assumptions with the 38% negative-control bound; Layer 1 framed as risk-factor contrasts | Phase 6 S1, Phase 1 framing |
| V3 | Legacy baseline models appear only as context (Table S1) with their caveats | Different label, cohort, lead labels and post-onset predictors |
