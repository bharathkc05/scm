# A Causal Analysis of Acute Myocardial Infarction: Risk Factors and ECG Responses in MIMIC-IV

**Draft v1.1 — 2026-11-21** (references updated after the targeted literature review). Authors and
affiliations: *to be added.* Tables: `TABLES.md` (generated). Figures: `../phase6_novelty/figures/`.
Every claim is mapped to its evidence and assumptions in `CLAIM_EVIDENCE_MAP.csv`. Reference status
(`LITERATURE_REVIEW.md`): ✅ verified; ◐ located, details to complete.

---

## Abstract

**Background.** Machine-learning models detect myocardial infarction (MI) from the electrocardiogram (ECG),
but they do not separate ECG changes caused by MI from differences that only reflect who tends to have an
MI. We built a two-layer structural causal model in which chronic conditions act on MI and MI acts on the
ECG.

**Methods.** We studied 430,852 adult admissions in MIMIC-IV v2.2, and the 188,808 of them that had an
ECG with delineated features in MIMIC-IV-ECG. Acute MI was defined from ICD-9 410.x0/x1 and ICD-10
I21/I22 codes. Layer 1 estimated how diabetes, hypertension, chronic kidney disease (CKD) and prior MI
relate to acute MI at admission. Layer 2 estimated the effect of acute MI on 53 handcrafted ECG features
and 64 β-variational-autoencoder latents. Layer 2 used only ECGs that followed MI onset by admission
context, and estimated the average effect among MI admissions (ATT). We used a revised, tested causal
graph, cross-fitted doubly robust estimators, patient-level bootstrap and false-discovery control. Stress
tests covered an NSTEMI-only analysis, a chest-pain comparator, a negative control, ECG-selection
weighting, BMI adjustment, alternative code definitions, subgroups and E-values.

**Results.** In Layer 1, diabetes (RR 1.54–1.67 depending on the adjustment set), hypertension
(1.44, 95% CI 1.35–1.53) and prior MI (2.35) were associated with acute MI under every definition.
CKD (1.34) disappeared for MI coded as the principal diagnosis (0.97). In Layer 2, MI depressed the ST
segment and flattened T waves in the lateral leads, with reciprocal changes in aVR: lateral ST −0.36 SD
(−0.40 to −0.33), T in V5 −0.33 SD. Fourteen features and two composites survived every check. The pattern
was stronger in NSTEMI alone (lateral ST −0.50 SD), so it does not arise from STEMI coding. In a negative
control, ECGs recorded before a later MI showed 38% of the effect size. Crude R, Q and S amplitude
differences were explained by confounding. MI explained less than 0.2% of the variance of each latent
dimension.

**Conclusions.** In hospitalized patients, acute MI causes a lateral ST-depression and T-wave pattern that
is robust to the main threats we could test. About 60% of it is specific to the acute event. Much of what
separates MI from non-MI ECGs in prediction models is confounding, not an effect of MI.

---

## 1. Introduction

Models that detect MI from the 12-lead ECG reach high discrimination [Strodthoff 2021; Al-Zaiti 2023], and our own
baseline models did so in this cohort (AUROC up to 0.87; Table S1). Such models answer *whether* an ECG
belongs to a patient with MI. They do not answer *what MI does to the ECG*. A classifier is free to use any
feature that differs between MI and non-MI patients, including features that differ only because MI
patients are older, more often male and more comorbid. For interpretation, clinical teaching and model
auditing, the causal question matters: which ECG changes are produced by the infarction itself?

We address two questions with a structural causal model on routinely collected hospital data:

1. **Layer 1.** Among admitted adults, how are diabetes, hypertension, CKD and prior MI, recorded at or
   before the admission, related to acute MI coded at that admission?
2. **Layer 2.** Among admitted adults with an ECG recorded after MI onset, what is the effect of acute MI
   on each ECG feature and on each dimension of a learned ECG representation, after adjusting for age, sex
   and comorbidity?

Layer 1 is a transparent, assumption-explicit confirmation of established risk factors, not a new discovery.
The contribution is Layer 2, which treats the ECG as a consequence of MI rather than a predictor. It also
supplies two audits that prediction studies do not: separating MI-attributable from confounded ECG
differences, and a causal audit of a β-VAE embedding.

Related work has brought causal or counterfactual ideas to ECG analysis, but in the service of prediction.
- Back-door adjustment has been built into classifiers to make them robust [Causal ECGNet 2025◐].
- Counterfactual ECGs have been used to explain MI classifiers, by showing which input change would flip the
  prediction [Tanyel 2025◐].
- β-VAE factors have been used as interpretable predictors [van de Leur 2022].
- Demographic "shortcuts" are typically checked only by ablation or subgroup performance.

None of these estimates what MI does to each ECG feature under an explicit causal graph. None asks how much of
a learned representation MI causes. A targeted literature search (November 2026; `LITERATURE_REVIEW.md`)
found no such study. To our knowledge this combination has not been reported; a formal systematic search
is planned before submission.

## 2. Data and cohort

**Sources.** MIMIC-IV v2.2 [Johnson 2023] provided admissions, demographics and ICD-9/ICD-10 diagnoses
for 431,231 adult admissions of 180,733 patients at Beth Israel Deaconess Medical Center. MIMIC-IV-ECG v1.0
[Gow 2023] provided 800,035 diagnostic 12-lead ECGs.

**ECG features.**
- *Handcrafted:* ECGdeli [Pilia 2021] delineated each ECG. We used 53 point measurements (ST level and
  Q, R, S and T amplitudes per lead, global QRS duration and RR interval); 41 beat-to-beat interquartile
  ranges were secondary outcomes.
- *Lead-label correction:* MIMIC-IV-ECG stores the augmented limb leads in the order aVR, aVF, aVL, but
  the extraction pipeline labelled them aVR, aVL, aVF. We detected this from violations of the Goldberger
  identity (aVF = (II + III)/2; r = −0.12 before vs 0.87–0.90 after relabelling) and corrected it.
- *Latents:* 64 latent means from a β-VAE (β = 4) trained without labels on per-lead standardized 10-s
  waveforms. Stored embeddings reproduced exactly from the checkpoint (maximum difference 2 × 10⁻⁶).

**Unit and linkage.** The unit is the admission. ECGs were linked to the admission whose window (ED
registration, or admission if there was none, to discharge) contained them; 464,033 ECGs were linked.
Layer 2 uses the earliest ECG with features per admission. Patients were kept whole within one data split
(70/15/15).

**Cohort flow** (`../phase3_dataset/cohort_flow.csv`).
- Layer 1: 430,852 admissions with at least one diagnosis code; 11,222 with acute MI.
- 193,470 admissions had an ECG with features.
- Layer 2 primary cohort: 188,808 admissions (98,662 patients) — 5,643 after-onset MI admissions and
  183,165 controls.
- Excluded from the Layer 2 primary cohort: 3,780 MI admissions with uncertain timing, 805 ICD-9 410.x2
  follow-up admissions and 77 admissions without diagnosis codes.

Table 1 describes the cohorts.

**Relation to prior baseline models.** Earlier prediction models on these data (Table S1) are reported for
context only. They used a troponin-timed label built from MIMIC-IV-ECG-Ext-ICD v1.0.1 [Strodthoff, PhysioNet], a cohort
that excluded post-onset ECGs, the uncorrected
aVL/aVF labels, and in-admission treatments as predictors. Their discrimination is therefore not comparable
to, and not evidence for, the causal estimates.

## 3. Causal framework

**Definitions** (`../phase1_questions/ESTIMANDS.md`).
- *Acute MI:* ICD-9 410.x0/x1 or ICD-10 I21/I22 at the current admission. ICD-9 410.x2 (subsequent episode
  of care) and old-MI codes are excluded.
- *Chronic conditions:* a code at any earlier admission, or a chronic code at the current admission.
  A strict prior-only definition was a sensitivity.
- *Prior MI:* an earlier acute MI code, or an old-MI code (412, I25.2). It is treated as a descriptive
  marker of coronary disease.

**Timing.** A causal ECG effect requires the ECG to follow MI onset. Troponin timestamps lag onset, so we
classified timing by admission context:
- *After onset:* MI was the principal diagnosis of an unplanned admission. Median time from arrival to ECG
  was 0.83 h.
- *Uncertain:* MI was coded secondary, or the admission was planned.
- *Before onset:* the ECG came from an earlier admission of a patient who later had an MI.

**Graph** (`../phase4_graph/GRAPH.md`). The roadmap draft graph had age and sex affecting all variables,
diabetes and hypertension affecting CKD, these conditions affecting prior MI, acute MI and the ECG, and
acute MI affecting the ECG and the probability of being recorded. Its testable implication that diabetes is
independent of hypertension given age and sex failed by a pre-declared effect-size criterion (adjusted
OR 5.77). We revised the graph to diabetes → hypertension plus an unmeasured shared metabolic cause.

Adjustment sets were derived by the back-door criterion under the revised graph and seven alternatives,
with an s-back-door check for ECG selection:
- *Hypertension:* adjusted for age, sex and diabetes.
- *Diabetes:* its total effect is not identified under the revised graph, so we report it as a bracket
  between {age, sex} and {age, sex, hypertension}.
- *CKD:* adjusted for age, sex, diabetes and hypertension.
- *Layer 2:* adjusted for age, sex, diabetes, hypertension, CKD and prior MI.

The ECG is one multivariate node; we make no claim about independence between features.

**Assumptions.** A1, history codes reflect pre-existing conditions. A2, after-onset ECGs follow MI onset.
A3, no unmeasured common cause of MI and the ECG beyond the adjustment set. A4, ECG recording is ignorable
given MI and measured covariates. A5, no unmeasured common cause of each condition and MI (Layer 1; known
to be violated by smoking and lipids, hence bounded). A6, the shared metabolic cause acts on MI only through
diabetes and hypertension.

## 4. Methods

**Estimators.** Augmented inverse-probability weighting (AIPW) with 5-fold cross-fitting, patients kept
whole within folds [Bang & Robins 2005; Chernozhukov 2018].
- *Layer 1:* risk differences and risk ratios for the ATE in the overlap region (propensity 0.01–0.99)
  [Crump 2009], with the ATT alongside. For diabetes and hypertension the overlap region includes every
  admission. For CKD and prior MI it includes 82% and 78% of admissions.
- *Layer 2:* the primary estimand is the ATT in control-standard-deviation units. ATE weights left age
  imbalanced, so the ATE is not interpreted.
- *Nuisance models:* logistic models (propensity, binary outcomes) and multi-output least squares
  (continuous outcomes). The specification was chosen on the validation split: a 5-df age spline with
  age × covariate and covariate × covariate interactions.
- *Outcomes:* winsorized at the training-split 0.5th and 99.5th percentiles; raw values were a sensitivity
  (effect correlation 0.94).

**Inference.** 1,000 patient-level bootstrap resamples with nuisances fixed (500 in stress tests).
Benjamini–Hochberg FDR within the 53 point features, the 41 IQR features and the 64 latents
[Benjamini & Hochberg 1995].

**Held-out check.** The test split was used once. Nuisances were fitted on training plus validation data
and the estimates recomputed on test admissions.

**Deconfounded ECG effects.** Each feature was classified as MI-attributable, partly confounded, explained
by confounding, revealed by adjustment, or not associated, by comparing crude and adjusted contrasts. A
feature was *robust* if it was significant after adjustment, kept its sign in NSTEMI, and had a negative-
control effect below half the primary effect [Lipsitch 2010].

**Latent audit.** For each latent we computed the cross-fitted R² from baseline factors, and the additional
R² from MI. We also computed the out-of-fold AUC for MI from baseline-residualized latents and from
baseline-residualized handcrafted features.

**Model-implied counterfactual ECGs.** For illustrative profiles we report the outcome-model predictions
with and without MI. These are exploratory conditional predictions, not observed counterfactuals.

**Stress tests.**
- E-values for Layer 1 risk ratios, and for Layer 2 via RR ≈ exp(0.91·d) [VanderWeele & Ding 2017;
  VanderWeele 2017].
- Inverse-probability-of-selection weights for having a feature ECG, modelled on 425,146 eligible
  admissions [Bareinboim & Tian 2015].
- Adjustment for pre-admission BMI from the outpatient record (55% coverage).
- Principal-only, broad and type-1-only MI definitions.
- Age (<65 / ≥65) and sex subgroups with interaction tests.
- Strict-flag and look-back variants of Layer 1.

**Reproducibility.** All code is in `roadmap/` (one folder per phase, single path configuration, fixed seed
20261121). Patient-level data stay within the credentialed MIMIC environment.

## 5. Results

### 5.1 Layer 1: chronic conditions and acute MI (Table 2)

Acute MI occurred in 2.60% of admissions.
- **Diabetes:** RR 1.67 (1.60–1.75) adjusting for age and sex, and 1.54 (1.48–1.62) when hypertension is
  also adjusted. The true total effect lies within this bracket under the revised graph only if one of its
  two limiting structures holds.
- **Hypertension:** RR 1.44 (1.35–1.53) given age, sex and diabetes.
- **CKD:** RR 1.34 (1.27–1.42) given age, sex, diabetes and hypertension.
- **Prior MI:** RR 2.35 (2.19–2.50), descriptive.

Weighted standardized mean differences were ≤ 0.013 for every primary estimate, and every test-split RR lay
on the same side of 1 (test RRs 1.22–2.17).

### 5.2 Layer 2: the effect of MI on the ECG (Figure 2, Table 3)

Weighting balanced the after-onset MI admissions and controls (maximum |SMD| 0.002; effective control
sample 139,545).
- **Significance:** 34 of 53 point features were significant at FDR 5%.
- **Groups:** ST (12 of 12; mean |effect| 0.18 SD) and T amplitude (10 of 12; 0.19 SD) dominated. R, S and
  Q amplitudes changed little (0.03–0.04 SD).
- **Largest effects:** ST depression in I (−0.36 SD) and aVL (−0.34), T flattening in V5 (−0.33) and V6
  (−0.31), and higher T (+0.29) and ST (+0.24) in aVR.
- **Composites:** lateral ST −0.36 SD (−0.016 mV); maximum ST depression −0.23 SD.
- **Machine-measured checks:** the RR-interval effect was confirmed (+19.4 vs +19.5 ms). The ECGdeli
  QRS-duration effect was not (machine +0.4 ms, CI −0.9 to 1.6), so QRS is excluded from the claims.
- **Held-out:** all 34 significant features kept their sign on the test split; median difference 0.023 SD.

### 5.3 Which ECG differences are caused by MI (Figure 1)

- **MI-attributable:** 21 of 24 ST and T features kept at least half their crude difference after
  adjustment.
- **Explained by confounding:** 15 of the 16 features in this class were R, Q or S amplitudes. Their crude
  MI-vs-control differences reflect patient characteristics.
- **Robust signs:** 14 features passed every robustness criterion — ST in I, aVL, V4, V5, V6, III and aVR;
  T in V4, V5, V6, aVR and II; and Q in I and aVL. So did the lateral ST and maximum ST-depression
  composites.
- **Not specific to MI:** the RR-interval effect was as large in ECGs recorded before a later MI
  (negative-control ratio 1.05).

### 5.4 Circularity, selection and the negative control

- **NSTEMI only** (4,149 cases), whose diagnosis does not rest on ST elevation: the ST/T pattern kept its
  direction (95% of significant ST/T features) and was stronger (1.25×; lateral ST −0.50 SD). Only maximum
  ST elevation vanished (+0.15 → −0.01 SD).
- **Chest-pain comparator** (similar ECG and troponin work-up: 88% and 93%): effects were larger (1.36×).
- **Negative control** (6,938 later-MI patients): ECGs recorded before the MI showed 38% of the primary
  ST/T magnitude (lateral ST −0.10 SD). This combines residual confounding, selection and pre-existing
  coronary disease, so about 60% of the effect is specific to the acute event.

### 5.5 The β-VAE latents (Figure 3)

- 43 of 64 latents differed significantly by MI, but MI explained a median of 0.009% (maximum 0.12%) of each
  latent's variance, below baseline factors in 63 of 64.
- The MI signal concentrated in a few dimensions tracking T-wave amplitude (z57 and z3 with T in aVR).
- Baseline-residualized latents predicted MI no better than residualized handcrafted features (AUC 0.659 vs
  0.653).

### 5.6 Model-implied counterfactual ECGs (Figure 4, exploratory)

The predicted lateral ST shift was −0.17 SD for a 55-year-old man without comorbidity and −0.79 SD for a
70-year-old man with diabetes, hypertension, CKD and prior MI. Only 20–25% of MI admissions deviated more
than 1 SD from their own predicted no-MI value in the direction of the effect.

## 6. Sensitivity analyses (`../phase7_stress_test/STRESS_TEST_REPORT.md`)

- **Unmeasured confounding.**
  - Layer 1 E-values: 2.02–2.73 (CI limits 1.85–2.58).
  - Layer 2 ST/T features: 1.55–2.13 (lateral ST 2.13; CI 2.03).
  - Q-amplitude features: 1.34–1.36.
  - A hidden confounder would need associations stronger than diabetes's RR of 1.67 with both MI and the
    lateral ST level.
- **ECG selection.** Weighting to all eligible admissions (weights 99th percentile 2.8) kept all 16 robust
  measures (magnitude 0.99×).
- **BMI.** Adjustment changed Layer 1 RRs by ≤ 0.03 and Layer 2 effects by < 1%.
- **Outcome codes.** Diabetes and hypertension held under broad (RR 1.69, 1.49), type-1-only (1.59, 1.40)
  and principal-only (1.44, 1.34) definitions. CKD did not (principal-only 0.97, 0.91–1.05). Layer 2 held
  with the broad definition (0.97×).
- **History definition.** Strict prior-only flags reversed hypertension (RR 0.72). Strict flags are zero on
  every first admission, and MI is more common on first admissions (3.27% vs 2.12%). Restricting to
  admissions with look-back raised every RR (hypertension 2.24), so the primary estimates are conservative.
- **Subgroups.** All 16 robust measures held in every age and sex subgroup.
  - Lateral precordial ST depression was larger at ≥ 65 years (V5: −0.29 vs −0.11 SD).
  - Inferior and high-lateral changes were larger under 65.
  - T-wave changes and ST depression were larger in women.

## 7. Discussion and limitations

Treating the ECG as an effect of MI gives a different picture from treating it as a predictor. The
MI-attributable ECG signature in hospitalized patients is lateral ST depression and T-wave flattening with
reciprocal aVR elevation — the subendocardial-ischemia pattern expected in a cohort where three-quarters of
infarctions are NSTEMI. The literature reads aVR elevation with diffuse ST depression as supply–demand
ischemia that predicts severe multivessel disease, rather than acute occlusion [aVR review◐]. Several crude
differences that a classifier would
exploit (R, Q and S amplitudes) are explained by who has an MI rather than by the infarction. The
pattern's persistence in NSTEMI, against chest-pain comparators, after selection weighting and in every
subgroup argues against coding circularity and selection as its source. The negative control limits the
claim: roughly 40% of the effect magnitude is not specific to the acute event.

The Layer 1 associations point the same way as the INTERHEART case-control estimates but are smaller
(diabetes OR 2.37, hypertension OR 1.91 there [Yusuf 2004]). This is expected when everyone studied is
hospitalized, because restricting to a common effect of illness distorts risk-factor associations
[Griffith 2020]. It also matches our look-back analysis, where removing a recording artefact
increased every risk ratio. The CKD association disappeared for MI coded as the reason for admission. That
fits the literature linking CKD to NSTEMI and supply–demand infarction rather than to presenting MI
[CKD–MI studies◐].

The latent audit is a negative result with practical value. Earlier work used β-VAE factors as
interpretable predictors [van de Leur 2022]. Our audit adds the causal question. The β-VAE representation is dominated by
variation unrelated to MI and to measured patient factors, and adds no MI information beyond handcrafted
features. It should not be used as an MI-specific node in causal models.

**Limitations.**
1. Diagnosis flags are not interventions; Layer 1 estimates are risk-factor contrasts among hospitalized
   adults, subject to hospital-population selection, and the diabetes total effect is not identified.
2. Smoking and lipids are unmeasured. E-values bound but do not remove this bias.
3. MI and history are taken from administrative codes without adjudication. MI timing is inferred from
   admission context; 3,780 MI admissions with uncertain timing were excluded.
4. Hospitalized patients with an ECG are not the general population. Selection weighting assumes recording
   is ignorable given measured variables.
5. ECG features come from automated delineation. The QRS-duration measure disagreed with machine
   measurements, and a lead-labelling error was found and corrected during the analysis.
6. Model-implied counterfactual ECGs are conditional predictions, not observed counterfactuals.
7. Single centre; external validation is needed.
8. The literature search was targeted, not systematic; a formal systematic search is planned.

**Conclusion.** Under explicit and partly tested assumptions, acute MI causes a lateral ST-depression and
T-wave pattern in hospitalized patients. About 60% of this pattern is specific to the acute event, and it
is robust to the main threats we could test. Separating what MI does to the ECG from who has an MI should
inform how ECG-based MI models are built and interpreted.

---

## References

✅ verified in the targeted review; ◐ located, details to complete (see `LITERATURE_REVIEW.md`).

**Data and tools**
- ✅ Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Sci Data*. 2023;10:1. doi:10.1038/s41597-022-01899-x. *(Also cite the MIMIC-IV v2.2 PhysioNet dataset record — DOI to add.)*
- ✅ Gow B, Pollard T, Nathanson LA, et al. MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (version 1.0). PhysioNet. 2023. doi:10.13026/4nqg-sb35.
- ✅ Strodthoff N, Lopez Alcaraz JM, Haverkamp W. MIMIC-IV-ECG-Ext-ICD: Diagnostic labels for MIMIC-IV-ECG (version 1.0.1). PhysioNet; arXiv:2312.11050.
- ✅ Pilia N, Nagel C, Lenis G, Becker S, Dössel O, Loewe A. ECGdeli — an open source ECG delineation toolbox for MATLAB. *SoftwareX*. 2021;13:100639. doi:10.1016/j.softx.2020.100639.
- ✅ Thygesen K, Alpert JS, Jaffe AS, et al. Fourth universal definition of myocardial infarction (2018). *Circulation*. 2018;138:e618–e651. doi:10.1161/CIR.0000000000000617.

**Methods**
- ✅ Bang H, Robins JM. Doubly robust estimation in missing data and causal inference models. *Biometrics*. 2005;61(4):962–973. doi:10.1111/j.1541-0420.2005.00377.x.
- ✅ Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine learning for treatment and structural parameters. *Econometrics J*. 2018;21(1):C1–C68.
- ✅ Crump RK, Hotz VJ, Imbens GW, Mitnik OA. Dealing with limited overlap in estimation of average treatment effects. *Biometrika*. 2009;96(1):187–199. doi:10.1093/biomet/asn055.
- ◐ Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc B*. 1995;57(1):289–300.
- ✅ VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. *Ann Intern Med*. 2017;167(4):268–274. doi:10.7326/M16-2607.
- ✅ VanderWeele TJ. On a square-root transformation of the odds ratio for a common outcome. *Epidemiology*. 2017;28(6):e58–e60.
- ✅ Lipsitch M, Tchetgen Tchetgen E, Cohen T. Negative controls: a tool for detecting confounding and bias in observational studies. *Epidemiology*. 2010;21(3):383–388. doi:10.1097/EDE.0b013e3181d61eeb.
- ✅ Bareinboim E, Tian J. Recovering causal effects from selection bias. *Proc AAAI Conf Artif Intell*. 2015;29:3475–3481. doi:10.1609/aaai.v29i1.9679.
- ✅ Higgins I, Matthey L, Pal A, Burgess C, Glorot X, Botvinick M, Mohamed S, Lerchner A. β-VAE: learning basic visual concepts with a constrained variational framework. *ICLR*. 2017.
- ◐ Hernán MA, Robins JM. *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC; 2020.
- ✅ Griffith GJ, et al. Collider bias undermines our understanding of COVID-19 disease risk and severity. *Nat Commun*. 2020. doi:10.1038/s41467-020-19478-2.

**ECG and clinical context**
- ✅ Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep learning for ECG analysis: benchmarks and insights from PTB-XL. *IEEE J Biomed Health Inform*. 2021;25(5). doi:10.1109/JBHI.2020.3022989.
- ✅ Al-Zaiti SS, et al. Machine learning for ECG diagnosis and risk stratification of occlusion myocardial infarction. *Nat Med*. 2023;29(7):1804–1813. doi:10.1038/s41591-023-02396-3.
- ✅ van de Leur RR, et al. Improving explainability of deep neural network-based electrocardiogram interpretation using variational auto-encoders. *Eur Heart J Digit Health*. 2022;3:390–404.
- ◐ Causal ECGNet: leveraging causal inference for robust ECG classification in cardiac disorders. 2025. PubMed 40458653 (authors and journal to complete).
- ◐ Tanyel T, et al. Interpretable ECG analysis for myocardial infarction detection through counterfactuals. *Biomed Signal Process Control*. 2025; arXiv:2312.08304 (volume to complete).
- ✅ Yusuf S, et al. Effect of potentially modifiable risk factors associated with myocardial infarction in 52 countries (the INTERHEART study): case-control study. *Lancet*. 2004;364(9438):937–952. doi:10.1016/S0140-6736(04)17018-9.
- ◐ Review of ST elevation in aVR with diffuse ST depression (PMC12966643; full citation to complete).
- ◐ CKD and MI subtype studies (ISCHEMIA-CKD analyses; albuminuria/eGFR and NSTEMI risk, *JAHA* 2018; KidDiCo) — full citations to complete.
