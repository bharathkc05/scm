# Literature Review (targeted) — for MANUSCRIPT.md

**Date:** 2026-11-21 · **Method:** targeted web searches (23 queries). Each reference was checked against
publisher, PubMed/PMC, PhysioNet or arXiv records shown in the results. This is **not** a systematic review:
no PubMed/Embase/Scopus search strings were run and no screening log exists. Section 5 gives the strings
for the formal search still needed before submission.

## 1. Reference verification

✅ = located with full bibliographic details. ◐ = located, but some details still to complete.

| # | Reference | Status | Verified details |
|---|---|---|---|
| 1 | Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset | ✅ | *Sci Data* 2023;10:1. doi:10.1038/s41597-022-01899-x. Two author corrections (2023). Also cite the MIMIC-IV **v2.2** PhysioNet record actually used (DOI to add). |
| 2 | Gow B, Pollard T, Nathanson LA, et al. MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset (v1.0) | ✅ | PhysioNet 2023. doi:10.13026/4nqg-sb35. PhysioNet also lists a later journal article (not verified). |
| 3 | Strodthoff N, Lopez Alcaraz JM, Haverkamp W. MIMIC-IV-ECG-Ext-ICD: Diagnostic labels for MIMIC-IV-ECG | ✅ | PhysioNet v1.0.1 (version used); arXiv:2312.11050. Source of the legacy label. |
| 4 | Pilia N, Nagel C, Lenis G, Becker S, Dössel O, Loewe A. ECGdeli — an open source ECG delineation toolbox for MATLAB | ✅ | *SoftwareX* 2021;13:100639. doi:10.1016/j.softx.2020.100639 |
| 5 | Thygesen K, Alpert JS, Jaffe AS, et al. Fourth universal definition of myocardial infarction (2018) | ✅ | *Circulation* 2018;138:e618–e651. doi:10.1161/CIR.0000000000000617. Also published in *EHJ* and *JACC*. |
| 6 | Bang H, Robins JM. Doubly robust estimation in missing data and causal inference models | ✅ | *Biometrics* 2005;61(4):962–973. doi:10.1111/j.1541-0420.2005.00377.x |
| 7 | Chernozhukov V, et al. Double/debiased machine learning for treatment and structural parameters | ✅ | *Econometrics J* 2018;21(1):C1–C68 |
| 8 | Crump RK, Hotz VJ, Imbens GW, Mitnik OA. Dealing with limited overlap in estimation of average treatment effects | ✅ | *Biometrika* 2009;96(1):187–199. doi:10.1093/biomet/asn055 |
| 9 | VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value | ✅ | *Ann Intern Med* 2017;167(4):268–274. doi:10.7326/M16-2607 |
| 10 | VanderWeele TJ. On a square-root transformation of the odds ratio for a common outcome | ✅ | *Epidemiology* 2017;28(6):e58–e60 |
| 11 | Lipsitch M, Tchetgen Tchetgen E, Cohen T. Negative controls: a tool for detecting confounding and bias in observational studies | ✅ | *Epidemiology* 2010;21(3):383–388. doi:10.1097/EDE.0b013e3181d61eeb |
| 12 | Bareinboim E, Tian J. Recovering causal effects from selection bias | ✅ | *AAAI* 2015 (29th conf.), 3475–3481. doi:10.1609/aaai.v29i1.9679 |
| 13 | Higgins I, et al. β-VAE: learning basic visual concepts with a constrained variational framework | ✅ | *ICLR* 2017 |
| 14 | Benjamini Y, Hochberg Y. Controlling the false discovery rate | ◐ | Standard reference (*J R Stat Soc B* 1995;57(1):289–300); not re-checked in this search |
| 15 | Hernán MA, Robins JM. *Causal Inference: What If* | ◐ | Chapman & Hall/CRC 2020; not re-checked in this search |

**Version note.** The manuscript uses MIMIC-IV **v2.2**; PhysioNet's current page shows v3.1. Cite the v2.2
dataset record explicitly, as the data-use agreement requires.

## 2. Related work found

### 2.1 ECG-based MI detection (the prediction framing)
- **Strodthoff N, Wagner P, Schaeffter T, Samek W.** Deep learning for ECG analysis: benchmarks and insights
  from PTB-XL. *IEEE J Biomed Health Inform* 2021;25(5). doi:10.1109/JBHI.2020.3022989. ✅
  Convolutional networks outperform feature-based algorithms on ECG statement tasks.
- **Al-Zaiti SS, et al.** Machine learning for ECG diagnosis and risk stratification of occlusion myocardial
  infarction. *Nat Med* 2023;29(7):1804–1813. doi:10.1038/s41591-023-02396-3. ✅
  Derived on 7,313 patients; outperformed clinicians and commercial systems; ECG features reviewed by experts.
- **Emergency-department deep-learning MI prediction** (*Sci Rep* 2022) and **transfer learning for AMI**
  (*PLOS Digit Health*). ◐ These check demographic "shortcuts" only through ablations and subgroup AUROCs.
  Neither estimates what MI does to the ECG.

**Positioning:** these studies answer *which ECGs belong to MI patients*. Our Layer 2 answers *what MI does
to the ECG*. Our finding that crude R/Q/S differences are confounding identifies signals a classifier can
use without them being effects of MI.

### 2.2 Causal and counterfactual ideas applied to ECG classifiers
- **Causal ECGNet** (2025; PMC12127154, PubMed 40458653). ◐ Applies back-door adjustment per lead inside a
  classifier, for infection-related cardiac abnormalities. Its goal is classification robustness, not
  effect estimation.
- **Tanyel T, et al.** Interpretable ECG analysis for myocardial infarction detection through counterfactuals
  (VCCE). *Biomed Signal Process Control* 2025; arXiv:2312.08304. ◐ Counterfactual explanations of an MI
  classifier on PTB-XL, rated by two cardiologists.
- **Prototype-based sparse ECG counterfactuals** (arXiv:2510.19514, 2025). ◐ Explanation method.

**Positioning:** these "counterfactuals" are classifier explanations (what input change flips the
prediction). Our model-implied counterfactual ECGs come from a structural model with an explicit graph and
adjustment set; they estimate an effect, not a decision boundary.

### 2.3 Learned ECG representations and explainability
- **van de Leur RR, et al.** Improving explainability of deep neural network-based electrocardiogram
  interpretation using variational auto-encoders (FactorECG). *Eur Heart J Digit Health* 2022;3:390–404. ✅
  A β-VAE on 1.1 million ECGs gives interpretable factors with performance close to black-box DNNs.
- **ECG foundation models** (e.g. ECG-FM; HuBERT-ECG; a 2026 self-supervised model with genetic analysis,
  reported in *Nat Commun* per a news summary). ◐

**Positioning:** prior work treats VAE factors as interpretable *predictors*. Our latent audit asks how much
of each factor is *caused by MI* after adjustment. MI explains < 0.2% of each latent's variance.

### 2.4 Risk factors and selection in hospital data (Layer 1 context)
- **Yusuf S, et al.** Effect of potentially modifiable risk factors associated with myocardial infarction in
  52 countries (INTERHEART). *Lancet* 2004;364(9438):937–952. doi:10.1016/S0140-6736(04)17018-9. ✅
  Case-control ORs: diabetes 2.37, hypertension 1.91; with smoking and ApoB/ApoA1, nine factors account for
  about 90% of population attributable risk.
- **Griffith GJ, et al.** Collider bias undermines our understanding of COVID-19 disease risk and severity.
  *Nat Commun* 2020 (published 12 Nov). doi:10.1038/s41467-020-19478-2. ✅ Restricting to hospitalized patients
  distorts risk-factor associations.

**Positioning:** our Layer 1 associations (diabetes RR 1.54–1.67, hypertension 1.44) point the same way as
INTERHEART but are smaller. That is expected among hospitalized adults, where everyone is selected on
illness (collider bias). It matches the look-back analysis (larger RRs once recording artefacts are removed).

### 2.5 ECG pattern of non-ST-elevation ischemia (Layer 2 interpretation)
- **ST elevation in aVR with diffuse ST depression** (review, PMC12966643; Cerit, *Cardiovasc J Afr*). ◐ The
  pattern indicates subendocardial ischemia / supply–demand mismatch and predicts severe multivessel or left
  main disease in NSTE-ACS, rather than acute left main occlusion. It is consistent with our lateral ST
  depression plus aVR elevation in a 74%-NSTEMI cohort.

### 2.6 CKD and MI subtypes (Phase 7 CKD finding)
- **ISCHEMIA-CKD** analyses (type 1 and type 2 MI in advanced CKD); **albuminuria/eGFR and NSTEMI risk**
  (*JAHA* 2018); **Danish KidDiCo** cohort. ◐ CKD relates to NSTEMI and to supply–demand MI. No study found
  separates type 2 MI risk in CKD as cleanly as our principal-diagnosis analysis.

## 3. Novelty assessment

The targeted search found **no study that estimates confounder-adjusted (causal) effects of MI on ECG
features under an explicit causal graph**, and none that audits learned ECG latents for MI-caused variance.
Adjacent work does exist:
- causal adjustment *inside classifiers* (Causal ECGNet);
- counterfactual *explanations* of MI classifiers (VCCE and others);
- explainable VAE factors as predictors (FactorECG);
- demographic-shortcut checks by ablation.

The manuscript's statement "to our knowledge, a causal audit of ECG features and latent embeddings with
respect to MI" is supported by this search. It should stay hedged until the systematic search (Section 5)
is complete.

## 4. Changes made to MANUSCRIPT.md
- Verified references given full details; † replaced by ✅ / ◐ status.
- Introduction: related-work paragraph and positioning; ECG-AI citations filled in.
- Discussion: INTERHEART comparison and collider-bias explanation of the smaller Layer 1 RRs; aVR/ST-depression
  literature for the Layer 2 pattern.
- Data section: MIMIC-IV-ECG-Ext-ICD cited as the source of the legacy label.

## 5. Still needed before submission

1. Systematic search with a PRISMA-style log. Suggested strings:
   - PubMed: `("electrocardiogra*"[tiab] OR ECG[tiab]) AND ("myocardial infarction"[tiab]) AND ("causal"[tiab]
     OR "confound*"[tiab] OR "counterfactual"[tiab] OR "structural causal model"[tiab] OR "doubly robust"[tiab])`
   - PubMed: `("variational autoencoder"[tiab] OR "latent"[tiab] OR "representation learning"[tiab]) AND
     ("electrocardiogra*"[tiab]) AND ("myocardial infarction"[tiab])`
   - arXiv / medRxiv equivalents.
2. Complete the ◐ entries (authors, volumes, pages): Causal ECGNet; Tanyel 2025 volume; aVR review; CKD
   studies; MIMIC-IV v2.2 dataset DOI.
3. A clinical co-author should check the ECG-pattern interpretation against cardiology guidelines.

## Sources consulted
- MIMIC-IV: https://pubmed.ncbi.nlm.nih.gov/36596836/ · https://www.nature.com/articles/s41597-022-01899-x.pdf
- MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg/1.0/
- MIMIC-IV-ECG-Ext-ICD: https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.0 · https://arxiv.org/html/2312.11050v2
- ECGdeli: https://github.com/KIT-IBT/ECGdeli · https://dblp1.uni-trier.de/rec/journals/softx/PiliaNLBDL21.html
- Fourth UDMI: https://pubmed.ncbi.nlm.nih.gov/30165617/
- Bang & Robins: https://pubmed.ncbi.nlm.nih.gov/16401269/
- Chernozhukov: https://econpapers.repec.org/RePEc:wly:emjrnl:v:21:y:2018:i:1:p:c1-c68
- Crump: https://econpapers.repec.org/RePEc:oup:biomet:v:96:y:2009:i:1:p:187-199
- E-value: https://dash.harvard.edu/bitstream/handle/1/36874927/EValue_FinalSubmission.pdf
- VanderWeele 2017: https://pmc.ncbi.nlm.nih.gov/articles/PMC5617805
- Negative controls: https://pmc.ncbi.nlm.nih.gov/articles/PMC3053408
- Bareinboim & Tian: https://ojs.aaai.org/index.php/AAAI/article/view/9679
- β-VAE: https://mlanthology.org/iclr/2017/higgins2017iclr-beta
- PTB-XL benchmarks: https://arxiv.org/pdf/2004.13701
- Al-Zaiti 2023: https://www.nature.com/articles/s41591-023-02396-3.pdf
- Causal ECGNet: https://pubmed.ncbi.nlm.nih.gov/40458653/
- VCCE: https://arxiv.org/html/2312.08304v3 · Prototype counterfactuals: https://arxiv.org/abs/2510.19514v1
- FactorECG: https://pmc.ncbi.nlm.nih.gov/articles/PMC9707974
- INTERHEART: https://pubmed.ncbi.nlm.nih.gov/15364185/
- Collider bias: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7665028/
- aVR review: https://pmc.ncbi.nlm.nih.gov/articles/PMC12966643/ · https://litfl.com/st-elevation-in-avr
- CKD and MI: https://pmc.ncbi.nlm.nih.gov/articles/PMC10865178 · https://pmc.ncbi.nlm.nih.gov/articles/PMC6474966
- Demographic shortcut checks: https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001058 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9666471/
