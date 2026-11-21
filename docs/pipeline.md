# Pipeline v4.0 — Structural Causal Model for Myocardial Infarction Risk

> **Last Updated:** 2026-10-09
> **Scope:** Actual implemented workflow (Phases A–H), optimized future plan (Phases I–N), and MCP dashboard architecture (Phase O).
> **Datasets:** MIMIC-IV v2.2, MIMIC-IV-ECG v1.0, MIMIC-IV-ECG-Ext-ICD v1.0.1
> **Goal:** Build a validated causal reasoning model for Myocardial Infarction (MI) to predict risk, estimate heterogeneous treatment effects (CATE), and generate patient-level counterfactuals.

---

## Status Summary

| Phase | Status | Key Artifact |
|---|---|---|
| A — Infrastructure | ✅ Complete | `requirements.txt`, conda env |
| B — Ingestion & Cohort Extraction | ✅ Complete | `data/interim/cohort_labels.csv` |
| C — Labeling & Temporal Refinement | ✅ Complete | `data/processed/final_ecg_cohorts_troponin.csv` |
| D — ECG Feature & Latent Engineering | ✅ Complete | `extracted_ecg_features_with_study_id.csv`, `ecg_signals.h5`, VAE checkpoint |
| E — Clinical Feature Extraction | ✅ Complete | `features_comorbidities.csv`, `features_treatments.csv`, `features_labs_fixed.csv`, `features_vitals_fixed.csv` |
| F — Master Dataset Construction | ✅ Complete | `data/modeling/master/{train,validation,test}.parquet` |
| G — Environment/IRM | ⏭️ Deferred | Not implemented (see deviations) |
| H — Baseline Predictive Models | ✅ Complete | `models/baselines/results.json`, calibrated models |
| I — DAG & SCM Specification | 🔜 Planned | — |
| J — Causal Effect Estimation | 🔜 Planned | — |
| K — Counterfactuals & Validation | 🔜 Planned | — |
| L — Longitudinal Extension | 🔨 Partial | `data/longitudinal/*.parquet` |
| M — Robustness & Sensitivity | 🔜 Planned | — |
| N — Reporting & Publication | 🔜 Planned | — |
| O — MCP Dashboard Architecture | 🔜 Planned | — |

---

## Deviations from Original Plan (v3.1)

This section explicitly documents where the implementation diverged from the original `pipeline.md` (v3.1), and why.

### D1. No DuckDB as Persistent Local Database (Phase B)
- **Original Plan:** Set up DuckDB as a persistent local analytical database with indices.
- **What Happened:** DuckDB is used as an *in-memory* analytical engine within individual scripts (`scripts/extract_cohort.py`, `scripts/longitudinal_phase1.py`). No persistent `.duckdb` file is created. Raw MIMIC-IV CSVs and Parquets are queried directly via `read_csv_auto()`.
- **Why:** In-memory querying was sufficient for the cohort sizes involved and avoids managing a separate database lifecycle. DuckDB's ability to query CSV/Parquet directly eliminated the need for an ETL step.

### D2. No DVC (Data Version Control) Used (Phase A)
- **Original Plan:** Use DVC for data artifact versioning.
- **What Happened:** Data artifacts are tracked via Git (`.gitignore` for large files) and manual directory conventions (`data/raw/`, `data/interim/`, `data/processed/`, `data/modeling/`).
- **Why:** The single-researcher workflow made DVC overhead unnecessary. File sizes (90 GB HDF5) exceed typical DVC remote storage budgets.

### D3. ECG Feature Extraction via MATLAB/ECGdeli, Not Python (Phase D)
- **Original Plan:** Use Python libraries (neurokit2, scipy) for fiducial detection and feature extraction. Validate against PTB-XL+ ground-truth fiducials.
- **What Happened:** ECG morphological features were extracted using **MATLAB with ECGdeli v1.1** (`extract_ecg_features.m`), a dedicated ECG delineation toolbox. PTB-XL validation was not performed.
- **Why:** ECGdeli provides production-grade, peer-reviewed cardiac waveform delineation with full 12-lead multi-channel annotation (`Annotate_ECG_Multi`). It handles noise mitigation (baseline removal, bandpass filtering, notch filtering, isoline correction) internally. The decision traded Python ecosystem consistency for delineation accuracy.

### D4. Labeling via ICD Codes + Troponin Dynamics, Not Troponin Thresholds Alone (Phase C)
- **Original Plan:** Define MI events purely via troponin threshold crossings (99th percentile URL) with sex/assay-stratified lookup tables.
- **What Happened:** The implemented labeling (`scripts/extract_cohort.py`) uses **ICD-10 diagnosis codes** from `MIMIC-IV-ECG-Ext-ICD` (`records_w_diag_icd10.csv`) as the primary MI indicator, then refines timing using a troponin delta-curve algorithm that identifies the first >20% rise/fall pattern (Fourth Universal Definition of MI). See `data/processed/Data_Engineering_Report.md`.
- **Why:** The ICD-based approach leverages clinician-adjudicated diagnoses rather than relying on programmatic threshold comparison across heterogeneous assay types. The troponin timing refinement then anchors the `index_mi_time` precisely, which is critical for the temporal labeling windows. This hybrid approach recovered ~3,000 pre-incident ECGs that the proxy method missed.

### D5. No Label Adjudication or Formal Power Analysis (Phase C.7–C.8)
- **Original Plan:** 100-case clinician review, formal power analysis with CATE simulation, Go/No-Go gates.
- **What Happened:** Neither was performed. The cohort produced 7,604 MI_Acute and 5,954 MI_Pre-Incident ECGs, well above the 500 minimum. Formal adjudication was not feasible within the project scope.
- **Why:** Pragmatic decision given timeline constraints. The ICD-code-based labeling provides a reasonable clinical standard. The sample sizes are sufficiently powered for the planned analyses.

### D6. VAE Trained on ALL Labels, Not Excluding MI_Acute (Phase D.3)
- **Original Plan:** Train VAE only on Control_Symptomatic + MI_Pre-Incident ECGs; exclude MI_Acute_Presentation and MI_Post-Incident.
- **What Happened:** The VAE was trained on the same subject-level splits as the modeling data (train/val/test), which include Control_Symptomatic, MI_Acute_Presentation, and MI_Pre-Incident. Only MI_Post-Incident was excluded (by `build_modeling_data.py`).
- **Why:** The VAE is an unsupervised representation learner; it does not use labels during training. Including MI_Acute ECGs gives the VAE access to the full morphological variation in the data, producing better-generalizing representations. This is standard practice in representation learning for downstream supervised tasks.

### D7. No Fiducial Validation Against PTB-XL+ (Phase D.1)
- **Original Plan:** Validate feature extractor against PTB-XL+ ground-truth fiducials with MAE thresholds.
- **What Happened:** Not performed. PTB-XL and PTB-XL+ datasets are listed in the original plan but were not used.
- **Why:** The switch to ECGdeli (a validated delineation library) was considered a sufficient quality guarantee. ECGdeli has published validation benchmarks in the literature.

### D8. Phase G (IRM / Environment Robustness) Deferred
- **Original Plan:** Extract ECG machine models from WFDB headers, define environments, train IRM.
- **What Happened:** Not implemented. No `environment_label` column exists in any dataset.
- **Why:** The IRM approach requires sufficient data in each environment (≥500 per class). The project prioritized core pipeline completion and causal analysis feasibility before robustness experiments. IRM can be added as a sensitivity analysis in Phase M.

### D9. Labs and Vitals Dropped from Master Dataset Due to <1% Coverage
- **Original Plan:** Include labs (LDL, HDL, creatinine, glucose, BNP) and vitals (SBP, DBP, SpO2, HR, Temp) in the master dataset, with MICE imputation for missing values.
- **What Happened:** Labs covered only 738/194,587 admissions (0.38%) and vitals only 1,457/194,587 (0.75%) after strict temporal filtering. Both were **excluded** from the master parquet files.
- **Why:** MICE imputation with >99% missingness produces essentially fabricated data. The strict pre-MI temporal windowing (measurements between `window_before_start` and `anchor_time`) eliminated most lab/vital values, which are predominantly recorded during or after the admission event. The clinical features that survived with full coverage (comorbidities, treatments, demographics) provide adequate confounding adjustment.

### D10. No Control_Asymptomatic Label Defined
- **Original Plan:** Define a secondary control group of patients with no troponin measurements.
- **What Happened:** The cohort exclusively uses `MIMIC-IV-ECG-Ext-ICD` which restricts to ED/hospital ECGs with diagnoses. All controls are `Control_Symptomatic` (troponin measured but below URL).
- **Why:** This actually strengthens the causal framing by comparing like-with-like: "among symptomatic patients evaluated for possible MI, what distinguishes those who are having an MI?"

### D11. Baseline Models Use XGBoost + LogReg, Not MLP or End-to-End CNN
- **Original Plan:** Three baselines: XGBoost (tabular), MLP (latent space + clinical), End-to-End CNN (raw waveform + clinical).
- **What Happened:** Four baselines: Logistic Regression (clinical only), XGBoost (ECG features + clinical), XGBoost (embeddings + clinical), XGBoost (all features). No MLP or CNN.
- **Why:** XGBoost with appropriate class weighting and early stopping provides strong, well-calibrated baselines that are interpretable and fast. The logistic regression baseline establishes the clinical-only performance floor. MLP and CNN baselines can be added in a future iteration but are not required for the causal analysis pipeline.

---

## PHASE A — INFRASTRUCTURE & ENVIRONMENT

**Status: ✅ Complete**

### A.1 Compliance & Repository
- All team members hold valid PhysioNet credentials for MIMIC-IV data access
- Git repository initialized at `SCM/` for code version control
- `.gitignore` configured to exclude data files (`.csv`, `.parquet`, `.h5`, `.pkl`, `.pt`)

### A.2 Environment
- **Python environment:** conda (`cuda_env`) with `requirements.txt`
- **MATLAB:** R2024a with Parallel Computing Toolbox (for ECGdeli extraction)
- **Key Python libraries:**
  - Data: `pandas`, `numpy`, `duckdb`, `pyarrow`, `h5py`
  - Signal/ECG: `wfdb` (Python bindings for WFDB file I/O)
  - ML/DL: `scikit-learn`, `xgboost`, `pytorch` (2.x with CUDA)
  - Visualization: `matplotlib`, `seaborn`
  - Planned (Phase I+): `dowhy`, `econml`

### A.3 Compute Resources
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM)
  - Sufficient for VAE training at batch_size=128 with mixed precision
  - VAE training completed in ~8.5 hours for 129 epochs on 262K ECGs
- **Storage:** Local SSD, ~1.5 TB active working set
  - `ecg_signals.h5`: 90.6 GB (pre-converted normalized waveforms)
  - Raw MIMIC-IV: ~100 GB
  - Master dataset (Parquet): ~556 MB across train/val/test

**Relevant files:**
- [`requirements.txt`](file:///c:/Users/Acer/Desktop/Capstone/SCM/requirements.txt)

---

## PHASE B — DATA INGESTION & COHORT EXTRACTION

**Status: ✅ Complete**
**Script:** [`scripts/extract_cohort.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/scripts/extract_cohort.py)

### B.1 Data Sources
The cohort extraction uses an in-memory DuckDB engine to query the following raw MIMIC-IV tables:

| Source | Table | Purpose |
|---|---|---|
| MIMIC-IV-ECG-Ext-ICD v1.0.1 | `records_w_diag_icd10.csv` | ECG metadata with ICD-10 diagnoses |
| MIMIC-IV v2.2 | `hosp/admissions.csv` | Admission times, demographics |
| MIMIC-IV v2.2 | `hosp/prescriptions.csv` | Medication orders (statins, beta-blockers, ACEi, aspirin) |
| MIMIC-IV v2.2 | `hosp/labevents.csv` | Laboratory measurements |
| MIMIC-IV v2.2 | `icu/chartevents.csv` | Vital signs (30 GB) |
| MIMIC-IV v2.2 | `icu/inputevents.csv` | Vasopressor administration |
| MIMIC-IV v2.2 | `icu/procedureevents.csv` | Mechanical ventilation |
| MIMIC-IV v2.2 | `hosp/omr.csv` | Outpatient blood pressure |
| MIMIC-IV v2.2 | `hosp/diagnoses_icd.csv` | ICD diagnosis codes (for comorbidities) |

### B.2 Base Cohort Construction
The script builds the cohort in 6 steps:

1. **ECG Cohort:** Selects ECGs taken in ED or hospital (`ecg_taken_in_ed_or_hosp = true`) from `records_w_diag_icd10.csv`, grouped by `hadm_id × subject_id`. Only ECGs with at least one ICD diagnosis are included.

2. **Temporal Boundaries:** For each admission, computes:
   - `anchor_time` = `COALESCE(edregtime, admittime)` (earliest clinical contact)
   - `window_before_start` = max(previous discharge, anchor_time − 7 days)
   - `window_after_end` = min(next admission, dischtime + 7 days)
   - These dynamic 7-day windows prevent data from adjacent admissions from leaking.

3. **MI Classification:** Parses ICD-10 codes from `all_diag_all` using Python's `ast.literal_eval`:
   - **Type 1 MI:** I210–I214, I219, I21
   - **Type 2 MI:** I21A1
   - **Subsequent MI:** I22*
   - **MI Complication:** I23*
   - Binary flag `is_MI` = any MI type present

4. **Treatment Extraction:** Temporal medication flags via SQL joins on prescriptions:
   - `statin_before/during/after` (drug LIKE '%vastatin%')
   - `beta_blocker_before/during` (LIKE '%olol%' OR '%carvedilol%')
   - `acei_before/during` (LIKE '%pril%')
   - `aspirin_before/during`
   - `vasopressor_during` (from ICU inputevents)
   - `mech_vent_during` (from ICU procedureevents)

5. **Lab Extraction:** LDL, HDL, troponin, creatinine, BUN, glucose, potassium, BNP — both `_before` and `_during` windows.

6. **ECG Mapping:** Creates `ecg_mapping.csv` linking `hadm_id ↔ subject_id ↔ study_id ↔ waveform_path ↔ ecg_time`.

**Output artifacts** (in `data/interim/`):
- `cohort_labels.csv` — 194,587 unique admissions with MI classification flags
- `ecg_mapping.csv` — 800K+ ECG-to-patient-to-admission linkages
- `features_treatments.csv` — per-admission treatment flags
- `features_labs.csv` — per-admission lab values
- `features_vitals.csv` — per-admission vital signs

---

## PHASE C — LABELING & TEMPORAL REFINEMENT

**Status: ✅ Complete**

### C.1 Troponin-Based MI Timing
The original cohort used `anchor_time` (ED registration) as a proxy for MI onset. This was substantially inaccurate for in-hospital MIs. A dedicated troponin analysis script refined the timing:

- Queried raw troponin lab results (itemids 51002, 51003, 52642) from `labevents.csv`
- Applied the **Fourth Universal Definition of MI**: Troponin T > 0.014 ng/mL with a >20% rising/falling delta pattern
- Established `index_mi_time` as the precise moment of troponin spike

**Impact (from [`data/processed/Data_Engineering_Report.md`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/processed/Data_Engineering_Report.md)):**
- Recovered ~3,000 pre-incident ECGs that the proxy method misclassified
- The MI_Pre-Incident cohort grew from ~4 to **5,957 ECGs**

### C.2 Temporal ECG Labels
Each ECG is labeled relative to `index_mi_time` using `time_diff_hours = ecg_time − final_index_time`:

| Label | Definition | Count | Modeling Use |
|---|---|---|---|
| `Control_Symptomatic` | Troponin measured, all values below URL | 363,545 | Negative class |
| `MI_Acute_Presentation` | ECG within −6h to +2h of `index_mi_time` | 7,611 | Positive class |
| `MI_Pre-Incident` | ECG >6h before `index_mi_time` | 5,957 | Negative class (pre-event baseline) |
| `MI_Post-Incident` | ECG >2h after `index_mi_time` | 31,572 | **Excluded** (confounded by acute treatment) |

### C.3 Final Cohort File
**Artifact:** [`data/processed/final_ecg_cohorts_troponin.csv`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/processed/final_ecg_cohorts_troponin.csv) — 408,685 rows × 11 columns.

Key columns: `study_id`, `subject_id`, `hadm_id`, `waveform_path`, `ecg_time`, `is_MI`, `index_mi_time`, `final_index_time`, `time_diff_hours`, `Label_Troponin`.

---

## PHASE D — ECG FEATURE & LATENT SPACE ENGINEERING

**Status: ✅ Complete**

### D.1 Morphological Feature Extraction (MATLAB / ECGdeli)

**Script:** [`extract_ecg_features.m`](file:///c:/Users/Acer/Desktop/Capstone/SCM/extract_ecg_features.m)

The pipeline uses ECGdeli v1.1 (a peer-reviewed MATLAB toolbox) for full morphological delineation:

1. **Signal Loading:** WFDB `rdsamp()` via MATLAB bindings (with a cd-based Windows path workaround)
2. **Noise Mitigation Chain:**
   - Baseline removal (`ECG_Baseline_Removal`, 0.5 Hz high-pass)
   - Bandpass filter (`ECG_High_Low_Filter`, 1–40 Hz)
   - 50 Hz notch filter (`Notch_Filter`)
   - Isoline correction (`Isoline_Correction`)
3. **Full 12-Lead Delineation:** `Annotate_ECG_Multi` → Fiducial Point Table (FPT) per lead per beat
4. **Feature Aggregation:** From FPT and amplitude matrices:

| Feature Group | Features per Lead | Aggregation | Total |
|---|---|---|---|
| Global Timing | QRS_Dur_Global, RR_Mean_Global | mean, iqr, count | 6 |
| S-Wave Amplitudes | S_Amp_{V1,V2,V3} | mean, iqr, count | 9 |
| Q/R/T Amplitudes | Q_Amp, R_Amp, T_Amp × 12 leads | mean, iqr, count | 108 |
| ST Elevation | ST_Elev × 12 leads (J+60ms − QRS onset) | mean, iqr, count | 36 |
| **Total raw features** | | | **159** |

5. **Parallelization:** `parfor` across 4 workers, batch_size=1000 with auto-resume state detection.
6. **Feature Target List:** [`docs/final_features_to_extract.txt`](file:///c:/Users/Acer/Desktop/Capstone/SCM/docs/final_features_to_extract.txt) — 160 features defined.

**Output:** `extracted_ecg_features.csv` — 407,504 rows × 162 columns (SubjectID + 159 features + ecg_time + study_id)

### D.2 study_id Recovery (Greedy Sequence Alignment)

**Script:** [`src/data/recover_ecg_mapping.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/data/recover_ecg_mapping.py)

The MATLAB output lacked `study_id` (only contained `SubjectID`). Because MATLAB processed files in the same order as `ecg_mapping.csv`, a greedy O(n) pointer walk matched each feature row to its `study_id` and `ecg_time`:

```python
j = 0  # pointer into mapping
for each feature row i:
    advance j until mapping_subjects[j] == feature_subjects[i]
    assign mapping[j].study_id → features[i].study_id
```

**Output:** [`data/processed/extracted_ecg_features_with_study_id.csv`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/processed/extracted_ecg_features_with_study_id.csv) — 407,504 rows, 834 MB.

### D.3 HDF5 Waveform Conversion

**Script:** [`src/data/convert_to_hdf5.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/data/convert_to_hdf5.py)

Converted all modeling ECGs (MI_Post-Incident excluded) from individual WFDB files to a single HDF5 file for fast random-access during VAE training:

- Input: Raw WFDB `.dat/.hea` files via `wfdb.rdrecord()`
- Per-lead z-score normalization (zero-mean, unit-variance)
- Pad/truncate to 5000 samples (10s @ 500 Hz)
- NaN/Inf → 0.0

**Output:** [`data/processed/ecg_signals.h5`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/processed/ecg_signals.h5) — 90.6 GB
- `signals`: `(N, 12, 5000)` float32
- `study_ids`: `(N,)` int64

### D.4 β-VAE Architecture & Training

**Architecture:** [`src/models/vae_conv1d.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/models/vae_conv1d.py) — `Conv1DVAE`

```
Input: (batch, 12, 5000)
  ↓ Conv1D(12→32, k=15, s=2, p=7) + BN + LeakyReLU(0.2)     → (batch, 32, 2500)
  ↓ Conv1D(32→64, k=10, s=2, p=4) + BN + LeakyReLU(0.2)     → (batch, 64, 1250)
  ↓ Conv1D(64→128, k=5, s=2, p=2) + BN + LeakyReLU(0.2)     → (batch, 128, 625)
  ↓ Flatten                                                    → (batch, 80000)
  ↓ Linear(80000→512) + LeakyReLU + Dropout(0.2)
  ↓ Linear(512→256) + LeakyReLU
  ↓ fc_mu: Linear(256→64)        → μ
  ↓ fc_logvar: Linear(256→64)    → log(σ²), clamped to [-10, 10]
  ↓ Reparameterization: z = μ + σ·ε
  ↓ Linear(64→256) + LeakyReLU
  ↓ Linear(256→512) + LeakyReLU
  ↓ Linear(512→80000) + LeakyReLU
  ↓ Unflatten                                                  → (batch, 128, 625)
  ↓ ConvTranspose1D(128→64, k=5, s=2, p=2, op=1) + BN + LeakyReLU
  ↓ ConvTranspose1D(64→32, k=10, s=2, p=4, op=0) + BN + LeakyReLU
  ↓ ConvTranspose1D(32→12, k=15, s=2, p=7, op=1)             → (batch, 12, 5000)
Output: x_reconstructed
```

**Key design differences from original plan:**
- LeakyReLU(0.2) instead of ReLU (prevents dead neurons)
- Intermediate 512-dim FC layer (more gradual compression: 80000→512→256 vs. 80000→256)
- Dropout(0.2) in encoder FC
- Free bits constraint (`free_bits=2.0`) to prevent posterior collapse
- Reconstruction loss scaled by signal variance for KL/recon balance
- **82.4M trainable parameters**

**Training:** [`src/models/train_vae.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/models/train_vae.py)

| Hyperparameter | Value |
|---|---|
| `z_dim` | 64 |
| `β` (max) | 4.0 |
| `free_bits` | 2.0 |
| Batch size | 128 |
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| LR Schedule | ReduceLROnPlateau (patience=10, factor=0.5) |
| β Schedule | Cyclical annealing: 4 cycles × 40 epochs (0→β_max) |
| Early stopping | Patience=20 |
| Gradient clipping | max_norm=5.0 |
| Data loading | HDF5 mode (90.6 GB pre-converted) |

**Training Results** (from [`models/log.txt`](file:///c:/Users/Acer/Desktop/Capstone/SCM/models/log.txt)):
- Resumed from epoch 18 (prior runs: 4 earlier checkpoints in `models/checkpoints/`)
- **Best model: epoch 109, val_loss = 16,904.66**
- Early stopping triggered at epoch 129 (patience exhausted)
- Training rate: ~8.8 batches/sec on RTX 4050
- KL_raw remained well above 1.0 throughout (no posterior collapse)
- Final LR decayed to 1.25e-5

**Best checkpoint:** `models/checkpoints/vae_zdim64_beta4.0_20260821_030245/best_model.pt` (989 MB)

### D.5 Embedding Extraction

**Script:** [`src/models/extract_embeddings.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/models/extract_embeddings.py)

Runs the trained encoder on all ECGs in each split, extracting the latent mean vector μ (deterministic, not sampled):

**Output:** `data/modeling/ecg_embeddings/{train,validation,test}.csv`
- Columns: `study_id`, `SubjectID`, `z_ecg_1`, ..., `z_ecg_64`

### D.6 Embedding Validation

**Script:** [`src/models/validate_embeddings.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/models/validate_embeddings.py)

Four quality checks:
1. **Basic stats:** Shape, NaN/Inf, per-dimension mean/std ranges, collapsed dimension detection
2. **Dimension usage:** Variance per latent dim (active threshold = 0.01); saved plot `dimension_variance.png`
3. **Label separation:** t-SNE visualization on 5K subsampled train embeddings; saved `tsne_embeddings.png`
4. **Downstream discriminability:** Logistic regression (MI vs Control) on embeddings alone → AUROC reported

**Artifacts:** Plots saved in `models/checkpoints/vae_zdim64_beta4.0_20260821_030245/`

---

## PHASE E — CLINICAL FEATURE EXTRACTION

**Status: ✅ Complete**

### E.1 Comorbidities (Causal Confounders)

**Script:** [`src/features/extract_comorbidities.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/features/extract_comorbidities.py)

Extracts binary flags from ICD-9/ICD-10 codes in **prior admissions only** (temporal precedence enforced via `admittime < anchor_time`):

| Feature | ICD-9 Codes | ICD-10 Codes |
|---|---|---|
| `diabetes` | 250* | E08–E13* |
| `hypertension` | 401–405* | I10–I15* |
| `ckd` | 585* | N18* |
| `prior_mi` | 410*, 412* | I21*, I22* |

**Output:** [`data/processed/features_comorbidities.csv`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/processed/features_comorbidities.csv) — 194,587 rows (one per hadm_id)

### E.2 Treatments (Potential Interventions)

Extracted in Phase B by `extract_cohort.py` using temporal windows around admission:

| Feature | Window | Drug Pattern |
|---|---|---|
| `statin_before/during/after` | Pre/Intra/Post-admission | `%vastatin%` |
| `beta_blocker_before/during` | Pre/Intra | `%olol%`, `%carvedilol%` |
| `acei_before/during` | Pre/Intra | `%pril%` |
| `aspirin_before/during` | Pre/Intra | `%aspirin%` |
| `vasopressor_during` | Intra (ICU) | itemids 221906, 221289, 221662, 222315, 221653 |
| `mech_vent_during` | Intra (ICU) | itemids 225792, 225794, 224385 |

**Output:** [`data/processed/features_treatments.csv`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/processed/features_treatments.csv) — 194,587 rows

### E.3 Labs & Vitals (Dropped from Modeling)

**Script:** [`src/features/fix_temporal_features.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/features/fix_temporal_features.py)

Applied strict pre-MI temporal filtering (`window_before_start ≤ charttime ≤ anchor_time`):

| Feature File | Rows | Coverage | Status |
|---|---|---|---|
| `features_labs_fixed.csv` | 738 | 0.38% | ❌ Dropped from master |
| `features_vitals_fixed.csv` | 1,457 | 0.75% | ❌ Dropped from master |

**Rationale:** The strict temporal filtering correctly prevents leakage but produces unusably sparse data. These files are preserved in `data/processed/` and `data/modeling/clinical/` for reference but are not included in the master parquet files.

### E.4 Demographics

Extracted from MIMIC-IV `patients.csv` during master dataset construction (`build_modeling_data.py`):
- `anchor_age` — Age at anchor year (MIMIC-IV de-identification artifact)
- `gender` → encoded as `sex_male` (binary: 1=M, 0=F)

---

## PHASE F — MASTER DATASET CONSTRUCTION

**Status: ✅ Complete**

This phase is implemented across two scripts that perform complementary functions:

### F.1 Modeling Data Preparation (ECG Features + Labels + Splits)

**Script:** [`data/build_modeling_data.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/build_modeling_data.py)

This script rebuilds train/validation/test splits from scratch with proper label alignment:

1. **Join:** ECG features (`extracted_ecg_features_with_study_id.csv`) ↔ cohort labels (`final_ecg_cohorts_troponin.csv`) via `study_id`
2. **Exclude:** MI_Post-Incident rows (31,444 removed — confounded by acute treatment)
3. **Deduplicate:** On `SubjectID + feature columns` (55 exact duplicates removed)
4. **Subject-Level Stratified Split:** 70/15/15 (seed=42), stratified by majority label per subject
   - Zero subject overlap verified with assertions
5. **Feature Removal:** Regex patterns drop 64 redundant count/iqr columns:
   - `^ST_Elev_.*_iqr$` (12 ST elevation IQR features)
   - `^(?!QRS_Dur_Global_count$).+_count$` (all count features except QRS_Dur_Global_count)
   - Result: **95 retained ECG features** from 159 original
6. **Preprocessing Pipeline** (fit on train only, applied to all):
   - **Winsorization:** Per-group quantile clipping (0.5th/99.5th percentile), with physiological limits for RR_Mean_Global (200–2500 ms) and QRS_Dur_Global (40–500 ms)
   - **Transformation:** Yeo-Johnson for amplitude_mean and ST_elev groups; log1p for IQR group; none for global_mean and count
   - **Scaling:** StandardScaler for global_mean and count; RobustScaler for amplitude_mean, IQR, and ST_elev
7. **Add Demographics:** age, sex from MIMIC-IV `patients.csv`
8. **Export:** Separate CSV files for ECG features, labels, clinical tables, demographics, and preprocessing parameters
9. **Verification:** Automated checks for zero subject leakage, zero MI_Post-Incident contamination, row count alignment, and no NaN/Inf in final features

### F.2 Master Dataset Assembly (All Modalities Merged)

**Script:** [`src/features/build_master_dataset.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/features/build_master_dataset.py)

Merges all data sources into a single Parquet table per split:

```
Labels (backbone, via study_id)
  ↓ LEFT JOIN ecg_embeddings (via study_id)    → 64 z_ecg dimensions
  ↓ LEFT JOIN ecg_features (via study_id)      → 95 morphological features
  ↓ LEFT JOIN comorbidities (via hadm_id)      → diabetes, hypertension, ckd, prior_mi
  ↓ LEFT JOIN treatments (via hadm_id)         → 11 medication/intervention flags
  ↓ LEFT JOIN demographics (via SubjectID)     → anchor_age, sex_male
```

Missing indicators (`*_missing`) are automatically created for features with >10% missingness.

### F.3 Final Master Dataset Statistics

**Artifact:** `data/modeling/master/{train,validation,test}.parquet`

| Split | Rows | Subjects | Control_Symptomatic | MI_Acute | MI_Pre |
|---|---|---|---|---|---|
| train | 262,813 | 69,479 | 253,334 | 5,300 | 4,179 |
| validation | 56,319 | 14,888 | 54,302 | 1,145 | 872 |
| test | 56,798 | 14,889 | 54,736 | 1,159 | 903 |
| **Total** | **375,930** | **99,256** | **362,372** | **7,604** | **5,954** |

**MI prevalence (binary target):** 2.02% (severely imbalanced — addressed via `scale_pos_weight` in XGBoost and `class_weight='balanced'` in LogReg)

**Column groups in master parquet:**
- Identifiers: `study_id`, `SubjectID`, `hadm_id` (3)
- Labels/Time: `Label_Troponin`, `is_MI`, `ecg_time`, `index_mi_time`, `time_diff_hours` (5)
- ECG Embeddings: `z_ecg_1` ... `z_ecg_64` (64)
- ECG Morphological Features: 95 preprocessed features
- Clinical: `diabetes`, `hypertension`, `ckd`, `prior_mi`, 11 treatment flags, `anchor_age`, `sex_male` (17)

**Preprocessing parameters** (saved for inference reproducibility):
- [`data/modeling/parameters/winsorization_bounds.json`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/modeling/parameters/winsorization_bounds.json)
- [`data/modeling/parameters/transformation_params.pkl`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/modeling/parameters/transformation_params.pkl)
- [`data/modeling/parameters/scaler.pkl`](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/modeling/parameters/scaler.pkl)

---

## PHASE H — BASELINE PREDICTIVE MODELS

**Status: ✅ Complete**

### H.1 Model Configurations

**Script:** [`src/models/train_baselines.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/models/train_baselines.py)

Four baseline models, all trained as binary classifiers (`MI_Acute_Presentation` vs rest):

| Model | Features | Count | Type |
|---|---|---|---|
| **Logistic Regression (Clinical)** | 17 clinical features | 17 | `LogisticRegression(class_weight='balanced')` |
| **XGBoost (ECG Features + Clinical)** | 95 ECG morph + 17 clinical | 112 | `XGBClassifier(scale_pos_weight=48.6)` |
| **XGBoost (Embeddings + Clinical)** | 64 z_ecg + 17 clinical | 81 | Same XGBoost config |
| **XGBoost (All Features)** | 95 ECG + 64 z_ecg + 17 clinical | 176 | Same XGBoost config |

**XGBoost Configuration:**
```python
XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=n_neg/n_pos,  # ~48.6
    eval_metric='auc', early_stopping_rounds=50,
    tree_method='hist', n_jobs=-1, random_state=42
)
```

### H.2 Test Set Results (Pre-Calibration)

**Artifact:** [`models/baselines/results.json`](file:///c:/Users/Acer/Desktop/Capstone/SCM/models/baselines/results.json) (timestamp: 2026-08-22)

| Model | AUROC | AUPRC | Brier | ECE |
|---|---|---|---|---|
| LogReg (Clinical) | 0.8164 | 0.0650 | 0.1852 | 0.3214 |
| XGBoost (ECG+Clinical) | **0.8667** | **0.1747** | 0.1275 | 0.2480 |
| XGBoost (Emb+Clinical) | 0.8412 | 0.1043 | 0.1467 | 0.2734 |
| XGBoost (All Features) | 0.8667 | 0.1684 | 0.1261 | 0.2470 |

**Key findings:**
- ECG morphological features provide the most discriminative signal (AUROC 0.867 vs 0.816 clinical-only)
- VAE embeddings alone underperform explicit morphological features (0.841 vs 0.867) — the ECGdeli features capture clinically relevant patterns more directly
- Combining embeddings with morphological features yields no improvement (0.867 vs 0.867) — the morphological features already capture the discriminative information
- All models achieve ~90% sensitivity at specificity ~60-64%
- AUPRC is low across the board (0.065–0.175) due to 2% prevalence — this is expected

### H.3 Subgroup Analysis (Test Set)

| Model | Age <50 | Age 50-70 | Age >70 | Male | Female | Diabetes | No Diabetes | Prior MI | No Prior MI |
|---|---|---|---|---|---|---|---|---|---|
| LogReg | 0.900 | 0.819 | 0.764 | 0.796 | 0.833 | 0.775 | 0.828 | 0.694 | 0.830 |
| XGB (ECG+Clin) | **0.934** | **0.869** | **0.825** | **0.856** | **0.875** | **0.826** | **0.879** | **0.753** | **0.879** |
| XGB (Emb+Clin) | 0.914 | 0.848 | 0.790 | 0.827 | 0.853 | 0.789 | 0.856 | 0.708 | 0.855 |
| XGB (All) | 0.933 | 0.868 | 0.826 | 0.856 | 0.875 | 0.827 | 0.879 | 0.755 | 0.878 |

**Notable disparities:** Prior MI patients show substantially lower AUROC (0.694–0.755) across all models, likely because their baseline ECG abnormalities (Q-waves, ST changes from prior infarction) overlap with acute MI patterns. This subgroup warrants special attention in the causal analysis.

### H.4 Post-Hoc Calibration

**Script:** [`src/models/calibrate_baselines.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/models/calibrate_baselines.py)

Isotonic regression calibration fitted on validation set, evaluated on test set:

**Artifact:** [`models/baselines/calibration_results.json`](file:///c:/Users/Acer/Desktop/Capstone/SCM/models/baselines/calibration_results.json)

| Model | Brier (Before) | Brier (After) | ECE (Before) | ECE (After) |
|---|---|---|---|---|
| LogReg | 0.185 | **0.019** | 0.321 | **0.0003** |
| XGB (ECG+Clin) | 0.128 | **0.018** | 0.248 | **0.0014** |
| XGB (Emb+Clin) | 0.147 | **0.019** | 0.273 | **0.0004** |
| XGB (All) | 0.126 | **0.018** | 0.247 | **0.0011** |

Calibration reduced ECE by 2-3 orders of magnitude. AUROC is preserved (discrimination unchanged by calibration, as expected). Calibrated models saved as `*_calibrated.pkl`.

### H.5 Saved Artifacts

All in `models/baselines/`:
- `results.json` — Complete metrics, subgroup analysis
- `calibration_results.json` — Pre/post calibration comparison
- `model_comparison.png` — ROC + PR curves
- `calibration_plots.png` — Reliability diagrams
- `calibration_before_after.png` — Pre/post calibration comparison
- `subgroup_fairness.png` — Grouped bar charts by age/sex/diabetes/prior_mi
- `feature_importance.png` — Top 30 features (gain) from best XGBoost
- `*.pkl` — Serialized models (raw and calibrated)

### H.6 Feature Importance Analysis

**Notebook:** [`notebooks/feature_importance_analysis.ipynb`](file:///c:/Users/Acer/Desktop/Capstone/SCM/notebooks/feature_importance_analysis.ipynb)

**Reports:** `reports/feature_importances_*.csv` — Three analyses:
1. Core 177 features only
2. Full 531 features (with all variants)
3. Aggregated family-level importance (core + IQR + count)

---

## PHASE L — LONGITUDINAL DATA FOUNDATION (Partial)

**Status: 🔨 Partially Complete**

### L.1 Phase 1: Admission Chronology & Demographics

**Script:** [`scripts/longitudinal_phase1.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/scripts/longitudinal_phase1.py)

Built the temporal backbone for longitudinal analysis:

| Artifact | Contents |
|---|---|
| `admissions_chronology.parquet` | Complete admission history with sequence numbers, linkage to previous/next admissions |
| `patients_demographics.parquet` | Patient demographics with derived ethnicity groups |
| `diagnoses_all.parquet` + `diagnoses_mi_codes.parquet` | Full diagnosis history + MI-specific codes |
| `mi_admissions.parquet` | First/subsequent MI episodes per patient |
| `observation_end.parquet` | Death/censoring times |
| `phase1_stats.json` | Summary statistics |

### L.2 Phase 2: Clinical Factor Coverage Catalogues

**Script:** [`scripts/longitudinal_phase2.py`](file:///c:/Users/Acer/Desktop/Capstone/SCM/scripts/longitudinal_phase2.py)

Catalogued coverage rates for prioritized clinical factors:

| Artifact | Contents |
|---|---|
| `d_labitems.parquet` | Lab item dictionary |
| `lab_coverage_catalogue.parquet` | Coverage rates for lipids, glycemia, renal, cardiac biomarkers, inflammatory markers |
| `omr_coverage_catalogue.parquet` | Outpatient measurement coverage (BMI, BP, smoking) |
| `vitals_coverage_catalogue.parquet` | ICU vitals coverage |
| `phase2_stats.json` | Detailed coverage statistics |

**Purpose:** These catalogues inform which clinical factors have sufficient longitudinal coverage to include in the causal DAG. They will guide variable selection for Phase I.

---

## PHASE I — DAG DESIGN & SCM SPECIFICATION (Planned)

**Status: 🔜 Planned**

### I.1 Causal DAG Design

Based on the available data and clinical domain knowledge, the proposed DAG uses the following nodes:

**Observed confounders (W):**
- `anchor_age`, `sex_male` — demographics
- `diabetes`, `hypertension`, `ckd`, `prior_mi` — comorbidities from prior admissions

**Treatment (T):**
- Primary: `statin_before` (binary, precisely measured, no temporal leakage)
- Secondary analyses: `beta_blocker_before`, `acei_before`, `aspirin_before`

**Mediator:**
- `Z_ecg` — 64-dimensional VAE latent representation (captures ECG physiology)
- Alternatively, selected morphological features (ST_Elev, R_Amp, QRS_Dur_Global)

**Outcome (Y):**
- `MI_Acute_Presentation` (binary)

**Decision: Statin as Primary Treatment Variable (Option B from original plan)**

Rationale:
1. Binary, precisely measured — no measurement error
2. Labs (LDL/HDL) have <1% coverage and cannot be used
3. Clinically actionable — physicians prescribe statins, not "set LDL to X"
4. 194,587 admissions have treatment data (full coverage)

**Proposed DAG edges:**

```
Age → Statin, Z_ecg, MI_Acute
Sex → Statin, Z_ecg, MI_Acute
Diabetes → Statin, Z_ecg, MI_Acute
Hypertension → Statin, Z_ecg, MI_Acute
CKD → MI_Acute
Prior_MI → Statin, Z_ecg, MI_Acute
Statin → Z_ecg, MI_Acute
Z_ecg → MI_Acute
```

### I.2 Implementation Plan

1. **DAG specification** using `dowhy.CausalModel` with DOT graph string
2. **Identify estimand** via backdoor criterion
3. **Adjustment set verification:** `{Age, Sex, Diabetes, Hypertension, CKD, Prior_MI}`
4. **Positivity check:** Verify treatment overlap (statin=1 vs statin=0) in each confounder stratum
5. **Save:** `dag_v1.png`, `causal_estimands.md`

### I.3 Addressing LDL Measurement Error

Given that labs have <1% coverage, **LDL is not available as a continuous variable**. The causal analysis will:
1. Use `statin_before` as the primary treatment (no measurement error)
2. If longitudinal LDL data (from Phase L catalogues) shows sufficient coverage in specific subgroups, conduct a restricted sensitivity analysis
3. Document the unmeasured confounding concern (statin effects beyond LDL lowering — pleiotropic effects) and apply bounds analysis (Manski bounds or E-values)

---

## PHASE J — CAUSAL EFFECT ESTIMATION (Planned)

### J.1 Average Treatment Effect (ATE)

**Method:** Double Machine Learning (DML) via `econml.dml.LinearDML`

```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestClassifier

T = df['statin_before']
Y = (df['Label_Troponin'] == 'MI_Acute_Presentation').astype(int)
W = df[['anchor_age', 'sex_male', 'diabetes', 'hypertension', 'ckd', 'prior_mi']]

dml = LinearDML(
    model_y=RandomForestClassifier(n_estimators=200, min_samples_leaf=10),
    model_t=RandomForestClassifier(n_estimators=200, min_samples_leaf=10),
    discrete_treatment=True
)
dml.fit(Y, T, X=None, W=W)
ate = dml.ate()
ate_ci = dml.ate_interval(alpha=0.05)
```

**Design considerations:**
- Use train split for fitting, test split for final ATE reporting
- Cross-fitting (built into DML) prevents overfitting of nuisance models
- Report ATE with 95% CI and interpret as risk difference

### J.2 Conditional Average Treatment Effect (CATE)

**Method:** `econml.dml.CausalForestDML`

Heterogeneity features (X in CausalForest):
- Demographics: `anchor_age`, `sex_male`
- Comorbidities: `diabetes`, `hypertension`, `ckd`, `prior_mi`
- ECG embeddings: `z_ecg_1` ... `z_ecg_64` (to discover ECG-based effect modifiers)

**Outputs:**
- Per-patient CATE estimates with confidence intervals
- CATE × Age scatter plots
- CATE distribution by diabetes/prior_mi/sex subgroups
- High-benefit patient profiling (CATE < −0.10)

### J.3 Mediation Analysis (ECG Pathway)

To decompose the statin effect into direct and ECG-mediated components:
- **Total Effect (TE):** statin → MI (adjusting for W)
- **Natural Direct Effect (NDE):** statin → MI holding Z_ecg at its natural value
- **Natural Indirect Effect (NIE):** statin → Z_ecg → MI

Use `dowhy` mediation API or manual two-stage estimation.

---

## PHASE K — COUNTERFACTUALS & VALIDATION (Planned)

### K.1 Patient-Level Counterfactuals

For a specific patient who was NOT on a statin:
1. **Abduction:** Infer exogenous noise terms from observed data
2. **Action:** Set `statin_before = 1` in the SCM
3. **Prediction:** Propagate through the causal graph to get counterfactual P(MI)

Implementation via the trained f_y (classifier) with intervened features.

### K.2 Refutation Tests

Using `dowhy` refutation methods:
1. **Random Common Cause:** Add a random variable as confounder → ATE should not change
2. **Placebo Treatment:** Replace statin with random binary → ATE should be ~0
3. **Data Subset:** Re-estimate on random 80% subsets → ATE should be stable
4. **Sensitivity Analysis:** Rosenbaum bounds or E-values for unmeasured confounding

---

## PHASE M — ROBUSTNESS & SENSITIVITY (Planned)

### M.1 Unmeasured Confounding
- **E-values:** For each ATE/CATE estimate, compute the E-value — the minimum strength an unmeasured confounder would need to explain away the effect
- **Manski bounds:** Compute worst-case bounds under arbitrary unmeasured confounding

### M.2 IRM / Environment Robustness (Deferred from Phase G)
If pursued:
1. Extract ECG machine model from WFDB headers (`wfdb.rdheader()`)
2. Define environments: {TC50, TC70, Other}
3. Compare ERM vs IRM performance on held-out environment
4. Alternative: temporal split (train 2008-2015, test 2016-2019)

### M.3 Model Stability
- Bootstrap ATE/CATE (500 resamples)
- Cross-validation of nuisance models
- Comparison of XGBoost vs. LogReg as nuisance estimators in DML

---

## PHASE N — REPORTING & PUBLICATION (Planned)

### N.1 Results Summary Table
- ATE with 95% CI
- CATE by clinically relevant subgroups
- Refutation test results
- Baseline model comparison (already complete)

### N.2 Figures
- Causal DAG
- CATE heterogeneity plots
- Counterfactual case studies
- Feature importance (predictive vs. causal contributions)

### N.3 Limitations Statement
Must clearly distinguish:
1. **Predictive associations** (Phase H results) from **causal conclusions** (Phase J)
2. **Observational causal estimates** from **randomized trial evidence**
3. Explicitly list unmeasured confounders (lifestyle, genetic factors, medication adherence, socioeconomic status)

---

## PHASE O — MCP DASHBOARD ARCHITECTURE (Planned)

### O.1 Design Philosophy

The dashboard uses an **MCP (Model Context Protocol) server** that exposes scoped, validated tools to an LLM agent. The architecture enforces strict separation between:
- **Deterministic analytical logic** (statistical queries, model inference, causal estimates) — executed by MCP tools
- **Natural language interpretation** — handled by the LLM, constrained by tool output schemas

This prevents the LLM from hallucinating statistical results or making unsupported clinical claims.

### O.2 MCP Server Tool Definitions

#### Tool 1: `query_cohort_statistics`
```json
{
  "name": "query_cohort_statistics",
  "description": "Retrieve validated cohort statistics from the master dataset.",
  "inputs": {
    "split": "string (train|validation|test|all)",
    "group_by": "string[] (optional: age_group, sex_male, diabetes, prior_mi, Label_Troponin)",
    "metric": "string (count|prevalence|mean_age|label_distribution)"
  },
  "outputs": {
    "result": "DataFrame as JSON",
    "query_timestamp": "ISO timestamp",
    "data_version": "string (master dataset hash)"
  },
  "data_access": "READ-ONLY on data/modeling/master/*.parquet",
  "validation": "All queries pre-validated against known column names; no arbitrary SQL"
}
```

#### Tool 2: `get_baseline_results`
```json
{
  "name": "get_baseline_results",
  "description": "Retrieve validated baseline model performance metrics.",
  "inputs": {
    "model_name": "string (logreg_clinical|xgb_ecg_features_clinical|xgb_embeddings_clinical|xgb_all_features)",
    "metric": "string (auroc|auprc|brier_score|ece|subgroup_auroc|all)",
    "calibrated": "boolean (default: true)"
  },
  "outputs": {
    "metrics": "dict",
    "model_config": "dict (features used, hyperparameters)",
    "timestamp": "ISO timestamp of training run"
  },
  "data_access": "READ-ONLY on models/baselines/results.json, calibration_results.json"
}
```

#### Tool 3: `predict_mi_risk`
```json
{
  "name": "predict_mi_risk",
  "description": "Run MI risk prediction for a patient using a calibrated baseline model.",
  "inputs": {
    "model_name": "string",
    "features": "dict (named feature values matching model's expected input)",
    "return_explanation": "boolean (default: true)"
  },
  "outputs": {
    "probability": "float [0, 1]",
    "risk_category": "string (low|moderate|high)",
    "top_contributing_features": "list of {feature, value, shap_value}",
    "disclaimer": "string (always included: 'This is a predictive association, not a causal estimate.')"
  },
  "validation": [
    "Feature values checked against winsorization bounds",
    "Missing features filled with training median",
    "Probability calibrated via isotonic regression"
  ]
}
```

#### Tool 4: `get_causal_estimate` (Phase J+)
```json
{
  "name": "get_causal_estimate",
  "description": "Retrieve pre-computed causal effect estimates.",
  "inputs": {
    "treatment": "string (statin_before|beta_blocker_before|acei_before|aspirin_before)",
    "outcome": "string (MI_Acute_Presentation)",
    "estimand": "string (ate|cate)",
    "subgroup": "dict (optional: {diabetes: 1, age_group: '>70'})"
  },
  "outputs": {
    "effect": "float (risk difference)",
    "ci_lower": "float",
    "ci_upper": "float",
    "p_value": "float",
    "refutation_passed": "boolean",
    "interpretation": "string (pre-written, vetted causal language)",
    "caveats": "string[] (unmeasured confounders, observational design, etc.)"
  },
  "data_access": "READ-ONLY on pre-computed causal results JSON",
  "validation": [
    "Only returns pre-validated estimates (no on-the-fly causal computation)",
    "Always includes caveats about observational design",
    "Subgroup estimates only returned if n >= 100 in both treatment arms"
  ]
}
```

#### Tool 5: `generate_counterfactual` (Phase K+)
```json
{
  "name": "generate_counterfactual",
  "description": "Generate a counterfactual scenario for a specific patient.",
  "inputs": {
    "patient_features": "dict",
    "intervention": "dict ({statin_before: 1})",
    "model_name": "string (default: best calibrated model)"
  },
  "outputs": {
    "factual_probability": "float",
    "counterfactual_probability": "float",
    "risk_change": "float",
    "confidence_interval": "[float, float]",
    "causal_mechanism": "string (describes the DAG pathway)",
    "disclaimer": "string (observational counterfactual, not RCT evidence)"
  }
}
```

#### Tool 6: `visualize_risk_landscape`
```json
{
  "name": "visualize_risk_landscape",
  "description": "Generate pre-defined visualization of risk distributions or treatment effects.",
  "inputs": {
    "plot_type": "string (roc_comparison|calibration|cate_by_age|cate_by_subgroup|feature_importance)",
    "format": "string (png|svg, default: png)"
  },
  "outputs": {
    "image_path": "string (absolute path to generated/cached plot)",
    "caption": "string (descriptive caption for the visualization)",
    "data_summary": "dict (key statistics shown in the plot)"
  },
  "data_access": "READ-ONLY; serves pre-generated plots from models/baselines/ or generates from cached data"
}
```

### O.3 Execution Flow & Safeguards

```
User Query → LLM Agent
  ↓
LLM selects appropriate MCP tool(s)
  ↓
MCP Server validates inputs (schema, bounds, permissions)
  ↓
Tool executes deterministic logic (Pandas query, sklearn predict, cached lookup)
  ↓
Tool returns structured output with mandatory disclaimers
  ↓
LLM formats response using ONLY tool outputs
  ↓
User receives answer with traceable provenance
```

**Validation safeguards:**
1. **No arbitrary code execution:** Tools are pre-defined functions, not eval/exec endpoints
2. **Read-only data access:** MCP server has no write permissions to data or model files
3. **Mandatory disclaimers:** Every clinical output includes language distinguishing prediction from causation
4. **Subgroup minimum n:** Causal estimates suppressed for subgroups with <100 patients per treatment arm
5. **Feature bounds checking:** Input features validated against training distribution bounds
6. **Audit logging:** Every tool call logged with timestamp, inputs, outputs for reproducibility

### O.4 LLM Behavior Constraints

The LLM agent system prompt must enforce:
1. **Never claim causal relationships from predictive models** — use language like "associated with" not "causes"
2. **Never provide clinical recommendations** — always defer to clinical judgment
3. **Always cite the specific model and metric** when reporting results
4. **Distinguish calibrated vs. uncalibrated probabilities** — always prefer calibrated
5. **Report confidence intervals** alongside point estimates
6. **Flag subgroups with known performance disparities** (e.g., prior MI patients)

---

## FILE REFERENCE INDEX

### Data Pipeline
| File | Phase | Purpose |
|---|---|---|
| `scripts/extract_cohort.py` | B | Cohort extraction, treatment/lab/vital features, ECG mapping |
| `src/data/recover_ecg_mapping.py` | D.2 | study_id recovery via greedy sequence alignment |
| `src/features/extract_comorbidities.py` | E.1 | ICD-based comorbidity flags |
| `src/features/fix_temporal_features.py` | E.3 | Strict pre-MI temporal filtering for labs/vitals |
| `extract_ecg_features.m` | D.1 | MATLAB ECGdeli feature extraction (parallelized) |
| `data/build_modeling_data.py` | F.1 | Label joining, splitting, preprocessing, export |
| `src/features/build_master_dataset.py` | F.2 | Multi-modal master table assembly |

### Models
| File | Phase | Purpose |
|---|---|---|
| `src/data/convert_to_hdf5.py` | D.3 | WFDB → HDF5 conversion for fast I/O |
| `src/data/ecg_dataset.py` | D.4 | PyTorch Dataset classes (WFDB + HDF5) |
| `src/models/vae_conv1d.py` | D.4 | β-VAE architecture (Conv1DVAE) |
| `src/models/train_vae.py` | D.4 | VAE training with cyclical β-annealing |
| `src/models/extract_embeddings.py` | D.5 | Latent embedding extraction |
| `src/models/validate_embeddings.py` | D.6 | Embedding quality validation |
| `src/models/train_baselines.py` | H | 4 baseline classifiers with evaluation |
| `src/models/calibrate_baselines.py` | H.4 | Isotonic calibration of baselines |

### Data Artifacts
| File | Size | Key |
|---|---|---|
| `data/processed/ecg_signals.h5` | 90.6 GB | Pre-normalized 12-lead waveforms |
| `data/processed/extracted_ecg_features_with_study_id.csv` | 834 MB | 159 morphological features × 407K ECGs |
| `data/processed/final_ecg_cohorts_troponin.csv` | 74 MB | Master label file (408K ECGs) |
| `data/modeling/master/{split}.parquet` | 556 MB total | Final modeling tables |
| `data/modeling/split_manifest.csv` | 20 MB | Complete row-to-split mapping |
| `models/baselines/results.json` | 4.7 KB | All baseline metrics |
| `models/checkpoints/vae_*/best_model.pt` | 989 MB | Best VAE checkpoint (epoch 109) |

### Documentation
| File | Purpose |
|---|---|
| `data/processed/README.md` | Processed data manifest |
| `data/processed/Data_Engineering_Report.md` | Challenges and resolutions narrative |
| `data/modeling/README.md` | Modeling data directory guide |
| `docs/Phase F — Master Dataset.md` | Master dataset construction notes |
| `docs/Phase H — Baseline Predictive Models.md` | Baseline modeling phase notes |
| `docs/VAE Embedding Pipeline.md` | VAE design and training notes |