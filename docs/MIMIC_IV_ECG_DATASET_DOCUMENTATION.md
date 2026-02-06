# MIMIC-IV ECG Dataset Documentation

**Comprehensive Guide to Datasets and Data Ingestion**  
**Project**: Causal Inference for ECG-Based Myocardial Infarction Analysis  
**Last Updated**: February 5, 2026

---

## Table of Contents

1. [Overview: Five Key Datasets](#overview)
2. [Dataset Details](#dataset-details)
   - [MIMIC-IV v2.2](#mimic-iv)
   - [MIMIC-IV-ECG v1.0](#mimic-ecg)
   - [MIMIC-IV-ECG EXT v1.0.1](#mimic-ecg-ext)
   - [PTB-XL v1.0.3](#ptb-xl)
   - [PhysioNet Waveform Database](#physionet)
3. [Data Ingestion (Phase B)](#data-ingestion)
4. [Dataset Usage Summary](#usage-summary)
5. [Research Papers](#research-papers)

---

<a name="overview"></a>
## 🎯 Overview: Five Key Datasets

We use **5 complementary datasets**, each serving a specific purpose in our causal inference pipeline:

```
📦 MIMIC-IV (2023)        →  Clinical context & patient data
📈 MIMIC-IV-ECG (2023)    →  Raw ECG signals (our primary data)
📊 MIMIC-IV-ECG EXT (2021)→  Extended ECG coverage
🏆 PTB-XL (2020)          →  Gold-standard validation
📚 PhysioNet DB (2017+)   →  Preprocessing standards
```

---

<a name="dataset-details"></a>
## 📚 Dataset Details

<a name="mimic-iv"></a>
### 1️⃣ MIMIC-IV v2.2 (2023)

#### What It Is
- **Full Name**: Medical Information Mart for Intensive Care IV
- **Released**: 2023 (latest version)
- **Source**: Beth Israel Deaconess Medical Center (Boston, USA)
- **Size**: 299,712 ICU patients (2008-2019)
- **Type**: Public ICU Electronic Health Record (EHR) dataset

#### What It Contains
| Data Category | Tables | What We Use |
|--------------|--------|-------------|
| **Demographics** | `patients` | Age, sex, race |
| **Admissions** | `admissions`, `transfers` | ICU stays, admission time, discharge time |
| **Laboratory** | `labevents`, `d_labitems` | **Troponin** (for MI labels), LDL, creatinine |
| **Diagnoses** | `diagnoses_icd` | Comorbidities (diabetes, hypertension, prior MI) |
| **Medications** | `prescriptions` | Statin use (our intervention variable) |
| **Vitals** | `chartevents` | Blood pressure, heart rate, SpO2 |

#### How It Helps Our Project

**Role 1: Cohort Building (Phase C)**
- We use **troponin measurements** to label MI cases:
  ```
  Troponin T > 0.014 ng/mL → MI_Acute_Presentation (with delta criterion)
  Troponin I > 0.040 ng/mL → MI_Acute_Presentation (with delta criterion)
  ```
- Result: Identified **14,809 MI cases** from 270,068 ECG records

**Role 2: Time-Anchoring**
- Admissions table provides timestamps to align:
  - ECG recording time
  - Troponin measurement time
  - Symptom onset
- This ensures we don't use "future information" to predict the past (temporal leakage prevention)

**Role 3: Confounding Variables (Phase E-F)**
- We extract variables that affect BOTH treatment and outcome:
  - Age → affects statin prescription AND MI risk
  - Diabetes → affects both
  - LDL cholesterol → affects both
- These must be controlled in causal analysis to isolate treatment effect

**Role 4: Intervention Variable**
- `prescriptions` table tells us who received statins **before** their ECG
- This is our **treatment variable** for causal analysis

#### Our Usage Statistics
```
✅ Database Size: 18.4 GB (DuckDB format)
✅ Tables Loaded: 9 tables
✅ Indices Created: 6 (for fast queries)
✅ Records in cohort_master: 270,068 ECGs
```

#### Research Paper

**Title**: "MIMIC-IV, a freely accessible electronic health record dataset"  
**Authors**: Johnson, A.E.W., Bulgarelli, L., Shen, L., et al.  
**Published**: *Scientific Data* (2023)  
**DOI**: 10.1038/s41597-022-01899-x  
**Institution**: MIT Laboratory for Computational Physiology, Beth Israel Deaconess Medical Center  
**Impact**: 2,000+ citations, most-used critical care dataset worldwide

**Key Contributions**:
- Largest public ICU dataset (299,712 patients, 11 years)
- HIPAA-compliant de-identification methodology
- Enhanced data linkage (ED, outpatient, social history)
- Improved data quality (<5% missing lab values vs 15-20% in MIMIC-III)

---

<a name="mimic-ecg"></a>
### 2️⃣ MIMIC-IV-ECG v1.0 (2023)

#### What It Is
- **Full Name**: MIMIC-IV Electrocardiography Database
- **Released**: 2023 (companion to MIMIC-IV)
- **Source**: Same patients as MIMIC-IV (Beth Israel Deaconess)
- **Size**: ~800,000+ ECG recordings
- **Format**: WFDB (Waveform Database) - 12-lead ECG, 500 Hz, 10 seconds
- **Type**: Digital ECG waveform repository

#### What It Contains
```
Each ECG recording includes:
├── 12 leads (I, II, III, aVR, aVL, aVF, V1-V6)
├── 5,000 samples per lead (10 seconds × 500 Hz)
├── Metadata (subject_id, study_id, ecg_time)
├── Machine diagnostics (automated interpretation by ECG device)
└── File path (e.g., p100/p10000032/s41256771/s41256771.hea)
```

#### How It Helps Our Project

**Role 1: Primary Data Source**
This is our **MAIN dataset** - the raw ECG signals for analysis.

**Role 2: Feature Extraction (Phase D.2)**
We extract **24 clinical features** from each ECG using NeuroKit2:
- Heart rate
- PR interval, QRS duration, QT interval, QTc
- P-wave, QRS, T-wave amplitudes and durations
- ST-segment elevation/depression (critical for MI)

**Role 3: ECG-to-Clinical Linkage**
Each ECG is time-stamped, allowing us to:
- Match ECG to troponin measurement (for labeling)
- Determine if ECG was taken **during** acute MI or **before** MI
- Create time-anchored cohorts

#### Our Usage Statistics
```
✅ Raw Dataset: ~800,000 ECGs available
✅ Cohort Master: 270,068 ECGs (with labels)
✅ Final cohort labels:
   ├── Control_Asymptomatic: 133,235 (49.3%)
   ├── Control_Symptomatic: 106,380 (39.4%)
   ├── MI_Pre-Incident: 15,644 (5.8%)
   └── MI_Acute_Presentation: 14,809 (5.5%)
```

---

<a name="mimic-ecg-ext"></a>
### 3️⃣ MIMIC-IV-ECG EXT v1.0.1 (2021)

#### What It Is
- **Full Name**: MIMIC-IV-ECG Extended with ICD Diagnoses
- **Released**: 2021
- **Source**: Same institution, additional time periods
- **Size**: Additional ~200,000 ECG recordings
- **Type**: Extension providing broader temporal coverage

#### How It Helps Our Project

**Role 1: Expanding ECG Diversity**
- Includes ECGs from different time periods, patient populations, clinical contexts
- Improves model generalizability

**Role 2: Timeline Coverage**
- Some patients have ECGs spanning 10+ years
- Allows study of **ECG evolution** over time

**Role 3: ICD Diagnosis Cross-Validation**
- EXT includes direct ICD codes for each ECG
- Validates our troponin-based labels against physician diagnoses

#### Our Usage Statistics
```
⏳ Status: Available but NOT yet ingested
📋 Plan: Phase E-F (dataset expansion)
🎯 Purpose: Increase sample size for rare subgroups
```

---

<a name="ptb-xl"></a>
### 4️⃣ PTB-XL v1.0.3 (2020)

#### What It Is
- **Full Name**: Physikalisch-Technische Bundesanstalt ECG Database XL
- **Released**: 2020 (benchmark dataset)
- **Source**: PTB Hospital (Germany), multiple centers
- **Size**: 21,837 ECG recordings from 18,885 patients
- **Format**: WFDB, 12-lead, 500 Hz, 10 seconds (same as MIMIC)
- **Type**: **Gold-standard validation dataset** with expert annotations

#### What It Contains
| Data Type | Details |
|-----------|---------|
| **ECG Waveforms** | 21,837 high-quality 12-lead ECGs |
| **Expert Labels** | Cardiologist-annotated diagnoses (71 classes) |
| **Fiducial Points** | P-wave onset/offset, QRS onset/offset, T-wave end (PTB-XL+) |
| **Demographic Data** | Age, sex, device type |
| **Diagnostic Statements** | "Myocardial Infarction", "Normal ECG", "LBBB", etc. |

#### How It Helps Our Project

**Role 1: Feature Extraction Validation (Phase D.1)**
- PTB-XL+ has **expert-annotated fiducial points** (ground truth)
- We test our NeuroKit2 extraction against these annotations
- Ensures our feature measurements are clinically accurate

**Role 2: ECG Distribution Validation**
- Compare MIMIC-IV statistics to PTB-XL (general population)
- Identify biases in MIMIC-IV (e.g., sicker ICU population)

**Role 3: Cross-Dataset Generalization Test**
- Test trained models on PTB-XL
- Proves robustness beyond MIMIC-specific artifacts

#### Our Usage Statistics
```
✅ Downloaded: PTB-XL v1.0.3 + PTB-XL+ v1.0.1
✅ Validation Completed: 100 ECGs processed
✅ Reports Generated: fiducial_validation_report.md
⏳ Status: Ready for Phase D.4-D.5 (latent space validation)
```

#### Research Paper

**Title**: "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL"  
**Authors**: Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W.  
**Published**: *arXiv preprint* (2020)  
**DOI**: https://doi.org/10.48550/arXiv.2004.13701  
**Relevance**: Comprehensive benchmarking of deep learning methods on PTB-XL dataset

---

<a name="physionet"></a>
### 5️⃣ PhysioNet Waveform Database (2017+)

#### What It Is
- **Full Name**: PhysioNet Clinical Waveform Repository
- **Established**: 2017+ (ongoing updates)
- **Source**: Multiple hospitals and research institutions worldwide
- **Size**: >100 different ECG databases
- **Type**: Standard-setting repository for physiological signals

#### What It Contains
- **Databases**: ~15 major ECG databases (MIT-BIH, AHA, ESC, etc.)
- **Formats**: WFDB standard (header + signal files)
- **Tools**: WFDB library (Python/C) for reading/writing signals
- **Standards**: Sampling rates, amplitude units, annotation formats

#### How It Helps Our Project

**Role 1: Cross-Checking ECG Preprocessing Standards**

| Preprocessing Step | PhysioNet Standard | Our Implementation |
|--------------------|-------------------|-------------------|
| **Sampling Rate** | 500 Hz (clinical standard) | ✅ 500 Hz (MIMIC-IV-ECG native) |
| **Duration** | 10 seconds (standard 12-lead) | ✅ 10 seconds |
| **Bandpass Filter** | 0.5-50 Hz (remove baseline wander + high-freq noise) | ✅ 0.5-50 Hz |
| **Amplitude Units** | mV (millivolts) | ✅ mV (WFDB format) |
| **Lead Order** | I, II, III, aVR, aVL, aVF, V1-V6 | ✅ Standard order |

**Role 2: WFDB Library Usage**
We use the WFDB Python library (official PhysioNet tool):
```python
import wfdb

# Load ECG signal
record = wfdb.rdrecord('path/to/ecg/p100/p10000032/s41256771')

# Access data
signal = record.p_signal  # Shape: (5000, 12) - 12 leads
sampling_rate = record.fs  # 500 Hz
lead_names = record.sig_name  # ['I', 'II', 'III', ...]
```

#### Our Usage Statistics
```
✅ Library Installed: wfdb 4.3.0
✅ Used For: Loading MIMIC-IV-ECG and PTB-XL
✅ Standards Adopted: All PhysioNet preprocessing recommendations
✅ Validation: Compared our features against PTB-XL+ ground truth
```


---

**Last Updated**: February 5, 2026  
**Database Version**: MIMIC-IV v2.2, MIMIC-IV-ECG v1.0  
**Python Version**: 3.11+  
**Key Libraries**: duckdb, wfdb, pandas, numpy
