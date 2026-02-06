# PTB-XL & PTB-XL+ Dataset Analysis

**Generated**: February 6, 2026  
**Author**: Capstone Research Team

---

## 📊 Dataset Overview

### PTB-XL (v1.0.3) - Base ECG Dataset
- **Location**: `data/raw/PTB-XL-1.0.3/`
- **Records**: 21,801 10-second 12-lead ECGs
- **Patients**: ~18,000 unique patients
- **Sampling Rates**: 100Hz (low-res) and 500Hz (high-res)
- **Format**: WFDB format (.dat + .hdr files)

### PTB-XL+ (v1.0.1) - Enhanced Annotations
- **Location**: `data/raw/PTB-XL+-1.0.1/`
- **Purpose**: Ground truth fiducial points & extracted features
- **Annotation Method**: ECGdeli algorithm (expert-validated)

---

## 📁 Directory Structure

### PTB-XL-1.0.3/
```
├── ptbxl_database.csv          # Main metadata (demographics, diagnoses)
├── scp_statements.csv          # Diagnostic code definitions
├── records100/                 # 100Hz ECG signals
│   ├── 00000/                  # Batches of 1000 records
│   │   ├── 00001_lr.dat
│   │   ├── 00001_lr.hdr
│   │   └── ...
│   ├── 01000/
│   └── ...
└── records500/                 # 500Hz ECG signals (USE THIS)
    ├── 00000/
    │   ├── 00001_hr.dat
    │   ├── 00001_hr.hdr
    │   └── ...
    └── ...
```

### PTB-XL+-1.0.1/
```
├── fiducial_points/
│   └── ecgdeli/                # Ground truth fiducial annotations
│       ├── 00000/              # Per-lead fiducial points
│       │   ├── 00622_points_global.atr      # Global measurements
│       │   ├── 00622_points_lead_I.atr      # Lead I fiducials
│       │   ├── 00622_points_lead_II.atr     # Lead II fiducials
│       │   ├── ...                          # All 12 leads
│       │   └── 00622_points_lead_V6.atr
│       └── ...
├── features/
│   ├── ecgdeli_features.csv    # 🚨 Extracted features (>50MB)
│   ├── 12sl_features.csv       # Alternative features
│   ├── unig_features.csv       # University Göttingen features
│   └── feature_description.csv # Feature definitions (198 features!)
├── labels/
│   ├── 12sl_statements.csv     # Enhanced diagnostic labels
│   ├── ptbxl_statements.csv    # Original labels
│   └── snomed_description.csv  # SNOMED-CT codes
└── median_beats/               # Averaged beat templates
```

---

## 🔍 Key Files Explained

### 1. **ptbxl_database.csv** (21,801 records)
**Columns**:
- `ecg_id`: Unique ECG identifier (1-21837)
- `patient_id`: Patient identifier
- `age`, `sex`, `height`, `weight`: Demographics
- `filename_lr`: Path to 100Hz ECG
- `filename_hr`: Path to 500Hz ECG (USE THIS)
- `scp_codes`: Diagnostic codes (JSON format)
- `report`: Original German report

**Example**:
```csv
ecg_id,patient_id,age,sex,filename_hr
1,15709,56,1,records500/00000/00001_hr
```

### 2. **Fiducial Point Files (.atr)**
**Format**: WFDB annotation files  
**Types**:
- `*_points_global.atr`: Global measurements (HR, QT interval, etc.)
- `*_points_lead_X.atr`: Per-lead fiducial points (P, Q, R, S, T onsets/offsets)

**How to read**:
```python
import wfdb

# Load global annotations
ann = wfdb.rdann('data/raw/PTB-XL+-1.0.1/fiducial_points/ecgdeli/00000/00622_points_global', 
                  'atr', return_label_elements=['symbol'])
```

### 3. **ecgdeli_features.csv** (GROUND TRUTH)
**Size**: >50MB  
**Features**: ~198 features per ECG  
**Examples**:
- `PWd_II`: P-wave duration in lead II (ms)
- `QPa_II`: Q-wave amplitude in lead II (mV)
- `QRSd_max`: Maximum QRS duration across all leads (ms)
- `QTi_max`: Maximum QT interval (ms)
- `RRi_max`: RR interval (ms)

---

## 📐 Feature Categories (198 total)

From `feature_description.csv`:

### Global Features (per ECG)
```
P_AxisFront_Global       # P-wave axis (frontal plane)
P_Dur_Global             # P-wave duration (ms)
P_On_Global              # P-wave onset (ms from start)
P_Off_Global             # P-wave offset (ms)
PR_Int_Global            # PR interval (ms)
QRS_Dur_Global          # QRS duration (ms)
QRS_AxisFront_Global    # QRS axis (degrees)
QT_Int_Global           # QT interval (ms)
RR_Mean_Global          # Mean RR interval (ms)
RR_StdDev_Global        # RR variability (HRV, ms)
T_AxisFront_Global      # T-wave axis (degrees)
HR__Global              # Heart rate (bpm)
```

### LeadSpecific Features (per lead X = I, II, III, aVR, aVL, aVF, V1-V6)
```
P_Amp_X                 # P-wave amplitude (mV)
P_Dur_X                 # P-wave duration (ms)
Q_Amp_X                 # Q-wave amplitude (mV)
Q_Dur_X                 # Q-wave duration (ms)
R_Amp_X                 # R-wave amplitude (mV)
R_Dur_X                 # R-wave duration (ms)
S_Amp_X                 # S-wave amplitude (mV)
S_Dur_X                 # S-wave duration (ms)
ST_Slope_X              # ST-segment slope (degrees)
ST_Amp_X                # ST elevation/depression (mV)
T_Amp_X                 # T-wave amplitude (mV)
QRS_Dur_X               # QRS duration in lead X (ms)
QT_Int_X                # QT interval in lead X (ms)
```

### Morphology Features
```
t_wave_inverted         # Boolean flag
q_wave_present          # Boolean flag
st_deviation            # ST-segment deviation from baseline
```

---

## 🎯 Ground Truth Validation Strategy

### Approach
1. **Load PTB-XL signal** → Extract features using NeuroKit2
2. **Load PTB-XL+ ground truth** → Read from `ecgdeli_features.csv`
3. **Compare** → Calculate MAE, Pearson R for each feature
4. **Validate** → Check if algorithm meets clinical thresholds

### Key Metrics to Validate
```python
VALIDATION_METRICS = {
    'QRS_Dur_Global': {
        'ground_truth_col': 'QRSd_max',
        'mae_threshold': 10,  # ms
        'correlation_threshold': 0.90
    },
    'QT_Int_Global': {
        'ground_truth_col': 'QTi_max',
        'mae_threshold': 20,  # ms
        'correlation_threshold': 0.85
    },
    'PR_Int_Global': {
        'ground_truth_col': 'PRi_max',
        'mae_threshold': 15,  # ms
        'correlation_threshold': 0.85
    },
    'heart_rate': {
        'ground_truth_col': 'HR__Global',
        'mae_threshold': 5,  # bpm
        'correlation_threshold': 0.95
    }
}
```

---

## 🔧 Usage Examples

### Load ECG Signal
```python
import wfdb
import pandas as pd

# Load metadata
metadata = pd.read_csv('data/raw/PTB-XL-1.0.3/ptbxl_database.csv')

# Get first record
ecg_id = 1
filename = metadata[metadata['ecg_id'] == ecg_id]['filename_hr'].values[0]
record_path = f"data/raw/PTB-XL-1.0.3/{filename}"

# Load signal
record = wfdb.rdrecord(record_path)
signal = record.p_signal          # (5000, 12) - 10s @ 500Hz
lead_names = record.sig_name      # ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
sampling_rate = record.fs         # 500
```

### Load Ground Truth Features
```python
import pandas as pd

# Load PTB-XL+ features
features_gt = pd.read_csv('data/raw/PTB-XL+-1.0.1/features/ecgdeli_features.csv')

# Get features for ecg_id 1
ecg_features = features_gt[features_gt['ecg_id'] == 1]

# Access specific features
qrs_duration_gt = ecg_features['QRSd_max'].values[0]  # ms
qt_interval_gt = ecg_features['QTi_max'].values[0]    # ms
heart_rate_gt = ecg_features['HR__Global'].values[0]  # bpm
```

### Load Fiducial Points
```python
import wfdb

# Load global fiducial points
ecg_id = 622
fiducial_dir = 'data/raw/PTB-XL+-1.0.1/fiducial_points/ecgdeli'
folder = f"{ecg_id // 1000:05d}"
record_name = f"{fiducial_dir}/{folder}/{ecg_id:05d}_points_global"

ann = wfdb.rdann(record_name, 'atr')
# ann.sample = sample indices of fiducial points
# ann.symbol = point types ('(', 'N', 'p', 't', ')', etc.)
```

---

## 📊 Diagnostic Labels

### SCP Codes (Standardized Communication Protocol)
PTB-XL uses SCP statement codes for diagnoses. Example:
```python
scp_codes = {'MI': 100.0, 'LAFB': 50.0}
# MI = Myocardial Infarction (100% confidence)
# LAFB = Left Anterior Fascicular Block (50% confidence)
```

### Load Diagnostic Definitions
```python
import pandas as pd

scp_statements = pd.read_csv('data/raw/PTB-XL-1.0.3/scp_statements.csv')
# Columns: scp_code, statement, diagnostic, subdiagnostic, ...
```

---

## 🎓 Best Practices

### ✅ DO:
- Use 500Hz recordings (`records500/`) for feature extraction
- Validate on PTB-XL first before processing MIMIC-IV
- Compare extracted features against `ecgdeli_features.csv` ground truth
- Use Lead II for rhythm analysis (gold standard)
- Check physiological plausibility (HR: 30-200 bpm, QRS: 40-200 ms)

### ❌ DON'T:
- Use 100Hz recordings for detailed waveform analysis
- Skip ground truth validation
- Assume all features will extract successfully
- Ignore nan values in ground truth (some features may be missing)

---

## 📈 Expected Performance

Based on literature and PTB-XL+ benchmarks:

| Feature | Expected MAE | Expected R |
|---------|--------------|------------|
| Heart Rate | <3 bpm | >0.95 |
| QRS Duration | <10 ms | >0.90 |
| QT Interval | <20 ms | >0.85 |
| PR Interval | <15 ms | >0.85 |
| P-wave Duration | <8 ms | >0.80 |

---

## 🔗 References

1. **PTB-XL Dataset**: Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset", Scientific Data 2020
2. **PTB-XL+ Annotations**: Koka et al., "PTB-XL+, a comprehensive electrocardiographic feature dataset", Scientific Data 2024
3. **ECGdeli Algorithm**: Pilia et al., "ECGdeli - An open source ECG delineation toolbox", SoftwareX 2021

---

## 📝 Summary

**PTB-XL provides**:
- Raw ECG SIGNALS for feature extraction

**PTB-XL+ provides**:
1. **Fiducial POINTS** (.atr files) - Expert-annotated wave boundaries
2. **Extracted FEATURES** (ecgdeli_features.csv) - Ground truth measurements
3. **Enhanced LABELS** - Refined diagnostic codes

**Workflow**:
```
PTB-XL Signal → Your Algorithm → Extracted Features
                                      ↓ COMPARE
PTB-XL+ ecgdeli_features.csv → Ground Truth Features
                                      ↓
                                 VALIDATION METRICS
                                  (MAE, Correlation)
```

This enables **rigorous validation** before deploying on MIMIC-IV (which has no ground truth).
