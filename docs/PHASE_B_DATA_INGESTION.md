# Phase B: Data Ingestion Documentation

**Complete Multi-Dataset Database Ingestion**  
**Last Updated**: February 5, 2026  
**Status**: ✅ Complete - All 5 Datasets Loaded

---

## Table of Contents

1. [Overview](#overview)
2. [Purpose & Objectives](#purpose)
3. [Prerequisites](#prerequisites)
4. [Directory Structure](#directory-structure)
5. [Implementation Details](#implementation)
6. [Database Schema](#schema)
7. [Performance Optimization](#performance)
8. [Verification & Quality Checks](#verification)
9. [How to Run](#how-to-run)
10. [Troubleshooting](#troubleshooting)
11. [Outputs](#outputs)

---

<a name="overview"></a>
## 📋 Overview

**Phase B** is the comprehensive data ingestion phase that loads all 5 datasets (MIMIC-IV, MIMIC-IV-ECG, MIMIC-IV-ECG-Ext-ICD, PTB-XL, PTB-XL+) into a local DuckDB database for efficient querying and analysis.

**Script**: `src/data/data_ingestion.py`  
**Duration**: ~2-3 hours (depending on system)  
**Output**: `data/mimic_database.duckdb` (~38.8 GB)

### What This Phase Does

1. **Loads 11 MIMIC-IV tables** from CSV files (clinical data)
2. **Loads 4 MIMIC-IV-ECG tables** from CSV files (ECG metadata & measurements)
3. **Loads 1 MIMIC-IV-ECG-Ext-ICD table** (ECG-ICD10 linkage)
4. **Loads 2 PTB-XL tables** (external validation dataset)
5. **Loads 3 PTB-XL+ tables** (expert interpretations)
6. **Creates 6 performance indices** for fast queries
7. **Validates data integrity** (456M+ rows, 21 tables)

### Why DuckDB?

- ✅ **Fast**: Columnar storage optimized for analytics
- ✅ **Embedded**: No separate server process needed
- ✅ **SQL Support**: Full SQL with window functions, CTEs
- ✅ **Efficient**: Automatic compression saves disk space
- ✅ **Portable**: Single file database (easy backup/transfer)
- ✅ **Scalable**: Handles 456M+ rows across 21 tables efficiently

---

<a name="purpose"></a>
## 🎯 Purpose & Objectives

### Primary Objectives

1. **Centralize Data Access**
   - Consolidate all 5 datasets (31 CSV files) into single queryable database
   - Enable fast SQL queries across linked tables
   - Eliminate need to load CSVs repeatedly

2. **Enable Downstream Phases**
   - Phase C: Cohort definition requires troponin labs + ECG timestamps + ICD diagnoses
   - Phase D: VAE training needs ECG waveforms and machine measurements
   - Phase E-F: Feature extraction needs patient demographics + clinical data + procedures
   - Phase G-M: Causal inference requires medications + outcomes + external validation

3. **Optimize Query Performance**
   - Create indices on frequently joined columns
   - Reduce query time from minutes to seconds
   - Support complex multi-table joins across datasets

4. **Ensure Data Quality**
   - Verify all columns loaded correctly (100% validation)
   - Detect missing or corrupted files
   - Validate table linkages (subject_id, hadm_id, study_id)
   - Check foreign key integrity and duplicate detection

---

<a name="prerequisites"></a>
## ⚙️ Prerequisites

### Required Data Files

Download and extract the following datasets to `data/raw/`:

#### 1. MIMIC-IV v2.2 (11 files)
```
data/raw/MIMIC-IV-2.2/
├── hosp/
│   ├── patients.csv (9.42 MB)
│   ├── admissions.csv (62.03 MB)
│   ├── diagnoses_icd.csv (394.83 MB)
│   ├── labevents.csv (9,234.83 MB) ⚠️ Large file
│   ├── d_labitems.csv (0.04 MB)
│   ├── prescriptions.csv (764.25 MB)
│   ├── d_icd_diagnoses.csv (8.44 MB) 🆕
│   └── procedures_icd.csv (25.10 MB) 🆕
└── icu/
    ├── chartevents.csv (28,805.28 MB) ⚠️ Very large file
    ├── icustays.csv (10.94 MB) 🆕
    └── d_items.csv (0.35 MB) 🆕 CRITICAL - Dictionary for chartevents
```

**Download**: https://physionet.org/content/mimiciv/2.2/  
**Access**: Requires PhysioNet credentialed user account

#### 2. MIMIC-IV-ECG v1.0 (4 files)
```
data/raw/MIMIC-IV-ECG-1.0/
├── record_list.csv (67.14 MB)
├── machine_measurements.csv (174.21 MB)
├── machine_measurements_data_dictionary.csv (0.00 MB) 🆕 CRITICAL
└── waveform_note_links.csv (56.20 MB) 🆕
```

**Download**: https://physionet.org/content/mimic-iv-ecg/1.0/  
**Access**: Requires PhysioNet credentialed user account

#### 3. MIMIC-IV-ECG-Ext-ICD v1.0.1 (1 file)
```
data/raw/MIMIC-IV-ECG-Ext-ICD-1.0.1/
└── records_w_diag_icd10.csv (323.21 MB) 🆕 CRITICAL - ECG-ICD10 linkage
```

**Download**: https://physionet.org/content/mimic-iv-ecg-icd/1.0.1/  
**Access**: Requires PhysioNet credentialed user account  
**Purpose**: Links ECG studies to ICD-10 diagnoses for ground truth MI labels

#### 4. PTB-XL v1.0.3 (2 files)
```
data/raw/PTB-XL-1.0.3/
├── ptbxl_database.csv (6.29 MB) 🆕 CRITICAL - External validation
└── scp_statements.csv (0.01 MB) 🆕 CRITICAL - SCP code dictionary
```

**Download**: https://physionet.org/content/ptb-xl/1.0.3/  
**Access**: Open access (no credentials required)  
**Purpose**: External validation dataset with 21,799 ECGs

#### 5. PTB-XL+ v1.0.1 (3 files)
```
data/raw/PTB-XL+-1.0.1/
└── labels/
    ├── 12sl_statements.csv (16.94 MB) 🆕
    ├── ptbxl_statements.csv (15.20 MB) 🆕
    └── snomed_description.csv (0.06 MB) 🆕 CRITICAL - SNOMED dictionary
```

**Download**: https://physionet.org/content/ptb-xl-plus/1.0.1/  
**Access**: Open access (no credentials required)  
**Purpose**: Expert and machine interpretations for validation

### System Requirements

- **Disk Space**: 50+ GB free (CSVs + database + temp files)
- **RAM**: 16 GB minimum, 32 GB recommended (for chartevents/labevents loading)
- **CPU**: Multi-core processor (DuckDB is multi-threaded)
- **OS**: Windows 10/11, macOS 10.15+, Linux

### Software Requirements

- **Python**: 3.11 or higher
- **DuckDB**: 0.9.0+ (installed via pip)

```bash
pip install duckdb
```

---

<a name="directory-structure"></a>
## 📁 Directory Structure

### Before Running Phase B

```
Capstone/
├── data/
│   ├── raw/
│   │   ├── MIMIC-IV-2.2/
│   │   │   ├── hosp/
│   │   │   │   ├── patients.csv
│   │   │   │   ├── admissions.csv
│   │   │   │   ├── diagnoses_icd.csv
│   │   │   │   ├── labevents.csv
│   │   │   │   ├── d_labitems.csv
│   │   │   │   ├── prescriptions.csv
│   │   │   │   ├── d_icd_diagnoses.csv
│   │   │   │   └── procedures_icd.csv
│   │   │   └── icu/
│   │   │       ├── chartevents.csv
│   │   │       ├── icustays.csv
│   │   │       └── d_items.csv
│   │   ├── MIMIC-IV-ECG-1.0/
│   │   │   ├── record_list.csv
│   │   │   ├── machine_measurements.csv
│   │   │   ├── machine_measurements_data_dictionary.csv
│   │   │   └── waveform_note_links.csv
│   │   ├── MIMIC-IV-ECG-Ext-ICD-1.0.1/
│   │   │   └── records_w_diag_icd10.csv
│   │   ├── PTB-XL-1.0.3/
│   │   │   ├── ptbxl_database.csv
│   │   │   └── scp_statements.csv
│   │   └── PTB-XL+-1.0.1/
│   │       └── labels/
│   │           ├── 12sl_statements.csv
│   │           ├── ptbxl_statements.csv
│   │           └── snomed_description.csv
│   ├── interim/
│   ├── processed/
│   └── mimic_database.duckdb (will be created)
└── src/
    └── data/
        └── data_ingestion.py
```

### After Running Phase B

```
Capstone/
├── data/
│   ├── mimic_database.duckdb ✅ (~38.8 GB)
│   └── raw/ (unchanged)
└── DATABASE_VALIDATION_REPORT.md ✅ (quality assurance)
```

---

<a name="implementation"></a>
## 🔧 Implementation Details

### Step-by-Step Process

#### B.1: Database Connection
```python
con = duckdb.connect('data/mimic_database.duckdb')
```
- Creates new database file if doesn't exist
- Opens existing database if already present (⚠️ overwrites tables!)

#### B.2: Load MIMIC-IV Tables (11 tables)

**Pattern used for all tables**:
```python
con.execute(f"""
    CREATE TABLE {table_name} AS 
    SELECT * FROM read_csv_auto('{csv_path}', header=True)
""")
```

**Tables loaded**:
1. **patients** (299,712 rows) - Patient demographics
2. **admissions** (431,231 rows) - Hospital admissions
3. **diagnoses_icd** (4,756,326 rows) - ICD diagnosis codes
4. **labevents** (118,171,367 rows) - Laboratory measurements
5. **d_labitems** (1,622 rows) - Lab item dictionary
6. **prescriptions** (15,416,708 rows) - Medication orders
7. **d_icd_diagnoses** (109,775 rows) - ICD diagnosis descriptions 🆕
8. **procedures_icd** (669,186 rows) - ICD procedure codes (PCI, CABG) 🆕
9. **chartevents** (313,645,063 rows) - ICU vital signs ⚠️ Largest table
10. **icustays** (73,181 rows) - ICU stay information 🆕
11. **d_items** (4,014 rows) - Chart items dictionary 🆕 CRITICAL

#### B.3: Load MIMIC-ECG Tables (4 tables)

1. **record_list** (800,035 rows) - ECG study metadata
2. **machine_measurements** (800,035 rows) - ECG automated measurements
3. **machine_measurements_data_dictionary** (16 rows) - ECG dictionary 🆕 CRITICAL
4. **waveform_note_links** (609,272 rows) - Links to clinical notes 🆕

#### B.3.1: Load MIMIC-ECG-Ext-ICD Tables (1 table)

1. **records_w_diag_icd10** (800,035 rows) - ECG-ICD10 diagnosis linkage 🆕 CRITICAL

#### B.3.2: Load PTB-XL Tables (2 tables)

1. **ptbxl_database** (21,799 rows) - PTB-XL metadata 🆕 CRITICAL
2. **scp_statements** (71 rows) - SCP diagnostic code dictionary 🆕 CRITICAL

#### B.3.3: Load PTB-XL+ Tables (3 tables)

1. **statements_12sl** (21,799 rows) - Machine interpretations 🆕
2. **ptbxl_statements** (21,799 rows) - Expert interpretations 🆕
3. **snomed_description** (287 rows) - SNOMED code dictionary 🆕 CRITICAL

#### B.4: Create Performance Indices (6 indices)

**Purpose**: Dramatically speed up JOIN and WHERE queries

```python
# Enable fast patient lookups
CREATE INDEX idx_labevents_subject ON labevents(subject_id)
CREATE INDEX idx_chartevents_subject ON chartevents(subject_id)

# Enable fast admission lookups
CREATE INDEX idx_labevents_hadm ON labevents(hadm_id)
CREATE INDEX idx_diagnoses_hadm ON diagnoses_icd(hadm_id)

# Enable fast temporal queries
CREATE INDEX idx_labevents_charttime ON labevents(charttime)
CREATE INDEX idx_chartevents_charttime ON chartevents(charttime)
```

**Performance Impact**:
- ❌ Without indices: ~60 seconds for troponin query
- ✅ With indices: ~0.3 seconds for same query (200x speedup)

---

<a name="schema"></a>
## 📊 Database Schema

### Complete Table Inventory (21 Tables, 456M+ Rows)

#### MIMIC-IV Tables (11 tables)

| Table | Rows | Columns | Purpose | Size |
|-------|------|---------|---------|------|
| **patients** | 299,712 | 6 | Demographics | ~10 MB |
| **admissions** | 431,231 | 16 | Hospital visits | ~62 MB |
| **diagnoses_icd** | 4,756,326 | 5 | ICD codes | ~395 MB |
| **labevents** | 118,171,367 | 16 | Lab measurements | ~9.2 GB |
| **d_labitems** | 1,622 | 4 | Lab dictionary | <1 MB |
| **prescriptions** | 15,416,708 | 21 | Medications | ~764 MB |
| **d_icd_diagnoses** 🆕 | 109,775 | 3 | ICD descriptions | ~8 MB |
| **procedures_icd** 🆕 | 669,186 | 6 | Procedures (PCI/CABG) | ~25 MB |
| **chartevents** | 313,645,063 | 11 | ICU vitals | ~28.8 GB |
| **icustays** 🆕 | 73,181 | 8 | ICU stays | ~11 MB |
| **d_items** 🆕 | 4,014 | 9 | Chart dictionary | <1 MB |

#### MIMIC-ECG Tables (4 tables)

| Table | Rows | Columns | Purpose | Size |
|-------|------|---------|---------|------|
| **record_list** | 800,035 | 5 | ECG metadata | ~67 MB |
| **machine_measurements** | 800,035 | 33 | ECG measures | ~174 MB |
| **machine_measurements_data_dictionary** 🆕 | 16 | 2 | ECG dictionary | <1 MB |
| **waveform_note_links** 🆕 | 609,272 | 6 | Clinical notes | ~56 MB |

#### MIMIC-ECG-Ext-ICD Tables (1 table)

| Table | Rows | Columns | Purpose | Size |
|-------|------|---------|---------|------|
| **records_w_diag_icd10** 🆕 | 800,035 | 22 | ECG-ICD10 linkage | ~323 MB |

#### PTB-XL Tables (2 tables)

| Table | Rows | Columns | Purpose | Size |
|-------|------|---------|---------|------|
| **ptbxl_database** 🆕 | 21,799 | 28 | External validation | ~6 MB |
| **scp_statements** 🆕 | 71 | 13 | SCP code dictionary | <1 MB |

#### PTB-XL+ Tables (3 tables)

| Table | Rows | Columns | Purpose | Size |
|-------|------|---------|---------|------|
| **statements_12sl** 🆕 | 21,799 | 4 | Machine interpretations | ~17 MB |
| **ptbxl_statements** 🆕 | 21,799 | 4 | Expert interpretations | ~15 MB |
| **snomed_description** 🆕 | 287 | 8 | SNOMED dictionary | <1 MB |

**Total**: 456,653,333 rows across 21 tables → ~38.8 GB database

### Key Column Relationships

```
patients.subject_id ←→ admissions.subject_id
                    ←→ labevents.subject_id
                    ←→ chartevents.subject_id
                    ←→ diagnoses_icd.subject_id
                    ←→ prescriptions.subject_id
                    ←→ record_list.subject_id
                    ←→ records_w_diag_icd10.subject_id

admissions.hadm_id ←→ labevents.hadm_id
                   ←→ diagnoses_icd.hadm_id
                   ←→ prescriptions.hadm_id
                   ←→ procedures_icd.hadm_id

record_list.study_id ←→ machine_measurements.study_id
                     ←→ records_w_diag_icd10.study_id
                     ←→ waveform_note_links.study_id

ptbxl_database.ecg_id ←→ statements_12sl.ecg_id
                      ←→ ptbxl_statements.ecg_id
```

### Critical Columns for Causal Inference Pipeline

#### Phase C (Cohort Labeling)
- `labevents.itemid` - Identifies troponin tests
- `labevents.valuenum` - Troponin concentration (ng/mL)
- `labevents.charttime` - Lab measurement timestamp
- `record_list.ecg_time` - ECG recording timestamp
- `admissions.admittime` - Hospital admission time
- `records_w_diag_icd10.all_diag_all` - ICD-10 diagnoses for ground truth 🆕

#### Phase D (VAE Training)
- `machine_measurements.qrs_axis`, `t_axis`, `p_axis` - ECG features
- `machine_measurements_data_dictionary` - Feature descriptions 🆕
- `ptbxl_database` - External validation ECGs 🆕

#### Phase E-F (Feature Extraction)
- `patients.gender` - Sex (confounder)
- `patients.anchor_age` - Age (confounder)
- `diagnoses_icd.icd_code` - Comorbidities
- `d_icd_diagnoses.long_title` - Readable diagnosis names 🆕
- `procedures_icd.icd_code` - PCI/CABG procedures 🆕
- `prescriptions.drug` - Statin use (treatment)
- `icustays.los` - ICU severity 🆕
- `d_items` - Interpret chartevents features 🆕

#### Phase G-M (Causal Analysis)
- `prescriptions.starttime` - Intervention timing
- `labevents.valuenum` (LDL) - Mechanism variable
- `admissions.hospital_expire_flag` - Outcome
- `ptbxl_database` - External validation cohort 🆕

---

<a name="performance"></a>
## ⚡ Performance Optimization

### Index Strategy

**Why These 6 Indices?**

1. **idx_labevents_subject**: Most queries filter by patient
2. **idx_labevents_hadm**: Cohort labeling joins on admission
3. **idx_labevents_charttime**: Time-windowed troponin queries
4. **idx_chartevents_subject**: Vital signs by patient
5. **idx_chartevents_charttime**: Temporal vital sign queries
6. **idx_diagnoses_hadm**: Comorbidity lookups per admission

**Index Overhead**:
- Storage: ~2 GB additional space
- Build time: ~10-15 minutes
- Query speedup: 100-200x for filtered queries

### DuckDB Auto-Optimization

DuckDB automatically:
- ✅ Compresses data (saves ~60% disk space)
- ✅ Uses parallel processing (multi-core)
- ✅ Optimizes query plans
- ✅ Caches frequently accessed data

---

<a name="verification"></a>
## ✅ Verification & Quality Checks

### Automated Verification

**Script**: `scripts/validate_database_quality.py`  
**Report**: `DATABASE_VALIDATION_REPORT.md`

Comprehensive checks performed:
1. ✅ All 21 tables present with data
2. ✅ 456,653,333 total rows loaded
3. ✅ No duplicate records in primary keys
4. ✅ Perfect foreign key integrity (0 orphaned records)
5. ✅ Data types consistent across all columns
6. ✅ NULL values within expected ranges for medical data
7. ✅ Date shifting working correctly (MIMIC-IV privacy)

**Run verification**:
```bash
python scripts/validate_database_quality.py
```

**Expected output**:
```
DATABASE VALIDATION: PASSED
Total Tables: 21 (all loaded successfully)
Total Rows: 456,653,333
Quality Status: EXCELLENT - No critical errors
```

### Validation Summary

**✅ Database Quality: 100%**
- Table Existence: 21/21 ✅
- Primary Keys: No duplicates ✅
- Foreign Keys: No orphans ✅
- Data Types: All consistent ✅
- Critical Fields: Complete ✅

### Manual Verification Queries

**Check table counts**:
```python
import duckdb
con = duckdb.connect('data/mimic_database.duckdb')

# Should return 21
print(con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main'").fetchone()[0])

# Should return 6
print(con.execute("SELECT COUNT(*) FROM pg_indexes").fetchone()[0])

con.close()
```

**Check troponin data (critical for Phase C)**:
```sql
SELECT 
    itemid, 
    label, 
    COUNT(*) as measurement_count
FROM labevents le
JOIN d_labitems di ON le.itemid = di.itemid
WHERE di.label LIKE '%troponin%'
GROUP BY itemid, label;
```

**Expected output**:
```
itemid  label         measurement_count
51002   Troponin I    163
51003   Troponin T    359,000
52642   Troponin I    (additional)
```

---

<a name="how-to-run"></a>
## 🚀 How to Run

### First-Time Setup

1. **Download datasets** (requires PhysioNet credentials):
   - MIMIC-IV v2.2: https://physionet.org/content/mimiciv/2.2/
   - MIMIC-IV-ECG v1.0: https://physionet.org/content/mimic-iv-ecg/1.0/

2. **Extract to `data/raw/`**:
   ```bash
   # Extract MIMIC-IV
   unzip mimic-iv-2.2.zip -d data/raw/MIMIC-IV-2.2/
   
   # Extract MIMIC-ECG
   unzip mimic-iv-ecg-1.0.zip -d data/raw/MIMIC-IV-ECG-1.0/
   ```

3. **Verify directory structure**:
   ```bash
   # Should show all MIMIC-IV files
   ls data/raw/MIMIC-IV-2.2/hosp/  # 8 CSV files
   ls data/raw/MIMIC-IV-2.2/icu/   # 3 CSV files
   
   # Should show all MIMIC-ECG files
   ls data/raw/MIMIC-IV-ECG-1.0/  # 4 CSV files
   
   # Should show extension datasets
   ls data/raw/MIMIC-IV-ECG-Ext-ICD-1.0.1/  # 1 CSV file
   ls data/raw/PTB-XL-1.0.3/  # 2 CSV files
   ls data/raw/PTB-XL+-1.0.1/labels/  # 3 CSV files
   ```

### Running Phase B

**Command**:
```bash
python src/data/data_ingestion.py
```

**Expected runtime**:
- Fast SSD + 32GB RAM: ~2 hours
- Regular HDD + 16GB RAM: ~3 hours
- Cloud VM (8 cores): ~1.5 hours

**Console output**:
```
======================================================================
PHASE B: MIMIC DATA INGESTION
======================================================================
Database file: mimic_database.duckdb
MIMIC-IV Hospital path: data/raw/MIMIC-IV-2.2/hosp/
MIMIC-IV ICU path: data/raw/MIMIC-IV-2.2/icu/
MIMIC-ECG path: data/raw/MIMIC-IV-ECG-1.0/
MIMIC-ECG-Ext-ICD path: data/raw/MIMIC-IV-ECG-Ext-ICD-1.0.1/
PTB-XL path: data/raw/PTB-XL-1.0.3/
PTB-XL+ path: data/raw/PTB-XL+-1.0.1/
======================================================================

======================================================================
B.2: LOADING MIMIC-IV TABLES
======================================================================

Loading 'patients' table...
✓ 'patients' table loaded successfully.

Loading 'admissions' table...
✓ 'admissions' table loaded successfully.

... (continues for all 21 tables) ...

======================================================================
MIMIC-IV TABLES LOADED: 11/11
======================================================================

======================================================================
B.3: LOADING MIMIC-ECG TABLES
======================================================================
... (4 tables) ...

======================================================================
B.3.1: LOADING MIMIC-ECG-EXT-ICD TABLES
======================================================================
... (1 table) ...

======================================================================
B.3.2: LOADING PTB-XL TABLES
======================================================================
... (2 tables) ...

======================================================================
B.3.3: LOADING PTB-XL+ TABLES
======================================================================
... (3 tables) ...

======================================================================
B.4: CREATING PERFORMANCE INDICES
======================================================================

Creating index on labevents.subject_id...
✓ Index 'idx_labevents_subject' created.

... (continues for all indices) ...

======================================================================
Phase B ingestion complete. Database is ready.
======================================================================

Database location: C:\Users\...\Capstone\data\mimic_database.duckdb
Total tables loaded: 9
Total indices created: 6

You can now proceed to Phase C: Data Processing and Analysis
======================================================================
```

### Re-Running Phase B

⚠️ **Warning**: Re-running will **OVERWRITE existing tables**

If you need to re-run:
1. **Backup existing database**:
   ```bash
   cp data/mimic_database.duckdb data/mimic_database_backup.duckdb
   ```

2. **Delete existing database** (optional but recommended):
   ```bash
   rm data/mimic_database.duckdb
   ```

3. **Run Phase B**:
   ```bash
   python src/data/data_ingestion.py
   ```

---

<a name="troubleshooting"></a>
## 🔧 Troubleshooting

### Common Errors

#### Error: "File not found: patients.csv"

**Cause**: Incorrect directory structure or paths

**Solution**:
1. Verify CSVs are in correct location:
   ```bash
   ls data/raw/MIMIC-IV-2.2/hosp/patients.csv
   ```
2. Check path configuration in `data_ingestion.py`:
   ```python
   MIMIC_IV_HOSP_PATH = "data/raw/MIMIC-IV-2.2/hosp/"
   ```

#### Error: "Out of memory"

**Cause**: Insufficient RAM for large tables (chartevents, labevents)

**Solution**:
1. Close other applications
2. Increase system swap/pagefile
3. Use a machine with more RAM (32GB recommended)

#### Error: "Catalog Error: Table already exists"

**Cause**: Database already contains tables from previous run

**Solution**:
1. Delete existing database:
   ```bash
   rm data/mimic_database.duckdb
   ```
2. Re-run ingestion

#### Slow Performance

**Cause**: Running on slow disk (HDD vs SSD)

**Solution**:
1. Move `data/` directory to SSD
2. Update paths in `data_ingestion.py`
3. Use `--memory-limit` flag for DuckDB (advanced)

---

<a name="outputs"></a>
## 📤 Outputs

### Primary Output

**File**: `data/mimic_database.duckdb`  
**Size**: ~38.8 GB  
**Format**: DuckDB database file  
**Contents**: 21 tables, 6 indices, 456M+ rows

### Database Statistics

```
Total Tables: 21
  - MIMIC-IV: 11 tables
  - MIMIC-ECG: 4 tables
  - MIMIC-ECG-Ext-ICD: 1 table
  - PTB-XL: 2 tables
  - PTB-XL+: 3 tables
Total Rows: 456,653,333
Total Indices: 6
Compression Ratio: ~60% (raw CSVs → compressed database)
```

### Quality Metrics

- ✅ **Completeness**: All 21 tables loaded with 100% data integrity
- ✅ **Integrity**: No duplicate primary keys, perfect foreign key relationships
- ✅ **Consistency**: Data types validated, NULL values within expected ranges
- ✅ **Performance**: <1s query time with indices
- ✅ **Validation**: DATABASE_VALIDATION_REPORT.md confirms EXCELLENT quality

### Validation Report

**File**: `DATABASE_VALIDATION_REPORT.md`  
**Summary**: 
- ✅ 21/21 tables present
- ✅ 0 duplicate records
- ✅ 0 orphaned foreign keys
- ✅ 100% data type consistency
- ⚠️ 51 warnings (all expected NULL values for medical data)

### Next Steps

After Phase B completes successfully:

1. ✅ **Run validation script**:
   ```bash
   python scripts/validate_database_quality.py
   ```

2. ✅ **Review validation report**:
   ```bash
   # Windows
   notepad DATABASE_VALIDATION_REPORT.md
   
   # macOS/Linux
   cat DATABASE_VALIDATION_REPORT.md
   ```

3. ✅ **Proceed to Phase C**:
   ```bash
   python src/data/cohort_labeling.py
   ```

4. ✅ **Check database size**:
   ```bash
   # Windows
   dir data\mimic_database.duckdb
   
   # macOS/Linux
   ls -lh data/mimic_database.duckdb
   ```

---

## 📊 Summary

**Phase B successfully loads all 5 datasets into a production-ready database:**

- 🎯 **21 tables** from MIMIC-IV, MIMIC-ECG, MIMIC-ECG-Ext-ICD, PTB-XL, PTB-XL+
- 📈 **456M+ rows** of clinical, ECG, and diagnostic data
- 🔍 **6 performance indices** for fast queries
- ✅ **100% validated** - no critical errors
- 🚀 **Ready for causal inference pipeline**

The database enables:
- Ground truth MI labeling (records_w_diag_icd10)
- External validation (PTB-XL)
- Complete clinical features (procedures, ICU stays, dictionaries)
- Multi-source diagnostic comparison (expert + machine interpretations)

**Total Processing Time**: ~2-3 hours  
**Output**: `data/mimic_database.duckdb` (~38.8 GB)

---

## 📚 References

- **DuckDB Documentation**: https://duckdb.org/docs/
- **MIMIC-IV Documentation**: https://mimic.mit.edu/docs/iv/
- **MIMIC-IV-ECG Documentation**: https://physionet.org/content/mimic-iv-ecg/1.0/
- **PhysioNet**: https://physionet.org/

---

## ✅ Completion Checklist

Before proceeding to Phase C, verify:

- [ ] All 9 tables loaded successfully
- [ ] All 6 indices created successfully
- [ ] Database file is ~18-20 GB
- [ ] Verification script passes (117/117 columns)
- [ ] Troponin data query returns results
- [ ] No error messages in console output

**If all checks pass**: ✅ Phase B Complete - Ready for Phase C

---

**Last Updated**: February 5, 2026  
**Script Version**: data_ingestion.py v1.0  
**Database Version**: MIMIC-IV v2.2, MIMIC-IV-ECG v1.0
