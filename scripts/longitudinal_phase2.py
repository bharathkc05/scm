"""
Longitudinal Data Foundation — Phase 2
=======================================
Builds:
  1. Lab item catalogue with coverage (lab_catalogue.parquet)
  2. Prioritized clinical factor coverage (clinical_factor_coverage.parquet)
  3. OMR outpatient measurements catalogue (omr_catalogue.parquet)
  4. ICU vitals catalogue (vitals_catalogue.parquet)

Uses DuckDB aggregation queries — does NOT load full 13.7GB labevents into memory.
"""
import duckdb
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

RAW_HOSP = r"C:\Users\Acer\Desktop\Capstone\Capstone\data\raw\MIMIC-IV-2.2\hosp"
RAW_ICU = r"C:\Users\Acer\Desktop\Capstone\Capstone\data\raw\MIMIC-IV-2.2\icu"
SCM = r"C:\Users\Acer\Desktop\Capstone\SCM"
OUT = os.path.join(SCM, "data", "longitudinal")
os.makedirs(OUT, exist_ok=True)

conn = duckdb.connect(database=":memory:")
conn.execute("SET memory_limit='4GB'")

stats = {}

# Load cohort info
labels = []
for split in ["train", "validation", "test"]:
    df = pd.read_csv(os.path.join(SCM, "data", "modeling", "labels", f"{split}.csv"))
    labels.append(df)
labels_df = pd.concat(labels, ignore_index=True)
cohort_subjects = set(labels_df["SubjectID"].unique())
cohort_hadm = set(labels_df["hadm_id"].unique())
print(f"Cohort: {len(cohort_subjects):,} patients, {len(cohort_hadm):,} hadm_ids")

# Save cohort subject IDs for DuckDB
cohort_sid = pd.DataFrame({"subject_id": list(cohort_subjects)})
cohort_sid_path = os.path.join(OUT, "_temp_cohort_sids.csv")
cohort_sid.to_csv(cohort_sid_path, index=False)

cohort_hadm_df = pd.DataFrame({"hadm_id": list(cohort_hadm)})
cohort_hadm_path = os.path.join(OUT, "_temp_cohort_hadms.csv")
cohort_hadm_df.to_csv(cohort_hadm_path, index=False)

# ============================================================
# 1. LAB ITEM DICTIONARY
# ============================================================
print("\n" + "=" * 70)
print("1. LAB ITEM DICTIONARY")
print("=" * 70)

d_lab = conn.execute(f"""
    SELECT * FROM read_csv_auto('{RAW_HOSP}/d_labitems.csv')
""").fetchdf()
print(f"  d_labitems: {len(d_lab):,} items")
print(f"  Columns: {d_lab.columns.tolist()}")
d_lab.to_parquet(os.path.join(OUT, "d_labitems.parquet"), index=False)

# ============================================================
# 2. PRIORITIZED CLINICAL FACTOR COVERAGE
# ============================================================
print("\n" + "=" * 70)
print("2. PRIORITIZED CLINICAL FACTOR COVERAGE (labevents)")
print("=" * 70)

# Define prioritized lab items by searching d_labitems
PRIORITY_LABS = {
    # Lipids
    "LDL_direct": [50197],
    "LDL_calculated": [50171],
    "HDL": [50168],
    "Total_Cholesterol": [50172],
    "Triglycerides": [50220],
    # Glycemia
    "HbA1c": [50852],
    "Glucose_blood": [50931],
    "Glucose_serum": [50809],
    # Renal
    "Creatinine": [50912],
    "BUN": [51006],
    # Electrolytes
    "Potassium": [50971, 52452],
    "Sodium": [50983, 52455],
    "Chloride": [50902],
    "Bicarbonate": [50882],
    "Calcium": [50893],
    "Magnesium": [50960],
    "Phosphate": [50970],
    # Cardiac (non-troponin)
    "BNP": [50963],
    "NT_proBNP": [50964],
    "CRP": [50889],
    # Hematology
    "Hemoglobin": [51222],
    "WBC": [51301],
    "Platelet": [51265],
    "Hematocrit": [51221],
    # Liver
    "ALT": [50861],
    "AST": [50878],
    "Albumin": [50862],
    "Bilirubin_Total": [50885],
    # Coagulation
    "INR": [51237],
    "PT": [51274],
    "PTT": [51275],
}

# First, verify these itemids exist and get their actual names
print("\n  Verifying itemids against d_labitems...")
for name, ids in PRIORITY_LABS.items():
    matches = d_lab[d_lab["itemid"].isin(ids)]
    if len(matches) == 0:
        print(f"  WARNING: {name} itemids {ids} NOT FOUND in d_labitems")
    else:
        for _, row in matches.iterrows():
            print(f"    {name}: itemid={row['itemid']}, label='{row['label']}', fluid='{row.get('fluid','?')}', category='{row.get('category','?')}'")

# Build flat list of all priority itemids
all_priority_ids = []
for ids in PRIORITY_LABS.values():
    all_priority_ids.extend(ids)
all_priority_ids_str = ",".join([str(x) for x in all_priority_ids])

# Query coverage for ALL prioritized labs in one pass
print("\n  Querying labevents coverage (single pass over 13.7GB)...")
lab_coverage = conn.execute(f"""
    WITH cohort AS (
        SELECT subject_id FROM read_csv_auto('{cohort_sid_path}')
    ),
    labs AS (
        SELECT
            le.subject_id,
            le.hadm_id,
            le.itemid,
            le.charttime::TIMESTAMP as charttime,
            le.valuenum,
            le.valueuom
        FROM read_csv_auto('{RAW_HOSP}/labevents.csv') le
        WHERE le.itemid IN ({all_priority_ids_str})
          AND le.valuenum IS NOT NULL
    )
    SELECT
        l.itemid,
        di.label as lab_name,
        di.fluid,
        di.category,
        COUNT(*) as total_measurements,
        COUNT(DISTINCT l.subject_id) as unique_patients,
        COUNT(DISTINCT l.hadm_id) as unique_admissions,
        COUNT(DISTINCT CASE WHEN c.subject_id IS NOT NULL THEN l.subject_id END) as cohort_patients,
        COUNT(CASE WHEN c.subject_id IS NOT NULL THEN 1 END) as cohort_measurements,
        MIN(l.charttime) as earliest_measurement,
        MAX(l.charttime) as latest_measurement,
        ROUND(AVG(l.valuenum), 2) as mean_value,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY l.valuenum), 2) as median_value,
        ROUND(STDDEV(l.valuenum), 2) as std_value,
        ROUND(MIN(l.valuenum), 2) as min_value,
        ROUND(MAX(l.valuenum), 2) as max_value,
        MODE(l.valueuom) as common_unit
    FROM labs l
    LEFT JOIN cohort c ON l.subject_id = c.subject_id
    LEFT JOIN read_csv_auto('{RAW_HOSP}/d_labitems.csv') di ON l.itemid = di.itemid
    GROUP BY l.itemid, di.label, di.fluid, di.category
    ORDER BY cohort_patients DESC
""").fetchdf()

print(f"\n  Lab coverage results ({len(lab_coverage)} items):")
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 40)
print(lab_coverage[["itemid", "lab_name", "total_measurements", "unique_patients",
                     "cohort_patients", "cohort_measurements", "common_unit",
                     "mean_value", "median_value"]].to_string(index=False))

lab_coverage.to_parquet(os.path.join(OUT, "lab_coverage_catalogue.parquet"), index=False)

# Map back to our categories
lab_cat_map = {}
for name, ids in PRIORITY_LABS.items():
    for iid in ids:
        lab_cat_map[iid] = name
lab_coverage["clinical_category"] = lab_coverage["itemid"].map(lab_cat_map)

stats["lab_coverage"] = {}
for _, row in lab_coverage.iterrows():
    stats["lab_coverage"][row["clinical_category"]] = {
        "itemid": int(row["itemid"]),
        "lab_name": row["lab_name"],
        "total_measurements": int(row["total_measurements"]),
        "unique_patients": int(row["unique_patients"]),
        "cohort_patients": int(row["cohort_patients"]),
        "cohort_measurements": int(row["cohort_measurements"]),
        "cohort_coverage_pct": round(100 * row["cohort_patients"] / len(cohort_subjects), 1),
    }

# ============================================================
# 3. ADMISSION-RELATIVE LAB TIMING
# ============================================================
print("\n" + "=" * 70)
print("3. LAB TIMING RELATIVE TO ADMISSION (cohort only)")
print("=" * 70)

# For the top labs, determine how many measurements are pre-admission vs during vs post
# Use a subset of most important labs
KEY_LABS = [50912, 50971, 50983, 50931, 51222, 51301, 50882, 50893]  # creat, K, Na, glucose, Hgb, WBC, bicarb, Ca
key_ids_str = ",".join([str(x) for x in KEY_LABS])

lab_timing = conn.execute(f"""
    WITH cohort AS (
        SELECT hadm_id FROM read_csv_auto('{cohort_hadm_path}')
    ),
    adm AS (
        SELECT hadm_id, admittime::TIMESTAMP as admittime, dischtime::TIMESTAMP as dischtime
        FROM read_csv_auto('{RAW_HOSP}/admissions.csv')
    ),
    labs AS (
        SELECT le.hadm_id, le.itemid, le.charttime::TIMESTAMP as charttime
        FROM read_csv_auto('{RAW_HOSP}/labevents.csv') le
        INNER JOIN cohort c ON le.hadm_id = c.hadm_id
        WHERE le.itemid IN ({key_ids_str})
          AND le.valuenum IS NOT NULL
    )
    SELECT
        l.itemid,
        di.label,
        COUNT(*) as total,
        SUM(CASE WHEN l.charttime < a.admittime THEN 1 ELSE 0 END) as before_admission,
        SUM(CASE WHEN l.charttime >= a.admittime AND l.charttime <= a.dischtime THEN 1 ELSE 0 END) as during_admission,
        SUM(CASE WHEN l.charttime > a.dischtime THEN 1 ELSE 0 END) as after_discharge
    FROM labs l
    INNER JOIN adm a ON l.hadm_id = a.hadm_id
    LEFT JOIN read_csv_auto('{RAW_HOSP}/d_labitems.csv') di ON l.itemid = di.itemid
    GROUP BY l.itemid, di.label
    ORDER BY total DESC
""").fetchdf()

print(lab_timing.to_string(index=False))

# ============================================================
# 4. OMR (OUTPATIENT) MEASUREMENTS
# ============================================================
print("\n" + "=" * 70)
print("4. OMR OUTPATIENT MEASUREMENTS")
print("=" * 70)

# First: inspect OMR schema and result_name values
omr_sample = conn.execute(f"""
    SELECT * FROM read_csv_auto('{RAW_HOSP}/omr.csv') LIMIT 10
""").fetchdf()
print(f"  OMR columns: {omr_sample.columns.tolist()}")
print(f"  Sample:")
print(omr_sample.to_string())

omr_coverage = conn.execute(f"""
    WITH cohort AS (
        SELECT subject_id FROM read_csv_auto('{cohort_sid_path}')
    )
    SELECT
        o.result_name,
        COUNT(*) as total_records,
        COUNT(DISTINCT o.subject_id) as unique_patients,
        COUNT(DISTINCT CASE WHEN c.subject_id IS NOT NULL THEN o.subject_id END) as cohort_patients,
        COUNT(CASE WHEN c.subject_id IS NOT NULL THEN 1 END) as cohort_records,
        MIN(o.chartdate) as earliest,
        MAX(o.chartdate) as latest
    FROM read_csv_auto('{RAW_HOSP}/omr.csv') o
    LEFT JOIN cohort c ON o.subject_id = c.subject_id
    GROUP BY o.result_name
    ORDER BY cohort_patients DESC
""").fetchdf()

print(f"\n  OMR result_name coverage:")
print(omr_coverage.to_string(index=False))

omr_coverage.to_parquet(os.path.join(OUT, "omr_coverage_catalogue.parquet"), index=False)

stats["omr_coverage"] = {}
for _, row in omr_coverage.iterrows():
    stats["omr_coverage"][row["result_name"]] = {
        "total_records": int(row["total_records"]),
        "unique_patients": int(row["unique_patients"]),
        "cohort_patients": int(row["cohort_patients"]),
        "cohort_records": int(row["cohort_records"]),
        "cohort_coverage_pct": round(100 * row["cohort_patients"] / len(cohort_subjects), 1),
    }

# ============================================================
# 5. ICU VITALS CATALOGUE
# ============================================================
print("\n" + "=" * 70)
print("5. ICU VITALS CATALOGUE")
print("=" * 70)

# Key vital sign itemids from d_items
VITAL_ITEMS = {
    "Heart_Rate": [220045],
    "Systolic_BP_Invasive": [220050],
    "Diastolic_BP_Invasive": [220051],
    "Mean_BP_Invasive": [220052],
    "Systolic_BP_NonInvasive": [220179],
    "Diastolic_BP_NonInvasive": [220180],
    "Mean_BP_NonInvasive": [220181],
    "Respiratory_Rate": [220210],
    "SpO2": [220277],
    "Temperature_F": [223761],
    "Temperature_C": [223762],
    "Rhythm": [220048],
    "Weight_kg": [224639],
    "Height_cm": [226730],
    "GCS_Total": [228112],
}

all_vital_ids = []
for ids in VITAL_ITEMS.values():
    all_vital_ids.extend(ids)
vital_ids_str = ",".join([str(x) for x in all_vital_ids])

print("  Querying chartevents coverage for vital signs (30.2GB file)...")
vital_coverage = conn.execute(f"""
    WITH cohort AS (
        SELECT hadm_id FROM read_csv_auto('{cohort_hadm_path}')
    )
    SELECT
        ce.itemid,
        di.label as vital_name,
        COUNT(*) as total_records,
        COUNT(DISTINCT ce.subject_id) as unique_patients,
        COUNT(DISTINCT ce.hadm_id) as unique_admissions,
        COUNT(DISTINCT CASE WHEN c.hadm_id IS NOT NULL THEN ce.hadm_id END) as cohort_admissions,
        COUNT(CASE WHEN c.hadm_id IS NOT NULL THEN 1 END) as cohort_records,
        ROUND(AVG(CASE WHEN ce.valuenum IS NOT NULL THEN ce.valuenum END), 2) as mean_value,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ce.valuenum), 2) as median_value
    FROM read_csv_auto('{RAW_ICU}/chartevents.csv') ce
    INNER JOIN read_csv_auto('{RAW_ICU}/d_items.csv') di ON ce.itemid = di.itemid
    LEFT JOIN cohort c ON ce.hadm_id = c.hadm_id
    WHERE ce.itemid IN ({vital_ids_str})
    GROUP BY ce.itemid, di.label
    ORDER BY cohort_records DESC
""").fetchdf()

print(f"\n  Vital signs coverage:")
print(vital_coverage.to_string(index=False))

vital_coverage.to_parquet(os.path.join(OUT, "vitals_coverage_catalogue.parquet"), index=False)

stats["vitals_coverage"] = {}
for _, row in vital_coverage.iterrows():
    name = [k for k, v in VITAL_ITEMS.items() if row["itemid"] in v]
    name = name[0] if name else str(row["itemid"])
    stats["vitals_coverage"][name] = {
        "itemid": int(row["itemid"]),
        "vital_name": row["vital_name"],
        "total_records": int(row["total_records"]),
        "unique_patients": int(row["unique_patients"]),
        "cohort_admissions": int(row["cohort_admissions"]),
        "cohort_records": int(row["cohort_records"]),
    }

# ============================================================
# 6. TREATMENT SOURCE PRESERVATION
# ============================================================
print("\n" + "=" * 70)
print("6. TREATMENT/PRESCRIPTION SOURCE")
print("=" * 70)

# Verify existing prescriptions source
rx_coverage = conn.execute(f"""
    WITH cohort AS (
        SELECT hadm_id FROM read_csv_auto('{cohort_hadm_path}')
    )
    SELECT
        COUNT(*) as total_rx_rows,
        COUNT(DISTINCT r.hadm_id) as unique_admissions,
        COUNT(DISTINCT r.subject_id) as unique_patients,
        COUNT(DISTINCT CASE WHEN c.hadm_id IS NOT NULL THEN r.hadm_id END) as cohort_admissions,
        SUM(CASE WHEN r.starttime IS NOT NULL THEN 1 ELSE 0 END) as with_starttime,
        SUM(CASE WHEN r.stoptime IS NOT NULL THEN 1 ELSE 0 END) as with_stoptime
    FROM read_csv_auto('{RAW_HOSP}/prescriptions.csv') r
    LEFT JOIN cohort c ON r.hadm_id = c.hadm_id
""").fetchdf()

print("  Prescriptions source coverage:")
print(rx_coverage.T.to_string())

# EMAR coverage already documented — just note it
print("  EMAR source: documented in EMAR_RECONSTRUCTION_DECISION.md (26.8M rows)")

stats["prescriptions"] = {
    "total_rows": int(rx_coverage.iloc[0]["total_rx_rows"]),
    "unique_admissions": int(rx_coverage.iloc[0]["unique_admissions"]),
    "unique_patients": int(rx_coverage.iloc[0]["unique_patients"]),
    "cohort_admissions": int(rx_coverage.iloc[0]["cohort_admissions"]),
    "with_starttime": int(rx_coverage.iloc[0]["with_starttime"]),
    "with_stoptime": int(rx_coverage.iloc[0]["with_stoptime"]),
}

# Clean up temp files
for f in ["_temp_cohort_sids.csv", "_temp_cohort_hadms.csv"]:
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        os.remove(p)

# Save stats
with open(os.path.join(OUT, "phase2_stats.json"), "w") as f:
    json.dump(stats, f, indent=2, default=str)

print(f"\n{'='*70}")
print("PHASE 2 COMPLETE")
print(f"{'='*70}")
for fn in sorted(os.listdir(OUT)):
    fp = os.path.join(OUT, fn)
    if os.path.isfile(fp):
        sz = os.path.getsize(fp)
        print(f"  {fn}: {sz/1e6:.1f} MB")

conn.close()
