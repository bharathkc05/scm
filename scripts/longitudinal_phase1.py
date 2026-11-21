"""
Longitudinal Data Foundation — Phase 1
=======================================
Builds:
  1. Complete admission chronology (admissions_chronology.parquet)
  2. Patient demographics (patients_demographics.parquet)
  3. MI/diagnosis history (diagnoses_mi.parquet)
  4. Death/observation end (observation_end.parquet)

Output: data/longitudinal/
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
# Increase memory for large joins
conn.execute("SET memory_limit='4GB'")

stats = {}  # collect audit stats

# ============================================================
# 1. COMPLETE ADMISSION CHRONOLOGY
# ============================================================
print("=" * 70)
print("1. COMPLETE ADMISSION CHRONOLOGY")
print("=" * 70)

adm = conn.execute(f"""
    SELECT
        subject_id,
        hadm_id,
        admittime::TIMESTAMP as admittime,
        dischtime::TIMESTAMP as dischtime,
        deathtime::TIMESTAMP as deathtime,
        edregtime::TIMESTAMP as edregtime,
        edouttime::TIMESTAMP as edouttime,
        admission_type,
        admission_location,
        discharge_location,
        insurance,
        race,
        hospital_expire_flag
    FROM read_csv_auto('{RAW_HOSP}/admissions.csv')
    ORDER BY subject_id, admittime
""").fetchdf()

print(f"  Total admissions: {len(adm):,}")
print(f"  Total patients: {adm['subject_id'].nunique():,}")

# Compute chronological sequence
adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
adm["admission_seq"] = adm.groupby("subject_id").cumcount() + 1

# Previous/next admission linkage
adm["previous_hadm_id"] = adm.groupby("subject_id")["hadm_id"].shift(1)
adm["previous_dischtime"] = adm.groupby("subject_id")["dischtime"].shift(1)
adm["next_hadm_id"] = adm.groupby("subject_id")["hadm_id"].shift(-1)
adm["next_admittime"] = adm.groupby("subject_id")["admittime"].shift(-1)

# Anchor time (consistent with existing project)
adm["anchor_time"] = adm["edregtime"].fillna(adm["admittime"])

# Admission count per patient
adm_counts = adm.groupby("subject_id")["hadm_id"].count().reset_index()
adm_counts.columns = ["subject_id", "total_admissions"]
adm = adm.merge(adm_counts, on="subject_id")

# Quality checks
bad_order = (adm["dischtime"] < adm["admittime"]).sum()
print(f"  Discharge before admission errors: {bad_order}")

multi_adm = (adm["total_admissions"] > 1).sum()
single_adm = (adm["total_admissions"] == 1).sum()
print(f"  Single-admission patients: {adm.loc[adm['admission_seq']==1, 'total_admissions'].eq(1).sum():,}")
print(f"  Multi-admission patients: {adm.loc[adm['admission_seq']==1, 'total_admissions'].gt(1).sum():,}")
print(f"  Admissions with deathtime: {adm['deathtime'].notna().sum():,}")
print(f"  Admissions with edregtime: {adm['edregtime'].notna().sum():,}")

adm.to_parquet(os.path.join(OUT, "admissions_chronology.parquet"), index=False)
print(f"  Saved: admissions_chronology.parquet ({len(adm):,} rows)")

stats["admissions"] = {
    "total_admissions": int(len(adm)),
    "total_patients": int(adm["subject_id"].nunique()),
    "single_admission_patients": int(adm.loc[adm['admission_seq']==1, 'total_admissions'].eq(1).sum()),
    "multi_admission_patients": int(adm.loc[adm['admission_seq']==1, 'total_admissions'].gt(1).sum()),
    "with_deathtime": int(adm['deathtime'].notna().sum()),
    "with_edregtime": int(adm['edregtime'].notna().sum()),
    "discharge_before_admit_errors": int(bad_order),
}

# ============================================================
# 2. PATIENT DEMOGRAPHICS
# ============================================================
print("\n" + "=" * 70)
print("2. PATIENT DEMOGRAPHICS")
print("=" * 70)

patients = conn.execute(f"""
    SELECT
        subject_id,
        gender,
        anchor_age,
        anchor_year,
        anchor_year_group,
        dod
    FROM read_csv_auto('{RAW_HOSP}/patients.csv')
""").fetchdf()

print(f"  Total patients: {len(patients):,}")
print(f"  Gender distribution: {patients['gender'].value_counts().to_dict()}")
print(f"  Anchor age range: {patients['anchor_age'].min()} - {patients['anchor_age'].max()}")
print(f"  Patients with dod (date of death): {patients['dod'].notna().sum():,}")

patients.to_parquet(os.path.join(OUT, "patients_demographics.parquet"), index=False)
print(f"  Saved: patients_demographics.parquet ({len(patients):,} rows)")

stats["patients"] = {
    "total_patients": int(len(patients)),
    "with_dod": int(patients['dod'].notna().sum()),
    "gender_M": int((patients['gender'] == 'M').sum()),
    "gender_F": int((patients['gender'] == 'F').sum()),
    "age_min": int(patients['anchor_age'].min()),
    "age_max": int(patients['anchor_age'].max()),
    "age_median": int(patients['anchor_age'].median()),
}

# ============================================================
# 3. MI / DIAGNOSIS HISTORY
# ============================================================
print("\n" + "=" * 70)
print("3. MI / DIAGNOSIS HISTORY")
print("=" * 70)

# Get ALL ICD diagnoses
diag = conn.execute(f"""
    SELECT
        d.subject_id,
        d.hadm_id,
        d.icd_code,
        d.icd_version,
        d.seq_num,
        a.admittime::TIMESTAMP as admittime,
        a.dischtime::TIMESTAMP as dischtime,
        CASE
            WHEN d.icd_version = 10 AND (
                d.icd_code LIKE 'I21%' OR d.icd_code LIKE 'I22%' OR d.icd_code LIKE 'I23%'
            ) THEN 1
            WHEN d.icd_version = 9 AND (
                d.icd_code LIKE '410%' OR d.icd_code LIKE '412%'
            ) THEN 1
            ELSE 0
        END as is_mi_code
    FROM read_csv_auto('{RAW_HOSP}/diagnoses_icd.csv') d
    INNER JOIN read_csv_auto('{RAW_HOSP}/admissions.csv') a
        ON d.hadm_id = a.hadm_id
    ORDER BY d.subject_id, a.admittime, d.seq_num
""").fetchdf()

print(f"  Total diagnosis rows: {len(diag):,}")
print(f"  Admissions with any MI code: {diag[diag['is_mi_code']==1]['hadm_id'].nunique():,}")
print(f"  Patients with any MI code: {diag[diag['is_mi_code']==1]['subject_id'].nunique():,}")

# Save MI-specific diagnoses
mi_diag = diag[diag["is_mi_code"] == 1].copy()
print(f"  MI diagnosis rows: {len(mi_diag):,}")

# Build admission-level MI flag
mi_admissions = mi_diag.groupby(["subject_id", "hadm_id"]).agg(
    mi_icd_codes=("icd_code", lambda x: ";".join(sorted(set(x)))),
    min_seq_num=("seq_num", "min"),
    admittime=("admittime", "first"),
).reset_index()
mi_admissions = mi_admissions.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
mi_admissions["mi_admission_seq"] = mi_admissions.groupby("subject_id").cumcount() + 1
mi_admissions["total_mi_admissions"] = mi_admissions.groupby("subject_id")["hadm_id"].transform("count")
mi_admissions["is_first_mi_admission"] = mi_admissions["mi_admission_seq"] == 1

print(f"  MI admissions: {len(mi_admissions):,}")
print(f"  Patients with 1 MI admission: {(mi_admissions['total_mi_admissions']==1).sum():,}")
print(f"  Patients with >1 MI admission: {mi_admissions[mi_admissions['mi_admission_seq']==1]['total_mi_admissions'].gt(1).sum():,}")

mi_admissions.to_parquet(os.path.join(OUT, "mi_admissions.parquet"), index=False)
mi_diag.to_parquet(os.path.join(OUT, "diagnoses_mi_codes.parquet"), index=False)

# Save full diagnoses (compressed)
diag.to_parquet(os.path.join(OUT, "diagnoses_all.parquet"), index=False)
print(f"  Saved: mi_admissions.parquet, diagnoses_mi_codes.parquet, diagnoses_all.parquet")

stats["diagnoses"] = {
    "total_diagnosis_rows": int(len(diag)),
    "mi_diagnosis_rows": int(len(mi_diag)),
    "admissions_with_mi": int(mi_admissions['hadm_id'].nunique()),
    "patients_with_mi": int(mi_admissions['subject_id'].nunique()),
    "patients_single_mi": int((mi_admissions.groupby('subject_id')['hadm_id'].count()==1).sum()),
    "patients_multi_mi": int((mi_admissions.groupby('subject_id')['hadm_id'].count()>1).sum()),
}

# ============================================================
# 4. OBSERVATION END / DEATH
# ============================================================
print("\n" + "=" * 70)
print("4. OBSERVATION END / DEATH")
print("=" * 70)

# Build observation-end table:
# - dod from patients table (date of death, may be out-of-hospital)
# - deathtime from admissions (in-hospital death)
# - last known admission discharge as censoring time

obs_end = conn.execute(f"""
    WITH last_adm AS (
        SELECT subject_id,
               MAX(dischtime::TIMESTAMP) as last_discharge,
               MAX(admittime::TIMESTAMP) as last_admittime,
               MAX(hadm_id) as last_hadm_id
        FROM read_csv_auto('{RAW_HOSP}/admissions.csv')
        GROUP BY subject_id
    ),
    in_hosp_death AS (
        SELECT subject_id,
               MAX(deathtime::TIMESTAMP) as in_hospital_deathtime
        FROM read_csv_auto('{RAW_HOSP}/admissions.csv')
        WHERE deathtime IS NOT NULL
        GROUP BY subject_id
    ),
    patient_dod AS (
        SELECT subject_id, dod
        FROM read_csv_auto('{RAW_HOSP}/patients.csv')
    )
    SELECT
        la.subject_id,
        la.last_discharge,
        la.last_admittime,
        ihd.in_hospital_deathtime,
        pd.dod,
        CASE
            WHEN pd.dod IS NOT NULL THEN 1
            WHEN ihd.in_hospital_deathtime IS NOT NULL THEN 1
            ELSE 0
        END as is_dead,
        COALESCE(pd.dod::VARCHAR, ihd.in_hospital_deathtime::VARCHAR, la.last_discharge::VARCHAR) as observation_end
    FROM last_adm la
    LEFT JOIN in_hosp_death ihd ON la.subject_id = ihd.subject_id
    LEFT JOIN patient_dod pd ON la.subject_id = pd.subject_id
    ORDER BY la.subject_id
""").fetchdf()

print(f"  Patients total: {len(obs_end):,}")
print(f"  Patients with dod: {obs_end['dod'].notna().sum():,}")
print(f"  Patients with in-hospital death: {obs_end['in_hospital_deathtime'].notna().sum():,}")
print(f"  Patients known dead: {obs_end['is_dead'].sum():,}")
print(f"  Patients alive/censored: {(obs_end['is_dead']==0).sum():,}")

obs_end.to_parquet(os.path.join(OUT, "observation_end.parquet"), index=False)
print(f"  Saved: observation_end.parquet")

stats["observation_end"] = {
    "total_patients": int(len(obs_end)),
    "with_dod": int(obs_end['dod'].notna().sum()),
    "with_in_hospital_death": int(obs_end['in_hospital_deathtime'].notna().sum()),
    "known_dead": int(obs_end['is_dead'].sum()),
    "alive_or_censored": int((obs_end['is_dead']==0).sum()),
}

# ============================================================
# 5. LINK TO EXISTING COHORT
# ============================================================
print("\n" + "=" * 70)
print("5. LINK TO EXISTING COHORT")
print("=" * 70)

# Load existing master cohort identifiers
labels = []
for split in ["train", "validation", "test"]:
    df = pd.read_csv(os.path.join(SCM, "data", "modeling", "labels", f"{split}.csv"))
    df["split"] = split
    labels.append(df)
labels_df = pd.concat(labels, ignore_index=True)

cohort_subjects = set(labels_df["SubjectID"].unique())
cohort_hadm = set(labels_df["hadm_id"].unique())

# How many cohort patients have multi-admission history?
cohort_adm = adm[adm["subject_id"].isin(cohort_subjects)]
print(f"  Cohort patients: {len(cohort_subjects):,}")
print(f"  Cohort hadm_ids in master: {len(cohort_hadm):,}")
print(f"  Cohort patients' total admissions: {len(cohort_adm):,}")
print(f"  Cohort single-admission patients: {cohort_adm.groupby('subject_id')['hadm_id'].nunique().eq(1).sum():,}")
print(f"  Cohort multi-admission patients: {cohort_adm.groupby('subject_id')['hadm_id'].nunique().gt(1).sum():,}")

# How many cohort patients have MI in ANY admission?
cohort_mi = mi_admissions[mi_admissions["subject_id"].isin(cohort_subjects)]
print(f"  Cohort patients with any MI admission: {cohort_mi['subject_id'].nunique():,}")
print(f"  Cohort MI admissions total: {len(cohort_mi):,}")

# How many have MI in a SUBSEQUENT admission (after a non-MI admission)?
# This is the key longitudinal question
cohort_adm_mi = cohort_adm.merge(
    mi_admissions[["hadm_id"]].assign(has_mi=1),
    on="hadm_id", how="left"
)
cohort_adm_mi["has_mi"] = cohort_adm_mi["has_mi"].fillna(0).astype(int)

# For each patient, find if there's a non-MI admission followed by an MI admission
subsequent_mi_count = 0
subsequent_mi_patients = set()
for sid, grp in cohort_adm_mi.groupby("subject_id"):
    grp = grp.sort_values("admittime")
    mi_idx = grp[grp["has_mi"] == 1].index
    non_mi_idx = grp[grp["has_mi"] == 0].index
    if len(mi_idx) > 0 and len(non_mi_idx) > 0:
        earliest_non_mi = grp.loc[non_mi_idx, "admittime"].min()
        later_mi = grp[(grp["has_mi"] == 1) & (grp["admittime"] > earliest_non_mi)]
        if len(later_mi) > 0:
            subsequent_mi_count += len(later_mi)
            subsequent_mi_patients.add(sid)

print(f"  Cohort patients with non-MI admission THEN later MI admission: {len(subsequent_mi_patients):,}")
print(f"  Total subsequent-MI admissions: {subsequent_mi_count:,}")

stats["cohort_linkage"] = {
    "cohort_patients": int(len(cohort_subjects)),
    "cohort_hadm_ids": int(len(cohort_hadm)),
    "total_admissions_for_cohort": int(len(cohort_adm)),
    "single_admission_patients": int(cohort_adm.groupby('subject_id')['hadm_id'].nunique().eq(1).sum()),
    "multi_admission_patients": int(cohort_adm.groupby('subject_id')['hadm_id'].nunique().gt(1).sum()),
    "patients_with_any_mi": int(cohort_mi['subject_id'].nunique()),
    "patients_nonmi_then_later_mi": int(len(subsequent_mi_patients)),
    "subsequent_mi_admissions": int(subsequent_mi_count),
}

# Save stats
with open(os.path.join(OUT, "phase1_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n{'='*70}")
print("PHASE 1 COMPLETE")
print(f"{'='*70}")
print(f"Output directory: {OUT}")
for fn in sorted(os.listdir(OUT)):
    fp = os.path.join(OUT, fn)
    sz = os.path.getsize(fp)
    print(f"  {fn}: {sz/1e6:.1f} MB")

conn.close()
