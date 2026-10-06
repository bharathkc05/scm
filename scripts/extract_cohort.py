import duckdb
import os
import pandas as pd
import ast

def main():
    # ═════════════════════════════════════════════════════════════
    # CONFIGURATION & PATHS
    # ═════════════════════════════════════════════════════════════
    RAW_DIR = r"c:\Users\Acer\Desktop\Capstone\Capstone\data\raw"
    PROCESSED_DIR = os.path.join(RAW_DIR, "processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"Starting extraction pipeline...\nRaw Data: {RAW_DIR}\nOutput: {PROCESSED_DIR}")

    # Initialize in-memory DuckDB
    conn = duckdb.connect(database=':memory:')

    # ═════════════════════════════════════════════════════════════
    # STEP 1: BASE COHORT & DYNAMIC TEMPORAL BOUNDARIES
    # ═════════════════════════════════════════════════════════════
    print("1. Building Base Cohort and calculating dynamic 7-day temporal windows...")
    
    conn.execute(f"""
    CREATE TABLE cohort_base AS 
    WITH ecg_cohort AS (
        SELECT 
            CAST(COALESCE(hosp_hadm_id, ed_hadm_id) AS BIGINT) AS hadm_id,
            CAST(subject_id AS BIGINT) AS subject_id,
            ANY_VALUE(gender) AS gender,
            ANY_VALUE(age) AS age_at_event,
            ANY_VALUE(all_diag_all) AS diag_list,
            MIN(ecg_time) AS first_ecg_time
        FROM read_csv_auto('{RAW_DIR}/MIMIC-IV-ECG-Ext-ICD-1.0.1/records_w_diag_icd10.csv')
        WHERE ecg_taken_in_ed_or_hosp = true 
          AND COALESCE(hosp_hadm_id, ed_hadm_id) IS NOT NULL
          AND all_diag_all != '[]'
        GROUP BY CAST(COALESCE(hosp_hadm_id, ed_hadm_id) AS BIGINT), CAST(subject_id AS BIGINT)
    ),
    admissions AS (
        SELECT 
            hadm_id,
            COALESCE(edregtime, admittime) AS anchor_time,
            dischtime,
            race
        FROM read_csv_auto('{RAW_DIR}/MIMIC-IV-2.2/hosp/admissions.csv')
    ),
    joined AS (
        SELECT 
            e.hadm_id, e.subject_id, e.gender, e.age_at_event, e.diag_list,
            COALESCE(a.anchor_time, CAST(e.first_ecg_time AS TIMESTAMP)) AS anchor_time,
            COALESCE(a.dischtime, CAST(e.first_ecg_time AS TIMESTAMP) + INTERVAL 1 DAY) AS dischtime,
            a.race,
            CASE WHEN a.anchor_time IS NOT NULL THEN TRUE ELSE FALSE END AS has_hosp_admission
        FROM ecg_cohort e
        LEFT JOIN admissions a ON e.hadm_id = a.hadm_id
    ),
    encounter_history AS (
        SELECT 
            *,
            LAG(dischtime) OVER (PARTITION BY subject_id ORDER BY anchor_time) AS prev_dischtime,
            LEAD(anchor_time) OVER (PARTITION BY subject_id ORDER BY anchor_time) AS next_anchor_time,
            CASE WHEN LAG(hadm_id) OVER (PARTITION BY subject_id ORDER BY anchor_time) IS NOT NULL 
                 THEN TRUE ELSE FALSE END AS had_prior_admission
        FROM joined
    )
    SELECT 
        hadm_id, subject_id, gender, age_at_event, race, diag_list, 
        anchor_time, dischtime, had_prior_admission, has_hosp_admission,
        
        CASE 
            WHEN prev_dischtime IS NOT NULL AND prev_dischtime > anchor_time - INTERVAL 7 DAY 
            THEN prev_dischtime 
            ELSE anchor_time - INTERVAL 7 DAY 
        END AS window_before_start,
        
        CASE 
            WHEN next_anchor_time IS NOT NULL AND next_anchor_time < dischtime + INTERVAL 7 DAY 
            THEN next_anchor_time 
            ELSE dischtime + INTERVAL 7 DAY 
        END AS window_after_end
    FROM encounter_history;
    """)

    row_count = conn.execute("SELECT COUNT(*) FROM cohort_base").fetchone()[0]
    matched_count = conn.execute("SELECT COUNT(*) FROM cohort_base WHERE has_hosp_admission = TRUE").fetchone()[0]
    print(f"   Base cohort established with {row_count} unique admissions.")
    print(f"   -> {matched_count} matched to hospital admissions ({row_count - matched_count} ED-only fallbacks)")

    # ═════════════════════════════════════════════════════════════
    # STEP 2: PARSE ICD LABELS
    # ═════════════════════════════════════════════════════════════
    print("2. Parsing MI type labels...")
    
    df_cohort = conn.execute("SELECT * FROM cohort_base").df()

    parse_errors = 0
    def classify_mi(diag_str):
        nonlocal parse_errors
        try:
            codes = set(str(c) for c in ast.literal_eval(diag_str))
        except:
            parse_errors += 1
            codes = set()
        
        type1 = any(c.startswith(('I210','I211','I212','I213','I214','I219')) for c in codes) or 'I21' in codes
        complication = any(c.startswith('I23') for c in codes)
        type2 = 'I21A1' in codes
        subsequent = any(c.startswith('I22') for c in codes)
        other = 'I21A' in codes or 'I21A9' in codes
        
        is_mi = type1 or type2 or subsequent or other or complication
        return pd.Series([is_mi, type1, type2, subsequent, other, complication])

    print("   Applying classification logic...")
    df_cohort[['is_MI', 'type_1_MI', 'type_2_MI', 'subsequent_MI', 'other_MI', 'complication_mi']] = df_cohort['diag_list'].apply(classify_mi)
    df_cohort.drop(columns=['diag_list'], inplace=True)
    
    if parse_errors > 0:
        print(f"   ⚠️ WARNING: {parse_errors} rows failed to parse diagnosis codes.")

    labels_path = os.path.join(PROCESSED_DIR, "cohort_labels.csv")
    df_cohort.to_csv(labels_path, index=False)
    print(f"   Saved -> {labels_path}")

    conn.execute(f"CREATE TABLE final_cohort AS SELECT * FROM read_csv_auto('{labels_path}')")

    # ═════════════════════════════════════════════════════════════
    # STEP 3: TREATMENTS
    # ═════════════════════════════════════════════════════════════
    print("3. Extracting Treatments (Statins, Vasopressors, Ventilation)...")
    print("   Note: 'before' meds only capture recent INPATIENT prescriptions, not true outpatient baselines.")
    
    conn.execute(f"""
    CREATE TABLE features_treatments AS
    WITH meds AS (
        SELECT 
            c.hadm_id,
            MAX(CASE WHEN LOWER(p.drug) LIKE '%vastatin%' AND p.starttime BETWEEN c.window_before_start AND c.anchor_time THEN 1 ELSE 0 END) AS statin_before,
            MAX(CASE WHEN LOWER(p.drug) LIKE '%vastatin%' AND p.starttime BETWEEN c.anchor_time AND c.dischtime THEN 1 ELSE 0 END) AS statin_during,
            MAX(CASE WHEN LOWER(p.drug) LIKE '%vastatin%' AND p.starttime BETWEEN c.dischtime AND c.window_after_end THEN 1 ELSE 0 END) AS statin_after,
            
            MAX(CASE WHEN (LOWER(p.drug) LIKE '%olol%' OR LOWER(p.drug) LIKE '%carvedilol%') AND p.starttime BETWEEN c.window_before_start AND c.anchor_time THEN 1 ELSE 0 END) AS beta_blocker_before,
            MAX(CASE WHEN (LOWER(p.drug) LIKE '%olol%' OR LOWER(p.drug) LIKE '%carvedilol%') AND p.starttime BETWEEN c.anchor_time AND c.dischtime THEN 1 ELSE 0 END) AS beta_blocker_during,
            
            MAX(CASE WHEN LOWER(p.drug) LIKE '%pril%' AND p.starttime BETWEEN c.window_before_start AND c.anchor_time THEN 1 ELSE 0 END) AS acei_before,
            MAX(CASE WHEN LOWER(p.drug) LIKE '%pril%' AND p.starttime BETWEEN c.anchor_time AND c.dischtime THEN 1 ELSE 0 END) AS acei_during,
            
            MAX(CASE WHEN LOWER(p.drug) LIKE '%aspirin%' AND p.starttime BETWEEN c.window_before_start AND c.anchor_time THEN 1 ELSE 0 END) AS aspirin_before,
            MAX(CASE WHEN LOWER(p.drug) LIKE '%aspirin%' AND p.starttime BETWEEN c.anchor_time AND c.dischtime THEN 1 ELSE 0 END) AS aspirin_during

        FROM final_cohort c
        LEFT JOIN read_csv_auto('{RAW_DIR}/MIMIC-IV-2.2/hosp/prescriptions.csv') p 
            ON c.subject_id = p.subject_id AND c.hadm_id = p.hadm_id
        GROUP BY c.hadm_id
    ),
    vasopressors AS (
        SELECT 
            c.hadm_id,
            MAX(CASE WHEN i.starttime BETWEEN c.anchor_time AND c.dischtime THEN 1 ELSE 0 END) AS vasopressor_during
        FROM final_cohort c
        LEFT JOIN read_csv_auto('{RAW_DIR}/MIMIC-IV-2.2/icu/inputevents.csv') i 
            ON c.subject_id = i.subject_id AND c.hadm_id = i.hadm_id
            AND i.itemid IN (221906, 221289, 221662, 222315, 221653)
        GROUP BY c.hadm_id
    ),
    ventilation AS (
        SELECT 
            c.hadm_id,
            MAX(CASE WHEN p.starttime BETWEEN c.anchor_time AND c.dischtime THEN 1 ELSE 0 END) AS mech_vent_during
        FROM final_cohort c
        LEFT JOIN read_csv_auto('{RAW_DIR}/MIMIC-IV-2.2/icu/procedureevents.csv') p 
            ON c.subject_id = p.subject_id AND c.hadm_id = p.hadm_id
            AND p.itemid IN (225792, 225794, 224385)
        GROUP BY c.hadm_id
    )
    SELECT 
        m.*, 
        COALESCE(v.vasopressor_during, 0) AS vasopressor_during, 
        COALESCE(ve.mech_vent_during, 0) AS mech_vent_during
    FROM meds m
    LEFT JOIN vasopressors v ON m.hadm_id = v.hadm_id
    LEFT JOIN ventilation ve ON m.hadm_id = ve.hadm_id;
    """)
    conn.execute(f"COPY features_treatments TO '{PROCESSED_DIR}/features_treatments.csv' (HEADER, DELIMITER ',')")
    print(f"   Saved -> {PROCESSED_DIR}\\features_treatments.csv")


    # ═════════════════════════════════════════════════════════════
    # STEP 4: LAB VALUES
    # ═════════════════════════════════════════════════════════════
    print("4. Extracting Labs...")
    conn.execute(f"""
    CREATE TABLE features_labs AS
    SELECT 
        c.hadm_id,
        MAX(CASE WHEN l.itemid IN (50905, 50906) AND l.charttime BETWEEN c.window_before_start AND c.anchor_time THEN l.valuenum END) AS ldl_before,
        MAX(CASE WHEN l.itemid = 50904 AND l.charttime BETWEEN c.window_before_start AND c.anchor_time THEN l.valuenum END) AS hdl_before,
        
        MAX(CASE WHEN l.itemid IN (51002, 51003, 52642) AND l.charttime BETWEEN c.anchor_time AND c.dischtime THEN l.valuenum END) AS peak_troponin_during,
        MAX(CASE WHEN l.itemid = 50912 AND l.charttime BETWEEN c.anchor_time AND c.dischtime THEN l.valuenum END) AS peak_creatinine_during,
        MAX(CASE WHEN l.itemid = 51006 AND l.charttime BETWEEN c.anchor_time AND c.dischtime THEN l.valuenum END) AS peak_bun_during,
        MAX(CASE WHEN l.itemid = 50931 AND l.charttime BETWEEN c.anchor_time AND c.dischtime THEN l.valuenum END) AS peak_glucose_during,
        MAX(CASE WHEN l.itemid = 50971 AND l.charttime BETWEEN c.anchor_time AND c.dischtime THEN l.valuenum END) AS peak_potassium_during,
        MAX(CASE WHEN l.itemid = 50963 AND l.charttime BETWEEN c.anchor_time AND c.dischtime THEN l.valuenum END) AS peak_bnp_during
    FROM final_cohort c
    LEFT JOIN read_csv_auto('{RAW_DIR}/MIMIC-IV-2.2/hosp/labevents.csv') l 
        ON c.subject_id = l.subject_id AND c.hadm_id = l.hadm_id
        AND l.itemid IN (50904, 50905, 50906, 51002, 51003, 52642, 50912, 51006, 50931, 50971, 50963)
    GROUP BY c.hadm_id;
    """)
    conn.execute(f"COPY features_labs TO '{PROCESSED_DIR}/features_labs.csv' (HEADER, DELIMITER ',')")
    print(f"   Saved -> {PROCESSED_DIR}\\features_labs.csv")


    # ═════════════════════════════════════════════════════════════
    # STEP 5: VITALS
    # ═════════════════════════════════════════════════════════════
    print("5. Extracting Vitals (Scanning 30GB chartevents file)...")
    conn.execute(f"""
    CREATE TABLE features_vitals AS
    WITH raw_vitals AS (
        SELECT 
            v.subject_id, v.hadm_id, v.itemid, v.charttime,
            CASE WHEN v.itemid = 223761 THEN (v.valuenum - 32) * 5/9 ELSE v.valuenum END AS valuenum
        FROM read_csv_auto('{RAW_DIR}/MIMIC-IV-2.2/icu/chartevents.csv') v
        WHERE v.itemid IN (220045, 220277, 223761, 223762, 220210, 224690, 220179, 220050)
    )
    SELECT 
        c.hadm_id,
        MAX(CASE WHEN v.itemid = 220045 AND v.charttime BETWEEN c.anchor_time AND c.dischtime THEN v.valuenum END) AS max_hr_during,
        MIN(CASE WHEN v.itemid = 220277 AND v.charttime BETWEEN c.anchor_time AND c.dischtime THEN v.valuenum END) AS min_spo2_during,
        MAX(CASE WHEN v.itemid IN (223761, 223762) AND v.charttime BETWEEN c.anchor_time AND c.dischtime THEN v.valuenum END) AS max_temp_celsius_during,
        MAX(CASE WHEN v.itemid IN (220210, 224690) AND v.charttime BETWEEN c.anchor_time AND c.dischtime THEN v.valuenum END) AS max_rr_during,
        MIN(CASE WHEN v.itemid IN (220179, 220050) AND v.charttime BETWEEN c.anchor_time AND c.dischtime THEN v.valuenum END) AS min_sbp_during
    FROM final_cohort c
    LEFT JOIN raw_vitals v 
        ON c.subject_id = v.subject_id AND c.hadm_id = v.hadm_id
    GROUP BY c.hadm_id;
    """)
    
    conn.execute(f"""
    CREATE TABLE omr_vitals AS
    SELECT 
        c.hadm_id,
        arg_max(
            TRY_CAST(SPLIT_PART(o.result_value, '/', 1) AS DOUBLE), 
            o.chartdate
        ) AS baseline_sys_bp_before,
        arg_max(
            TRY_CAST(SPLIT_PART(o.result_value, '/', 2) AS DOUBLE), 
            o.chartdate
        ) AS baseline_dia_bp_before
    FROM final_cohort c
    LEFT JOIN read_csv_auto('{RAW_DIR}/MIMIC-IV-2.2/hosp/omr.csv') o 
        ON c.subject_id = o.subject_id
        AND o.result_name = 'Blood Pressure' 
        AND o.chartdate BETWEEN c.window_before_start AND c.anchor_time
    GROUP BY c.hadm_id;
    """)
    
    conn.execute(f"""
    CREATE TABLE final_vitals AS
    SELECT v.*, o.baseline_sys_bp_before, o.baseline_dia_bp_before
    FROM features_vitals v
    LEFT JOIN omr_vitals o ON v.hadm_id = o.hadm_id;
    """)
    conn.execute(f"COPY final_vitals TO '{PROCESSED_DIR}/features_vitals.csv' (HEADER, DELIMITER ',')")
    print(f"   Saved -> {PROCESSED_DIR}\\features_vitals.csv")


    # ═════════════════════════════════════════════════════════════
    # STEP 6: ECG MAPPING
    # ═════════════════════════════════════════════════════════════
    print("6. Creating ECG Linkage Map...")
    conn.execute(f"""
    CREATE TABLE ecg_mapping AS
    SELECT 
        CAST(COALESCE(hosp_hadm_id, ed_hadm_id) AS BIGINT) AS hadm_id,
        CAST(subject_id AS BIGINT) AS subject_id,
        CAST(study_id AS BIGINT) AS study_id,
        file_name AS waveform_path,
        ecg_time
    FROM read_csv_auto('{RAW_DIR}/MIMIC-IV-ECG-Ext-ICD-1.0.1/records_w_diag_icd10.csv')
    WHERE ecg_taken_in_ed_or_hosp = true 
      AND COALESCE(hosp_hadm_id, ed_hadm_id) IS NOT NULL
    """)
    conn.execute(f"COPY ecg_mapping TO '{PROCESSED_DIR}/ecg_mapping.csv' (HEADER, DELIMITER ',')")
    print(f"   Saved -> {PROCESSED_DIR}\\ecg_mapping.csv")
    print("\n[SUCCESS] Extraction Complete! The data/raw/processed folder is ready.")

if __name__ == "__main__":
    main()
