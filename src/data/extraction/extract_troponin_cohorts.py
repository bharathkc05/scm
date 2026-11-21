import duckdb
import pandas as pd
import numpy as np

def extract_troponin():
    RAW_DIR = r"C:\Users\Acer\Desktop\Capstone\Capstone\data\raw"
    
    conn = duckdb.connect(database=':memory:')
    
    # 1. Setup thresholds table
    print("Setting up troponin thresholds...")
    thresholds_df = pd.DataFrame([
        {'itemid': 51003, 'assay_type': 'cTnT', 'url_value': 0.014}, # Troponin T
        {'itemid': 51002, 'assay_type': 'cTnI', 'url_value': 0.04},  # Troponin I
        {'itemid': 52642, 'assay_type': 'cTnI', 'url_value': 0.04}   # Troponin I (high sens)
    ])
    thresholds_df.to_csv('temp_thresholds.csv', index=False)
    conn.execute("CREATE TABLE troponin_thresholds AS SELECT * FROM read_csv_auto('temp_thresholds.csv')")
    
    # We only care about patients who are in our cohort.
    cohort = pd.read_csv(r'C:\Users\Acer\Desktop\Pipeline\cohort_labels.csv', usecols=['hadm_id', 'subject_id', 'is_MI', 'anchor_time'])
    cohort.to_csv('temp_cohort.csv', index=False)
    conn.execute("CREATE TABLE cohort AS SELECT * FROM read_csv_auto('temp_cohort.csv')")
    
    print("Querying raw labevents with window functions (this may take a minute)...")
    query = f"""
    WITH troponin_measurements AS (
        SELECT 
            le.hadm_id,
            le.subject_id,
            le.itemid,
            le.charttime,
            le.valuenum,
            tt.url_value,
            tt.assay_type,
            LAG(le.valuenum) OVER (PARTITION BY le.hadm_id, le.itemid ORDER BY le.charttime) as prev_value,
            LEAD(le.valuenum) OVER (PARTITION BY le.hadm_id, le.itemid ORDER BY le.charttime) as next_value
        FROM read_csv_auto('{RAW_DIR}/MIMIC-IV-2.2/hosp/labevents.csv') le
        INNER JOIN cohort c ON le.hadm_id = c.hadm_id
        INNER JOIN troponin_thresholds tt ON le.itemid = tt.itemid
        WHERE le.valuenum IS NOT NULL
    ),
    elevated_with_delta AS (
        SELECT 
            *,
            CASE WHEN valuenum > url_value THEN 1 ELSE 0 END as is_elevated,
            CASE WHEN prev_value > 0 THEN ABS((valuenum - prev_value) / prev_value * 100.0) ELSE NULL END as pct_change_from_prev,
            CASE WHEN valuenum > 0 THEN ABS((next_value - valuenum) / valuenum * 100.0) ELSE NULL END as pct_change_to_next
        FROM troponin_measurements
    ),
    acute_mi_events AS (
        SELECT 
            hadm_id,
            subject_id,
            charttime,
            valuenum,
            url_value,
            CASE 
                WHEN valuenum > (url_value * 2) AND (pct_change_from_prev >= 20 OR pct_change_to_next >= 20) THEN 'Acute_MI'
                WHEN valuenum > url_value AND valuenum <= (url_value * 2) AND (pct_change_from_prev >= 50 OR pct_change_to_next >= 50) THEN 'Acute_MI'
                WHEN valuenum > url_value THEN 'Chronic_Injury'
                ELSE 'Normal'
            END as mi_classification
        FROM elevated_with_delta
        WHERE is_elevated = 1
    ),
    first_acute_mi AS (
        SELECT 
            hadm_id,
            subject_id,
            MIN(charttime) as index_mi_time
        FROM acute_mi_events
        WHERE mi_classification = 'Acute_MI'
        GROUP BY hadm_id, subject_id
    )
    SELECT * FROM first_acute_mi
    """
    
    mi_events = conn.execute(query).fetchdf()
    print(f"Extracted true index_mi_time for {len(mi_events)} Acute MI admissions.")
    
    print("Loading mapping and building new cohorts...")
    ecg_mapping = pd.read_csv(r'C:\Users\Acer\Desktop\Pipeline\ecg_mapping.csv')
    
    # Merge ecg_mapping with cohort to get `is_MI` and `anchor_time` (fallback)
    df = ecg_mapping.merge(cohort, on=['hadm_id', 'subject_id'], how='inner')
    
    # Merge with true index_mi_time
    df = df.merge(mi_events, on=['hadm_id', 'subject_id'], how='left')
    
    df['ecg_time'] = pd.to_datetime(df['ecg_time'])
    df['anchor_time'] = pd.to_datetime(df['anchor_time'])
    df['index_mi_time'] = pd.to_datetime(df['index_mi_time'])
    
    # Use index_mi_time if available, otherwise fallback to anchor_time for non-MI controls
    # Wait! If is_MI is True but they don't have an index_mi_time (e.g. no troponin data or no delta), what do we do?
    # We should fallback to anchor_time so we don't drop them, OR we can label them 'Unknown'.
    # Let's fallback to anchor_time for robustness, but note that the truly validated ones use index_mi_time.
    df['final_index_time'] = df['index_mi_time'].fillna(df['anchor_time'])
    
    df['time_diff_hours'] = (df['ecg_time'] - df['final_index_time']).dt.total_seconds() / 3600.0
    
    conditions = [
        (df['is_MI'] == False),
        (df['is_MI'] == True) & (df['time_diff_hours'] >= -6) & (df['time_diff_hours'] <= 2),
        (df['is_MI'] == True) & (df['time_diff_hours'] < -6),
        (df['is_MI'] == True) & (df['time_diff_hours'] > 2)
    ]
    choices = [
        'Control_Symptomatic',
        'MI_Acute_Presentation',
        'MI_Pre-Incident',
        'MI_Post-Incident'
    ]
    df['Label_Troponin'] = np.select(conditions, choices, default='Unknown')
    
    out_path = r'C:\Users\Acer\Desktop\Pipeline\final_ecg_cohorts_troponin.csv'
    df.to_csv(out_path, index=False)
    print(f"Successfully saved to {out_path}")
    
    # Now generate confusion matrix with old labels
    old_df = pd.read_csv(r'C:\Users\Acer\Desktop\Pipeline\final_ecg_cohorts.csv')
    
    merged = df.merge(old_df[['study_id', 'Label']], on='study_id', how='inner', suffixes=('_trop', '_old'))
    
    # Confusion matrix: rows=old, cols=troponin
    cm = pd.crosstab(merged['Label_old'], merged['Label_trop'])
    
    print("\nConfusion Matrix (Rows=Proxy Anchor Time, Cols=True Troponin Time):")
    print(cm.to_string())
    
    # Save the CM output to a text file for the AI agent to read
    with open('confusion_matrix.txt', 'w') as f:
        f.write(cm.to_string())

if __name__ == '__main__':
    extract_troponin()
