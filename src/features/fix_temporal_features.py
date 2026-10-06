import pandas as pd
import numpy as np

def fix_temporal():
    cohort_path = r'C:\Users\Acer\Desktop\Pipeline\cohort_labels.csv'
    labs_path = r'C:\Users\Acer\Desktop\Capstone\Capstone\data\raw\MIMIC-IV-2.2\hosp\labevents.csv'
    vitals_path = r'C:\Users\Acer\Desktop\Capstone\Capstone\data\raw\MIMIC-IV-2.2\icu\chartevents.csv'
    
    print("Loading cohort labels...")
    cohort = pd.read_csv(cohort_path, usecols=['hadm_id', 'subject_id', 'anchor_time', 'window_before_start'])
    cohort['anchor_time'] = pd.to_datetime(cohort['anchor_time'])
    cohort['window_before_start'] = pd.to_datetime(cohort['window_before_start'])
    
    valid_subjects = set(cohort['subject_id'])
    
    # ------------------ LABS ------------------
    print("Processing labevents...")
    lab_chunks = pd.read_csv(labs_path, chunksize=2000000, usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'])
    lab_items = {
        50904: 'hdl_before',
        50905: 'ldl_before', 50906: 'ldl_before',
        51002: 'peak_troponin_before', 51003: 'peak_troponin_before', 52642: 'peak_troponin_before',
        50912: 'peak_creatinine_before',
        51006: 'peak_bun_before',
        50931: 'peak_glucose_before',
        50971: 'peak_potassium_before',
        50963: 'peak_bnp_before'
    }
    
    lab_results = []
    for chunk in lab_chunks:
        chunk = chunk[chunk['subject_id'].isin(valid_subjects) & chunk['itemid'].isin(lab_items.keys())]
        if chunk.empty: continue
        chunk['charttime'] = pd.to_datetime(chunk['charttime'])
        
        merged = chunk.merge(cohort, on=['subject_id', 'hadm_id'], how='inner')
        valid = merged[(merged['charttime'] >= merged['window_before_start']) & (merged['charttime'] <= merged['anchor_time'])].copy()
        
        if not valid.empty:
            valid['feature_name'] = valid['itemid'].map(lab_items)
            lab_results.append(valid[['hadm_id', 'feature_name', 'valuenum']])
            
    if lab_results:
        lab_df = pd.concat(lab_results)
        lab_agg = lab_df.groupby(['hadm_id', 'feature_name'])['valuenum'].max().unstack().reset_index()
        out_labs = r'C:\Users\Acer\Desktop\Pipeline\features_labs_fixed.csv'
        lab_agg.to_csv(out_labs, index=False)
        print(f"Saved {out_labs}")
        
    # ------------------ VITALS ------------------
    print("Processing chartevents (this may take a few minutes)...")
    vital_chunks = pd.read_csv(vitals_path, chunksize=2000000, usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'])
    vital_items = {
        220045: 'max_hr_before',
        220277: 'min_spo2_before',
        223761: 'max_temp_celsius_before', 223762: 'max_temp_celsius_before',
        220210: 'max_rr_before', 224690: 'max_rr_before',
        220179: 'min_sbp_before', 220050: 'min_sbp_before'
    }
    
    max_results = []
    min_results = []
    
    for i, chunk in enumerate(vital_chunks):
        chunk = chunk[chunk['subject_id'].isin(valid_subjects) & chunk['itemid'].isin(vital_items.keys())]
        if chunk.empty: continue
        chunk['charttime'] = pd.to_datetime(chunk['charttime'])
        
        mask = chunk['itemid'] == 223761
        chunk.loc[mask, 'valuenum'] = (chunk.loc[mask, 'valuenum'] - 32) * 5.0/9.0
        
        merged = chunk.merge(cohort, on=['subject_id', 'hadm_id'], how='inner')
        valid = merged[(merged['charttime'] >= merged['window_before_start']) & (merged['charttime'] <= merged['anchor_time'])].copy()
        
        if not valid.empty:
            valid['feature_name'] = valid['itemid'].map(vital_items)
            
            # split min vs max
            max_mask = valid['feature_name'].str.startswith('max')
            if max_mask.any():
                max_results.append(valid.loc[max_mask, ['hadm_id', 'feature_name', 'valuenum']])
            
            min_mask = valid['feature_name'].str.startswith('min')
            if min_mask.any():
                min_results.append(valid.loc[min_mask, ['hadm_id', 'feature_name', 'valuenum']])
                
        if i % 5 == 0:
            print(f"Processed chunk {i}")

    print("Aggregating vitals...")
    final_dfs = []
    if max_results:
        max_df = pd.concat(max_results)
        max_agg = max_df.groupby(['hadm_id', 'feature_name'])['valuenum'].max().unstack().reset_index()
        final_dfs.append(max_agg)
        
    if min_results:
        min_df = pd.concat(min_results)
        min_agg = min_df.groupby(['hadm_id', 'feature_name'])['valuenum'].min().unstack().reset_index()
        final_dfs.append(min_agg)
        
    if final_dfs:
        if len(final_dfs) == 2:
            vitals_agg = final_dfs[0].merge(final_dfs[1], on='hadm_id', how='outer')
        else:
            vitals_agg = final_dfs[0]
            
        out_vitals = r'C:\Users\Acer\Desktop\Pipeline\features_vitals_fixed.csv'
        vitals_agg.to_csv(out_vitals, index=False)
        print(f"Saved {out_vitals}")

if __name__ == '__main__':
    fix_temporal()
