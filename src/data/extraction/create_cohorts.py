import pandas as pd
import numpy as np

def create_cohorts():
    mapping_path = r'C:\Users\Acer\Desktop\Pipeline\ecg_mapping.csv'
    labels_path = r'C:\Users\Acer\Desktop\Pipeline\cohort_labels.csv'
    
    print("Loading datasets...")
    ecg_mapping = pd.read_csv(mapping_path)
    cohort_labels = pd.read_csv(labels_path, usecols=['hadm_id', 'is_MI', 'anchor_time'])
    
    print("Merging datasets on hadm_id...")
    df = ecg_mapping.merge(cohort_labels, on='hadm_id', how='inner')
    
    print("Calculating time differences...")
    df['ecg_time'] = pd.to_datetime(df['ecg_time'])
    df['anchor_time'] = pd.to_datetime(df['anchor_time'])
    
    # Calculate time difference in hours (ecg_time - anchor_time)
    df['time_diff_hours'] = (df['ecg_time'] - df['anchor_time']).dt.total_seconds() / 3600.0
    
    print("Assigning clinical cohort labels...")
    # Vectorized assignment for performance
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
    
    df['Label'] = np.select(conditions, choices, default='Unknown')
    
    print("\n==============================")
    print("Cohort Label Distribution:")
    print("==============================")
    print(df['Label'].value_counts())
    print("==============================\n")
    
    out_path = r'C:\Users\Acer\Desktop\Pipeline\final_ecg_cohorts.csv'
    df.to_csv(out_path, index=False)
    print(f"Successfully saved final cohorts to: {out_path}")

if __name__ == '__main__':
    create_cohorts()
