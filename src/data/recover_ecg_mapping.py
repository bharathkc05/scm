import pandas as pd
import os

def recover_mapping():
    print("Loading datasets...")
    mapping_path = r'C:\Users\Acer\Desktop\Pipeline\ecg_mapping.csv'
    features_path = r'C:\Users\Acer\Desktop\Pipeline\extracted_ecg_features.csv'
    
    mapping_df = pd.read_csv(mapping_path)
    features_df = pd.read_csv(features_path)

    print(f"Mapping rows: {len(mapping_df)}, Features rows: {len(features_df)}")
    
    mapping_subjects = mapping_df['subject_id'].values
    features_subjects = features_df['SubjectID'].values
    
    mapped_study_ids = []
    mapped_ecg_times = []
    
    print("Starting sequence alignment...")
    j = 0
    for i in range(len(features_subjects)):
        f_sub = features_subjects[i]
        while j < len(mapping_subjects) and mapping_subjects[j] != f_sub:
            j += 1
        if j < len(mapping_subjects):
            mapped_study_ids.append(mapping_df['study_id'].iloc[j])
            mapped_ecg_times.append(mapping_df['ecg_time'].iloc[j])
            j += 1
        else:
            raise ValueError(f"Could not match feature SubjectID {f_sub} at row {i}")
            
    features_df['study_id'] = mapped_study_ids
    features_df['ecg_time'] = mapped_ecg_times
    
    cols = ['SubjectID', 'study_id', 'ecg_time'] + [c for c in features_df.columns if c not in ['SubjectID', 'study_id', 'ecg_time']]
    features_df = features_df[cols]
    
    out_path = r'C:\Users\Acer\Desktop\Pipeline\extracted_ecg_features_with_study_id.csv'
    print(f"Saving to {out_path}...")
    features_df.to_csv(out_path, index=False)
    print("Success!")

if __name__ == '__main__':
    recover_mapping()
