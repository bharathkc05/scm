import pandas as pd
import os

def extract_comorbidities():
    cohort_path = r'C:\Users\Acer\Desktop\Pipeline\cohort_labels.csv'
    diag_path = r'C:\Users\Acer\Desktop\Capstone\Capstone\data\raw\MIMIC-IV-2.2\hosp\diagnoses_icd.csv'
    adm_path = r'C:\Users\Acer\Desktop\Capstone\Capstone\data\raw\MIMIC-IV-2.2\hosp\admissions.csv'
    
    print("Loading cohort labels...")
    cohort = pd.read_csv(cohort_path, usecols=['subject_id', 'hadm_id', 'anchor_time'])
    cohort['anchor_time'] = pd.to_datetime(cohort['anchor_time'])
    
    print("Loading admissions...")
    admissions = pd.read_csv(adm_path, usecols=['subject_id', 'hadm_id', 'admittime'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    
    # We need to map every diagnosis to its admission time to enforce the temporal rule
    print("Processing diagnoses in chunks...")
    
    # Initialize dictionary to store comorbidities for each hadm_id in cohort
    # We map hadm_id -> { 'diabetes': 0, 'hypertension': 0, 'ckd': 0, 'prior_mi': 0 }
    comorb_dict = { h: {'diabetes': 0, 'hypertension': 0, 'ckd': 0, 'prior_mi': 0} for h in cohort['hadm_id'].unique() }
    
    # Create a quick lookup for patient anchor times
    patient_anchors = cohort.set_index('hadm_id')[['subject_id', 'anchor_time']]
    
    # Merge admissions with patient_anchors isn't exactly right because one patient can have multiple anchor times 
    # for multiple hospitalizations in the cohort.
    # Better: for each patient, get their history of diagnoses and dates.
    
    # First, let's just get all diagnoses for subjects in our cohort
    valid_subjects = set(cohort['subject_id'].unique())
    
    diag_chunks = pd.read_csv(diag_path, chunksize=1000000)
    
    # We'll collect relevant diagnoses
    relevant_diags = []
    
    for chunk in diag_chunks:
        chunk = chunk[chunk['subject_id'].isin(valid_subjects)]
        if chunk.empty: continue
        
        # Filter for codes of interest
        # Diabetes: ICD9: 250*, ICD10: E08-E13*
        # Hypertension: ICD9: 401-405*, ICD10: I10-I15*
        # CKD: ICD9: 585*, ICD10: N18*
        # Prior MI: ICD9: 410*, 411*, 412* (wait, pipeline says I21%, I22% for ICD10)
        # Actually pipeline phase E.5 says:
        # Diabetes: ICD-9: 250%, ICD-10: E08-E13%
        # Hypertension: ICD-9: 401-405%, ICD-10: I10-I15%
        # CKD: ICD-9: 585%, ICD-10: N18%
        # Chronic MI: ICD-9: 412, ICD-10: I21%, I22% (from phase C.6)
        
        icd9 = chunk['icd_version'] == 9
        icd10 = chunk['icd_version'] == 10
        codes = chunk['icd_code'].astype(str)
        
        chunk['is_diabetes'] = ((icd9 & codes.str.startswith('250')) | 
                               (icd10 & codes.str.match(r'^(E0[89]|E1[0-3])')))
                               
        chunk['is_htn'] = ((icd9 & codes.str.match(r'^40[1-5]')) | 
                          (icd10 & codes.str.match(r'^I1[0-5]')))
                          
        chunk['is_ckd'] = ((icd9 & codes.str.startswith('585')) | 
                          (icd10 & codes.str.startswith('N18')))
                          
        chunk['is_prior_mi'] = ((icd9 & (codes.str.startswith('410') | codes.str.startswith('412'))) | 
                               (icd10 & (codes.str.startswith('I21') | codes.str.startswith('I22'))))
                               
        chunk = chunk[chunk['is_diabetes'] | chunk['is_htn'] | chunk['is_ckd'] | chunk['is_prior_mi']]
        relevant_diags.append(chunk[['subject_id', 'hadm_id', 'is_diabetes', 'is_htn', 'is_ckd', 'is_prior_mi']])

    print("Aggregating diagnoses...")
    if relevant_diags:
        all_diags = pd.concat(relevant_diags)
        all_diags = all_diags.groupby(['subject_id', 'hadm_id']).max().reset_index()
        
        # Merge with admittime
        all_diags = all_diags.merge(admissions, on=['subject_id', 'hadm_id'], how='left')
        
        # For each row in cohort, find prior diagnoses
        # This can be slow if done with iterrows, let's use a mergeasof or cross join
        # Since cohort is small (~200k), we can do it efficiently.
        print("Mapping to cohort...")
        
        # Let's sort both
        cohort = cohort.sort_values(by=['subject_id', 'anchor_time'])
        all_diags = all_diags.sort_values(by=['subject_id', 'admittime'])
        
        results = []
        for _, row in cohort.iterrows():
            subj = row['subject_id']
            atime = row['anchor_time']
            h_id = row['hadm_id']
            
            # get all prior diags for this subject
            prior = all_diags[(all_diags['subject_id'] == subj) & (all_diags['admittime'] < atime)]
            
            if not prior.empty:
                results.append({
                    'hadm_id': h_id,
                    'diabetes': int(prior['is_diabetes'].max()),
                    'hypertension': int(prior['is_htn'].max()),
                    'ckd': int(prior['is_ckd'].max()),
                    'prior_mi': int(prior['is_prior_mi'].max())
                })
            else:
                results.append({
                    'hadm_id': h_id,
                    'diabetes': 0, 'hypertension': 0, 'ckd': 0, 'prior_mi': 0
                })
                
        out_df = pd.DataFrame(results)
    else:
        # no diags found?
        out_df = pd.DataFrame({'hadm_id': cohort['hadm_id'], 'diabetes': 0, 'hypertension': 0, 'ckd': 0, 'prior_mi': 0})
        
    out_path = r'C:\Users\Acer\Desktop\Pipeline\features_comorbidities.csv'
    out_df.to_csv(out_path, index=False)
    print(f"Success! Saved to {out_path}")

if __name__ == '__main__':
    extract_comorbidities()
