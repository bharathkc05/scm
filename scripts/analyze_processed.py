import pandas as pd
import duckdb
import os

PROCESSED_DIR = r"c:\Users\Acer\Desktop\Capstone\Capstone\data\raw\processed"

files = {
    "cohort_labels": "cohort_labels.csv",
    "ecg_mapping": "ecg_mapping.csv",
    "features_labs": "features_labs.csv",
    "features_treatments": "features_treatments.csv",
    "features_vitals": "features_vitals.csv"
}

def analyze():
    print("Dataset Analysis:\n" + "="*40)
    for name, filename in files.items():
        filepath = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File {filename} not found.")
            continue
        
        try:
            df = pd.read_csv(filepath)
            print(f"\n--- {filename} ---")
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Columns: {', '.join(df.columns)}")
            
            if name == "cohort_labels":
                print("\nMI Class Distribution:")
                for col in ["is_MI", "type_1_MI", "type_2_MI", "subsequent_MI", "other_MI", "complication_mi"]:
                    if col in df.columns:
                        count = df[col].sum()
                        print(f"  {col}: {count} ({(count/df.shape[0])*100:.2f}%)")
                if "is_mi" in df.columns:
                    print(f"  is_mi: {df['is_mi'].sum()}")
                    
            if name == "ecg_mapping":
                if "ecg_taken_in_ed_or_hosp" in df.columns:
                    print("\nECG Availability:")
                    count = df["ecg_taken_in_ed_or_hosp"].sum()
                    print(f"  Mapped to ECG: {count} ({(count/df.shape[0])*100:.2f}%)")
                    
            print("\nMissing Values:")
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                for col, val in missing.items():
                    print(f"  {col}: {val} ({(val/df.shape[0])*100:.2f}%)")
            else:
                print("  No missing values.")
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    analyze()
