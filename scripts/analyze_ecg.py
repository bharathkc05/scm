import pandas as pd
import numpy as np

file_path = r"C:\Users\Acer\Desktop\Capstone\SCM\data\raw\processed\extracted_ecg_features.csv"
print("Loading dataset...")
df = pd.read_csv(file_path)
print(f"Total records: {len(df)}")

# Check for missing values
missing_counts = df.isna().sum()
cols_with_missing = missing_counts[missing_counts > 0]
if len(cols_with_missing) > 0:
    print("\nMissing values per column (Top 20):")
    print(cols_with_missing.head(20))
else:
    print("\nNo missing values found.")

rows_with_any_missing = df.isna().any(axis=1).sum()
print(f"Total rows with at least one missing value: {rows_with_any_missing}")

# Check for partial extraction (where some count columns might be 0)
count_cols = [c for c in df.columns if '_count' in c]
if count_cols:
    partial_extraction_mask = (df[count_cols] == 0).any(axis=1)
    print(f"\nRows with 0 count in any lead/feature (partial extraction): {partial_extraction_mask.sum()}")
else:
    partial_extraction_mask = pd.Series([False]*len(df))

# Check for impossible values
# e.g., QRS Duration or RR Mean should be positive.
impossible_qrs_mask = (df['QRS_Dur_Global'] <= 0) if 'QRS_Dur_Global' in df.columns else pd.Series([False]*len(df))
impossible_rr_mask = (df['RR_Mean_Global'] <= 0) if 'RR_Mean_Global' in df.columns else pd.Series([False]*len(df))
extreme_rr_mask = ((df['RR_Mean_Global'] > 5000) | (df['RR_Mean_Global'] < 200)) if 'RR_Mean_Global' in df.columns else pd.Series([False]*len(df))

print(f"\nImpossible values check:")
print(f"Rows with QRS Duration <= 0: {impossible_qrs_mask.sum()}")
print(f"Rows with RR Mean <= 0: {impossible_rr_mask.sum()}")
print(f"Rows with extreme RR values (<200ms or >5000ms): {extreme_rr_mask.sum()}")

# Combine masks to find problematic rows
problematic_mask = df.isna().any(axis=1) | partial_extraction_mask | impossible_qrs_mask | impossible_rr_mask | extreme_rr_mask
problematic_df = df[problematic_mask]

print(f"\nTotal records needing re-extraction: {len(problematic_df)}")
print(f"Total unique subjects needing re-extraction: {problematic_df['SubjectID'].nunique()}")

# Save list of subjects to re-extract
problematic_df[['SubjectID']].drop_duplicates().to_csv(r"C:\Users\Acer\Desktop\Capstone\SCM\data\raw\subjects_to_reextract.csv", index=False)
print("Saved SubjectIDs needing re-extraction to 'subjects_to_reextract.csv'")
