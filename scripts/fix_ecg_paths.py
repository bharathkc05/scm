import pandas as pd
import os

PROCESSED_DIR = r"c:\Users\Acer\Desktop\Capstone\Capstone\data\raw\processed"
mapping_file = os.path.join(PROCESSED_DIR, "ecg_mapping.csv")

print("Loading ecg_mapping.csv...")
df = pd.read_csv(mapping_file)

# The original path from physionet is 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/...'
# But the local folder is 'MIMIC-IV-ECG-1.0/files/...'
old_prefix = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
new_prefix = "MIMIC-IV-ECG-1.0/"

print(f"Replacing prefix '{old_prefix}' with '{new_prefix}'...")
df['waveform_path'] = df['waveform_path'].str.replace(old_prefix, new_prefix)

# Save it back
df.to_csv(mapping_file, index=False)
print("Updated ecg_mapping.csv successfully. Sample paths:")
print(df['waveform_path'].head())
