# ECG-Deli Automated Data Quality Conclusions

## 1. Missingness Concerns
No features have concerning levels (>10%) of missing data.

## 2. Physiological Plausibility (Outside PTB-XL Bounds)
Found 46 features where >5% of MIMIC-IV values fall outside bounds.
- **ST_Elev_aVL_iqr**: 74.3% outside reference bounds
- **ST_Elev_III_iqr**: 68.7% outside reference bounds
- **ST_Elev_aVF_iqr**: 68.2% outside reference bounds
- **ST_Elev_aVR_iqr**: 67.4% outside reference bounds
- **ST_Elev_I_iqr**: 65.9% outside reference bounds

## 3. Distribution Shifts (vs PTB-XL)
All features maintain a reasonable distribution similarity.

## 4. Overall Feature Quality Ratings
- **Excellent**: 97 features
- **Fair**: 32 features
- **Good**: 30 features