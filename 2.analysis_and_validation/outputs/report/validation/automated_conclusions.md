# ECG-Deli Extraction Validation Conclusions

## Overview
Comparing PTB-XL Re-extracted against PTB-XL Original.

## 1. Missingness Concerns (PTB-XL Re-extracted)
No features have concerning levels (>10%) of missing data in PTB-XL Re-extracted.

## 2. Physiological Plausibility (PTB-XL Re-extracted Outside PTB-XL Original Bounds)
Found 29 features where >5% of PTB-XL Re-extracted values fall outside PTB-XL Original bounds.
- **ST_Elev_aVL_iqr**: 74.8% outside reference bounds
- **ST_Elev_III_iqr**: 70.2% outside reference bounds
- **ST_Elev_aVF_iqr**: 68.5% outside reference bounds
- **ST_Elev_aVR_iqr**: 55.5% outside reference bounds
- **ST_Elev_I_iqr**: 46.5% outside reference bounds

## 3. Distribution Similarity (PTB-XL Original vs PTB-XL Re-extracted)
Found 16 features with severe distribution shifts (Score < 60/100).

## 4. Paired Validation Summary (Matched Records)
- **Features analyzed**: 159
- **Mean ICC**: 0.6681
- **Median ICC**: 0.7918
- **Mean Pearson r**: 0.7054
- **Mean MAE**: 0.3743
- **Mean MAPE**: 44.27%

### Agreement Distribution:
- **Excellent**: 42 features
- **Good**: 6 features
- **Moderate**: 7 features
- **Fair**: 59 features
- **Poor**: 45 features

### Features with Fair/Poor Agreement:
- **RR_Mean_Global_iqr**: ICC=0.9437, MAPE=23.4%
- **S_Amp_V2**: ICC=0.9131, MAPE=136.3%
- **Q_Amp_V1**: ICC=0.9117, MAPE=51.5%
- **Q_Amp_V2**: ICC=0.9114, MAPE=68.2%
- **S_Amp_V3**: ICC=0.9030, MAPE=96.3%
- **S_Amp_V1**: ICC=0.9009, MAPE=69.3%
- **Q_Amp_III**: ICC=0.8937, MAPE=63.8%
- **Q_Amp_aVF**: ICC=0.8917, MAPE=58.5%
- **Q_Amp_V3**: ICC=0.8774, MAPE=81.4%
- **T_Amp_V2**: ICC=0.8747, MAPE=117.9%

## Final Feature Quality Ratings
- **Excellent**: 105 features
- **Good**: 28 features
- **Fair**: 16 features
- **Poor**: 5 features
- **Very Poor**: 5 features