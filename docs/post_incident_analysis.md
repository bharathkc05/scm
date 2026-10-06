# Post-Incident ECG Data Analysis

## TL;DR: Your master dataset is CLEAN. MI_Post-Incident was correctly removed.

The numbers you're seeing (21,917 / 20,287 / 6,026) **do NOT come from the master dataset**. They likely come from a different temporal categorization (simple before/after/at MI cut, not the label-based pipeline). The master dataset has **zero** `MI_Post-Incident` rows.

---

## What's Actually in Each Stage

### 1. Source File: `final_ecg_cohorts_troponin.csv`
This file **does** contain MI_Post-Incident — by design. It's the raw labeled file.

| Label | Count |
|-------|-------|
| Control_Symptomatic | 363,545 |
| **MI_Post-Incident** | **31,572** |
| MI_Acute_Presentation | 7,611 |
| MI_Pre-Incident | 5,957 |

### 2. Master Dataset (train + val + test parquets) ✅
MI_Post-Incident was **correctly excluded** by [build_modeling_data.py](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/build_modeling_data.py#L293-L297):

| Label | Count |
|-------|-------|
| Control_Symptomatic | 362,372 |
| MI_Acute_Presentation | 7,604 |
| MI_Pre-Incident | 5,954 |
| **MI_Post-Incident** | **0** |

> [!IMPORTANT]
> The master dataset has **exactly zero** MI_Post-Incident rows. Verified across all three splits.

---

## Where Your Numbers (21,917 / 20,287 / 6,026) Come From

Your numbers sum to **48,230**, which doesn't match any of the standard label counts. These numbers likely come from a **different temporal categorization** — probably a simple split of ALL MI-admission ECGs using `time_diff_hours` relative to the MI index time, with different cutoff boundaries than the pipeline uses.

For reference, the pipeline boundaries are:
- **MI_Pre-Incident**: `hours < -6` (ECG taken >6h before MI)
- **MI_Acute_Presentation**: `-6 ≤ hours ≤ +2` (ECG taken near MI)
- **MI_Post-Incident**: `hours > +2` (ECG taken >2h after MI)

If you used a **simple before/at/after** split (e.g., hours < 0 / hours ≈ 0 / hours > 0), you'd get different numbers.

---

## Why `final_ecg_cohorts_troponin.csv` Still Contains MI_Post-Incident

This is **intentional and correct**. The filtering happens downstream:

```
final_ecg_cohorts_troponin.csv  (all 408,685 ECGs with all labels)
            │
            ▼  [build_modeling_data.py: Step 3]
            │  Filters: Label_Troponin != "MI_Post-Incident"
            ▼
data/modeling/labels/{train,validation,test}.csv  (375,930 ECGs, NO post-incident)
            │
            ▼  [build_master_dataset.py]
            │  Merges features, demographics, etc.
            ▼
data/modeling/master/{train,validation,test}.parquet  (375,930 ECGs, NO post-incident)
```

### Where the exclusion happens:

1. **[build_modeling_data.py, line 53](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/build_modeling_data.py#L53)**: `EXCLUDE_LABELS = ["MI_Post-Incident"]`
2. **[build_modeling_data.py, line 294-295](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/build_modeling_data.py#L294-L295)**: Actual filtering
3. **[build_modeling_data.py, line 531](file:///c:/Users/Acer/Desktop/Capstone/SCM/data/build_modeling_data.py#L531)**: Verification assertion confirms zero post-incident rows

The `convert_to_hdf5.py` also independently excludes MI_Post-Incident at [line 129](file:///c:/Users/Acer/Desktop/Capstone/SCM/src/data/convert_to_hdf5.py#L129).

---

## ⚠️ One Thing to Watch: `time_diff_hours` in the Master Dataset

While the **labels** are correct (no MI_Post-Incident), the `time_diff_hours` column in the master dataset is a bit misleading for **Control_Symptomatic** ECGs:

- For MI patients: `time_diff_hours` = ECG time − MI index time (meaningful)
- For Control patients: `time_diff_hours` = ECG time − anchor_time (admission time fallback — **not relative to any MI event**)

So if you filter the master dataset by `time_diff_hours > 0` you'll see 352,077 rows — but most of those are **controls** whose ECG was simply after their admission time, NOT post-MI ECGs.

The correct way to identify post-MI ECGs is by `Label_Troponin`, not by `time_diff_hours` alone.
