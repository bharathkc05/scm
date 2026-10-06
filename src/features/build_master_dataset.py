"""
Phase F: Build Master Dataset

Merges all data sources into a single master table per split:
  - Labels (backbone)
  - ECG embeddings (64-dim z_ecg)
  - ECG features (95 morphological features)
  - Comorbidities (diabetes, hypertension, ckd, prior_mi)
  - Treatments (statin, beta-blocker, ACEi, aspirin, etc.)
  - Demographics (age, sex)

Dropped (insufficient coverage <1%):
  - Labs (738 rows / 194K admissions)
  - Vitals (1,457 rows / 194K admissions)

Excluded per pipeline E.1:
  - peak_troponin_before (prevents circular prediction)

Output:
  data/modeling/master/{train,validation,test}.parquet

Usage:
    python src/features/build_master_dataset.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

SCM_DIR = Path(__file__).resolve().parent.parent.parent
MODELING_DIR = SCM_DIR / "data" / "modeling"
OUTPUT_DIR = MODELING_DIR / "master"


def load_labels(split: str) -> pd.DataFrame:
    """Load split labels as the backbone."""
    df = pd.read_csv(MODELING_DIR / "labels" / f"{split}.csv")
    # Keep only essential columns
    keep = ["study_id", "SubjectID", "hadm_id", "ecg_time", "Label_Troponin",
            "is_MI", "index_mi_time", "time_diff_hours"]
    return df[[c for c in keep if c in df.columns]]


def load_ecg_embeddings(split: str) -> pd.DataFrame:
    """Load 64-dim VAE embeddings."""
    df = pd.read_csv(MODELING_DIR / "ecg_embeddings" / f"{split}.csv")
    # Drop SubjectID (already in labels)
    if "SubjectID" in df.columns:
        df = df.drop(columns=["SubjectID"])
    return df


def load_ecg_features(split: str) -> pd.DataFrame:
    """Load morphological ECG features."""
    df = pd.read_csv(MODELING_DIR / "ecg_features" / f"{split}.csv")
    # Drop SubjectID (already in labels)
    if "SubjectID" in df.columns:
        df = df.drop(columns=["SubjectID"])
    return df


def load_comorbidities() -> pd.DataFrame:
    """Load comorbidity flags (hadm_id level)."""
    return pd.read_csv(MODELING_DIR / "clinical" / "features_comorbidities.csv")


def load_treatments() -> pd.DataFrame:
    """Load treatment flags (hadm_id level)."""
    return pd.read_csv(MODELING_DIR / "clinical" / "features_treatments.csv")


def load_demographics() -> pd.DataFrame:
    """Load demographics (SubjectID level)."""
    df = pd.read_csv(MODELING_DIR / "demographics" / "demographics.csv")
    # Encode gender as binary
    if "gender" in df.columns:
        df["sex_male"] = (df["gender"] == "M").astype(int)
        df = df.drop(columns=["gender"])
    return df


def create_missing_flags(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Create binary _missing flags for features with >10% missingness."""
    n = len(df)
    for col in feature_cols:
        if col not in df.columns:
            continue
        missing_rate = df[col].isna().sum() / n
        if missing_rate > 0.10:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def build_split(split: str, comorbidities: pd.DataFrame,
                treatments: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """Build master dataset for a single split."""
    print(f"\n{'='*60}")
    print(f"Building {split} split")
    print(f"{'='*60}")

    # 1. Load backbone
    labels = load_labels(split)
    n_start = len(labels)
    print(f"  Labels: {n_start:,} rows")

    # 2. Merge ECG embeddings (on study_id)
    embeddings = load_ecg_embeddings(split)
    master = labels.merge(embeddings, on="study_id", how="left")
    n_emb = master["z_ecg_1"].notna().sum()
    print(f"  + ECG embeddings: {n_emb:,}/{n_start:,} matched ({100*n_emb/n_start:.1f}%)")

    # 3. Merge ECG features (on study_id)
    ecg_feats = load_ecg_features(split)
    ecg_feat_cols = [c for c in ecg_feats.columns if c != "study_id"]
    master = master.merge(ecg_feats, on="study_id", how="left")
    n_feat = master[ecg_feat_cols[0]].notna().sum()
    print(f"  + ECG features: {n_feat:,}/{n_start:,} matched ({100*n_feat/n_start:.1f}%)")

    # 4. Merge comorbidities (on hadm_id)
    master = master.merge(comorbidities, on="hadm_id", how="left")
    n_comor = master["diabetes"].notna().sum()
    print(f"  + Comorbidities: {n_comor:,}/{n_start:,} matched ({100*n_comor/n_start:.1f}%)")

    # 5. Merge treatments (on hadm_id)
    master = master.merge(treatments, on="hadm_id", how="left")
    n_treat = master["statin_before"].notna().sum()
    print(f"  + Treatments: {n_treat:,}/{n_start:,} matched ({100*n_treat/n_start:.1f}%)")

    # 6. Merge demographics (on SubjectID)
    master = master.merge(demographics, on="SubjectID", how="left")
    n_demo = master["anchor_age"].notna().sum()
    print(f"  + Demographics: {n_demo:,}/{n_start:,} matched ({100*n_demo/n_start:.1f}%)")

    # Verify row count preserved
    assert len(master) == n_start, (
        f"Row count changed after merge! {n_start} -> {len(master)}. "
        f"Likely duplicate hadm_ids in clinical tables."
    )

    # 7. Create missing flags for clinical features
    clinical_cols = (
        list(comorbidities.columns[1:]) +  # skip hadm_id
        list(treatments.columns[1:]) +
        list(demographics.columns[1:])
    )
    # Remove sex_male which we created
    clinical_cols = [c for c in clinical_cols if c in master.columns]
    master = create_missing_flags(master, clinical_cols)

    # 8. Final column report
    z_cols = [c for c in master.columns if c.startswith("z_ecg_")]
    ecg_f_cols = [c for c in ecg_feat_cols if c in master.columns]
    missing_cols = [c for c in master.columns if c.endswith("_missing")]
    id_cols = ["study_id", "SubjectID", "hadm_id"]
    label_cols = ["Label_Troponin", "is_MI"]
    time_cols = ["ecg_time", "index_mi_time", "time_diff_hours"]

    print(f"\n  Final shape: {master.shape}")
    print(f"    Identifiers: {len([c for c in id_cols if c in master.columns])}")
    print(f"    Labels: {len([c for c in label_cols if c in master.columns])}")
    print(f"    Time: {len([c for c in time_cols if c in master.columns])}")
    print(f"    ECG embeddings: {len(z_cols)}")
    print(f"    ECG features: {len(ecg_f_cols)}")
    print(f"    Clinical: {len(clinical_cols)}")
    print(f"    Missing flags: {len(missing_cols)}")

    return master


def missingness_report(master: pd.DataFrame, split: str):
    """Print missingness statistics."""
    print(f"\n  Missingness report ({split}):")
    exclude = ["study_id", "SubjectID", "hadm_id", "ecg_time",
               "Label_Troponin", "is_MI", "index_mi_time", "time_diff_hours"]
    cols = [c for c in master.columns if c not in exclude and not c.endswith("_missing")]
    n = len(master)

    high_missing = []
    for col in sorted(cols):
        miss = master[col].isna().sum()
        rate = 100 * miss / n
        if rate > 5:
            high_missing.append((col, rate))

    if high_missing:
        print(f"    Features with >5% missing:")
        for col, rate in sorted(high_missing, key=lambda x: -x[1]):
            print(f"      {col}: {rate:.1f}%")
    else:
        print(f"    All features have <5% missing")


def main():
    print("Phase F: Building Master Dataset")
    print(f"Source: {MODELING_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared clinical tables (not split-specific)
    print("\nLoading clinical tables...")
    comorbidities = load_comorbidities()
    print(f"  Comorbidities: {len(comorbidities):,} hadm_ids")
    treatments = load_treatments()
    print(f"  Treatments: {len(treatments):,} hadm_ids")
    demographics = load_demographics()
    print(f"  Demographics: {len(demographics):,} SubjectIDs")

    # Build each split
    for split in ["train", "validation", "test"]:
        master = build_split(split, comorbidities, treatments, demographics)
        missingness_report(master, split)

        # Save as parquet
        out_path = OUTPUT_DIR / f"{split}.parquet"
        master.to_parquet(out_path, index=False)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"\n  Saved: {out_path} ({size_mb:.1f} MB)")

    # Final summary
    print("\n" + "=" * 60)
    print("MASTER DATASET BUILD COMPLETE")
    print("=" * 60)
    for split in ["train", "validation", "test"]:
        df = pd.read_parquet(OUTPUT_DIR / f"{split}.parquet")
        print(f"  {split}: {len(df):,} rows x {len(df.columns)} cols")

    # Label distribution sanity check
    print("\nLabel distribution (train):")
    train = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    print(train["Label_Troponin"].value_counts().to_string())


if __name__ == "__main__":
    main()
