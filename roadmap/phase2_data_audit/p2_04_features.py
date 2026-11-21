"""Phase 2, step 4 — ECG feature availability and completeness.

Locks the 95-feature list (from the legacy master schema) into ecg_feature_list.json and checks,
in the RAW full feature file, completeness for the earliest feature ECG of each candidate admission.

Summary: roadmap/phase2_data_audit/p2_04_features.json
"""
import json

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS

OUT = DATA_OUT / "phase2"
NON_FEATURE = {"study_id", "SubjectID", "hadm_id", "ecg_time", "Label_Troponin", "is_MI", "index_mi_time",
               "time_diff_hours", "diabetes", "hypertension", "ckd", "prior_mi", "anchor_age", "sex_male"}


def feature_list() -> list[str]:
    cols = pq.read_schema(PATHS["master_train"]).names
    return [c for c in cols if c not in NON_FEATURE and not c.startswith("z_ecg")
            and not any(c.startswith(p) for p in ("statin_", "beta_blocker_", "acei_", "aspirin_",
                                                   "vasopressor_", "mech_vent_"))]


def machine_check(x: pd.DataFrame) -> dict:
    """Plausibility: ECGdeli QRS/RR vs the ECG cart's own measurements (machine_measurements)."""
    con = duckdb.connect(str(PATHS["mimic_duckdb"]), read_only=True)
    mm = con.sql("select study_id, rr_interval, qrs_onset, qrs_end from machine_measurements").df()
    y = x[["study_id", "QRS_Dur_Global", "RR_Mean_Global"]].merge(mm, on="study_id")
    y = y[(y["qrs_onset"] < y["qrs_end"]) & (y["qrs_end"] < 29999) & (y["rr_interval"] < 29999)]
    y["machine_qrs"] = y["qrs_end"] - y["qrs_onset"]
    return {"n": len(y),
            "qrs_median_ecgdeli_vs_machine_ms": [round(float(y["QRS_Dur_Global"].median()), 1), float(y["machine_qrs"].median())],
            "qrs_spearman": round(float(y["QRS_Dur_Global"].corr(y["machine_qrs"], method="spearman")), 3),
            "rr_median_ecgdeli_vs_machine_ms": [round(float(y["RR_Mean_Global"].median()), 1), float(y["rr_interval"].median())],
            "rr_spearman": round(float(y["RR_Mean_Global"].corr(y["rr_interval"], method="spearman")), 3)}


def main() -> dict:
    feats = feature_list()
    point = [f for f in feats if not f.endswith(("_iqr", "_count"))]
    iqr = [f for f in feats if f.endswith("_iqr")]
    count = [f for f in feats if f.endswith("_count")]
    raw_cols = pd.read_csv(PATHS["ecg_features_full"], nrows=0).columns
    missing_in_raw = [f for f in feats if f not in raw_cols]

    ae = pd.read_parquet(OUT / "admission_ecg.parquet", columns=["hadm_id", "first_feat_study_id", "timing_class"])
    adm = pd.read_parquet(OUT / "admissions.parquet", columns=["hadm_id", "acute_mi", "chest_pain"])
    ae = ae.dropna(subset=["first_feat_study_id"]).merge(adm, on="hadm_id")
    ids = set(ae["first_feat_study_id"].astype("int64"))
    raw = pd.read_csv(PATHS["ecg_features_full"], usecols=["study_id"] + [f for f in feats if f in raw_cols])
    raw = raw[raw["study_id"].isin(ids)].drop_duplicates("study_id")
    x = ae.merge(raw, left_on="first_feat_study_id", right_on="study_id", how="left")

    complete_point = x[point].notna().all(axis=1)
    s = {
        "n_features": len(feats), "n_point": len(point), "n_iqr": len(iqr), "n_count": len(count),
        "features_missing_in_raw_file": missing_in_raw,
        "candidate_admissions": len(x),
        "admissions_all_95_complete": int(x[feats].notna().all(axis=1).sum()),
        "admissions_all_53_point_complete": int(complete_point.sum()),
        "point_feature_missing_rate_top10": x[point].isna().mean().sort_values(ascending=False).head(10).round(4).to_dict(),
        "complete_point_share_by_group": {
            "mi_after": round(float(complete_point[x["timing_class"] == "after"].mean()), 4),
            "mi_uncertain": round(float(complete_point[x["timing_class"] == "uncertain"].mean()), 4),
            "control": round(float(complete_point[x["timing_class"] == "control"].mean()), 4),
            "before": round(float(complete_point[x["timing_class"] == "before"].mean()), 4),
        },
        "qrs_beat_count_quantiles": x[count[0]].quantile([0.01, 0.05, 0.5, 0.95]).to_dict() if count else None,
        "raw_units_example_median": x[["ST_Elev_V2", "QRS_Dur_Global", "RR_Mean_Global", "T_Amp_II"]].median().round(4).to_dict(),
    }
    s["vs_machine_measurements"] = machine_check(x)
    (PHASE_DIRS[2] / "ecg_feature_list.json").write_text(json.dumps(
        {"all_95": feats, "point_53": point, "iqr_41": iqr, "count_1": count}, indent=2))
    return s


if __name__ == "__main__":
    s = main()
    (PHASE_DIRS[2] / "p2_04_features.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))
