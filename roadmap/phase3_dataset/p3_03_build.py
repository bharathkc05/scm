"""Phase 3, step 3 — build the Layer 1 and Layer 2 analysis datasets (one row per admission).

Layer 1: every adult admission with at least one diagnosis code.
Layer 2: every admission with an ECG that has extracted features; ECG = earliest such ECG in the
         admission window (Phase 2 linkage). Analysis-set flags select the primary / sensitivity rows.

Outputs (git-ignored): data/roadmap/phase3/layer1.parquet, layer2.parquet
Committed:            winsorization_bounds.json, cohort_flow.csv, p3_03_build.json
"""
import json

import duckdb
import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS
from roadmap.definitions import FORBIDDEN_COVARIATES

P2, OUT, DOC = DATA_OUT / "phase2", DATA_OUT / "phase3", PHASE_DIRS[3]
FEATS = json.loads((PHASE_DIRS[2] / "ecg_feature_list.json").read_text())
ALL95, POINT, IQR, COUNT = FEATS["all_95"], FEATS["point_53"], FEATS["iqr_41"], FEATS["count_1"][0]
Z = [f"z_ecg_{i}" for i in range(1, 65)]
WINSOR_Q = (0.005, 0.995)

ADM_COLS = ["subject_id", "hadm_id", "age", "sex_male", "arrived_via_ed", "planned_admission",
            "admission_type", "admission_location", "hospital_expire_flag", "has_any_dx",
            "acute_mi", "acute_mi_principal", "mi_stemi", "mi_nstemi", "mi_type2", "mi_other_unspecified",
            "mi_subsequent_episode_icd9", "old_mi", "chest_pain", "n_prior_admissions", "future_acute_mi",
            "diabetes_primary", "diabetes_strict", "hypertension_primary", "hypertension_strict",
            "ckd_primary", "ckd_strict", "prior_mi_primary", "prior_mi_strict"]


def machine_measurements(ids: set) -> pd.DataFrame:
    con = duckdb.connect(str(PATHS["mimic_duckdb"]), read_only=True)
    mm = con.sql("select study_id, rr_interval, qrs_onset, qrs_end from machine_measurements").df()
    mm = mm[mm["study_id"].isin(ids)]
    ok_qrs = (mm["qrs_onset"] < mm["qrs_end"]) & (mm["qrs_end"] < 29999)
    ok_rr = (mm["rr_interval"] > 0) & (mm["rr_interval"] < 29999)
    return pd.DataFrame({"study_id": mm["study_id"],
                         "machine_qrs_ms": (mm["qrs_end"] - mm["qrs_onset"]).where(ok_qrs),
                         "machine_rr_ms": mm["rr_interval"].where(ok_rr)})


def main() -> dict:
    adm = pd.read_parquet(P2 / "admissions.parquet", columns=ADM_COLS)
    splits = pd.read_parquet(OUT / "patient_splits.parquet", columns=["subject_id", "split"])
    adm = adm.merge(splits, on="subject_id", how="left", validate="many_to_one")
    adm["recent_mi_followup"] = ((adm["mi_subsequent_episode_icd9"] == 1) & (adm["acute_mi"] == 0)).astype("int8")

    # ---------------- Layer 1 ----------------
    l1 = adm[adm["has_any_dx"] == 1].drop(columns=["has_any_dx"]).copy()
    ae = pd.read_parquet(P2 / "admission_ecg.parquet")
    l1["has_feature_ecg"] = l1["hadm_id"].isin(set(ae.dropna(subset=["first_feat_study_id"])["hadm_id"])).astype("int8")
    l1["has_any_ecg"] = l1["hadm_id"].isin(set(ae["hadm_id"])).astype("int8")
    l1.to_parquet(OUT / "layer1.parquet", index=False)

    # ---------------- Layer 2 ----------------
    sel = ae.dropna(subset=["first_feat_study_id"]).rename(columns={
        "first_feat_study_id": "study_id", "first_feat_hours_from_arrival": "hours_from_arrival",
        "first_feat_legacy_label": "legacy_label"})
    sel["study_id"] = sel["study_id"].astype("int64")
    sel = sel[["hadm_id", "study_id", "hours_from_arrival", "legacy_label", "first_ecg_has_features", "timing_class"]]
    l2 = adm.merge(sel, on="hadm_id", how="inner", validate="one_to_one")  # keeps has_any_dx

    raw = pd.read_csv(PATHS["ecg_features_full"], usecols=["study_id"] + ALL95)
    raw = raw[raw["study_id"].isin(set(l2["study_id"]))].drop_duplicates("study_id")
    l2 = l2.merge(raw, on="study_id", how="left", validate="one_to_one")
    lat = pd.read_parquet(OUT / "latents_selected.parquet")
    l2 = l2.merge(lat, on="study_id", how="left", validate="one_to_one")
    l2 = l2.merge(machine_measurements(set(l2["study_id"])), on="study_id", how="left", validate="one_to_one")

    l2 = l2.copy()
    # Derived outcomes: territory ST composites (raw units)
    st = lambda leads: l2[[f"ST_Elev_{x}" for x in leads]]  # noqa: E731
    non_avr = ["I", "II", "III", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    der = pd.DataFrame({
        "ST_inferior_mean": st(["II", "III", "aVF"]).mean(axis=1),
        "ST_anterior_mean": st(["V1", "V2", "V3", "V4"]).mean(axis=1),
        "ST_lateral_mean": st(["I", "aVL", "V5", "V6"]).mean(axis=1),
        "ST_max_elevation": st(non_avr).max(axis=1),
        "ST_max_depression": st(non_avr).min(axis=1)}, index=l2.index)
    derived = list(der.columns)
    l2 = pd.concat([l2, der], axis=1)

    # Quality: detected beats vs beats expected in 10 s from the mean RR interval
    l2["beat_detection_ratio"] = l2[COUNT] / (10000.0 / l2["RR_Mean_Global"])
    l2["quality_ok"] = (l2["beat_detection_ratio"].between(0.8, 1.25)).astype("int8")

    # Timing sensitivity: admission context AND legacy troponin label agree on post-onset
    l2["after_strict"] = ((l2["timing_class"] == "after")
                          & l2["legacy_label"].isin(["MI_Acute_Presentation", "MI_Post-Incident"])).astype("int8")

    # Winsorized copies (bounds fitted on the training split only)
    wcols = ALL95 + derived
    train = l2[l2["split"] == "train"]
    bounds = {c: [float(train[c].quantile(WINSOR_Q[0])), float(train[c].quantile(WINSOR_Q[1]))] for c in wcols}
    l2 = pd.concat([l2, pd.DataFrame({f"w_{c}": l2[c].clip(*bounds[c]) for c in wcols}, index=l2.index)], axis=1)
    (DOC / "winsorization_bounds.json").write_text(json.dumps(
        {"quantiles": WINSOR_Q, "fitted_on": "layer2 train split", "bounds": bounds}, indent=2))

    # Analysis sets
    # No diagnosis codes at all -> MI status unknown, never a control.
    ctrl = (l2["acute_mi"] == 0) & (l2["recent_mi_followup"] == 0) & (l2["has_any_dx"] == 1)
    after = (l2["acute_mi"] == 1) & (l2["timing_class"] == "after")
    l2["set_primary"] = (after | ctrl).astype("int8")
    l2["set_active_comparator"] = (after | (ctrl & (l2["chest_pain"] == 1))).astype("int8")
    l2["set_after_strict"] = ((l2["after_strict"] == 1) | ctrl).astype("int8")
    l2["set_nstemi"] = ((after & (l2["mi_nstemi"] == 1) & (l2["mi_stemi"] == 0)) | ctrl).astype("int8")
    l2["set_no_type2"] = ((after & (l2["mi_type2"] == 0)) | ctrl).astype("int8")
    l2["set_first_ecg_has_features"] = (l2["set_primary"].astype(bool) & (l2["first_ecg_has_features"] == 1)).astype("int8")
    l2["set_negative_control"] = (ctrl & (l2["prior_mi_primary"] == 0)).astype("int8")  # exposure = future_acute_mi

    assert not set(FORBIDDEN_COVARIATES) & set(l2.columns) - {"legacy_label"}
    l2.to_parquet(OUT / "layer2.parquet", index=False)

    # ---------------- Flow + gate ----------------
    p2 = {k: json.loads((PHASE_DIRS[2] / f"{k}.json").read_text()) for k in ("p2_01_admissions", "p2_02_ecg_linkage")}
    flow = [
        ("All admissions (MIMIC-IV v2.2)", len(adm), adm["subject_id"].nunique()),
        ("Layer 1: with >=1 diagnosis code", len(l1), l1["subject_id"].nunique()),
        ("  of which acute MI", int(l1["acute_mi"].sum()), l1.loc[l1["acute_mi"] == 1, "subject_id"].nunique()),
        ("With any linked ECG", int(l1["has_any_ecg"].sum()), l1.loc[l1["has_any_ecg"] == 1, "subject_id"].nunique()),
        ("Layer 2 candidates: ECG with features", len(l2), l2["subject_id"].nunique()),
        ("  acute MI, after onset", int(after.sum()), l2.loc[after, "subject_id"].nunique()),
        ("  acute MI, uncertain timing (excluded from primary)", int(((l2["acute_mi"] == 1) & (l2["timing_class"] == "uncertain")).sum()), None),
        ("  410.x2 follow-up admissions (excluded from controls)", int(l2["recent_mi_followup"].sum()), None),
        ("  no diagnosis codes (excluded from controls)", int((l2["has_any_dx"] == 0).sum()), None),
        ("  controls (no acute MI)", int(ctrl.sum()), l2.loc[ctrl, "subject_id"].nunique()),
        ("Layer 2 primary set", int(l2["set_primary"].sum()), l2.loc[l2["set_primary"] == 1, "subject_id"].nunique()),
    ]
    pd.DataFrame(flow, columns=["step", "admissions", "patients"]).to_csv(DOC / "cohort_flow.csv", index=False)

    gate = {
        "layer1_rows_eq_admissions_with_dx": len(l1) == p2["p2_01_admissions"]["admissions"] - p2["p2_01_admissions"]["admissions_without_any_dx"],
        "layer2_rows_eq_admissions_with_feature_ecg": len(l2) == p2["p2_02_ecg_linkage"]["admissions_with_ecg_with_features"],
        "after_onset_eq_phase2": int(after.sum()) == p2["p2_02_ecg_linkage"]["mi_admissions_with_feature_ecg_by_timing"]["after"],
        "one_row_per_admission": bool(l2["hadm_id"].is_unique and l1["hadm_id"].is_unique),
        "every_patient_in_one_split": bool(l2.groupby("subject_id")["split"].nunique().max() == 1),
        "split_missing": int(l2["split"].isna().sum() + l1["split"].isna().sum()),
        "features_complete": bool(l2[ALL95].notna().all().all()),
        "latents_complete": bool(l2[Z].notna().all().all()),
    }
    sets = {c: {"rows": int(l2[c].sum()), "exposed": int((l2[c].astype(bool) & (l2["acute_mi"] == 1)).sum())}
            for c in l2.columns if c.startswith("set_")}
    sets["set_negative_control"]["exposed"] = int((l2["set_negative_control"].astype(bool) & (l2["future_acute_mi"] == 1)).sum())
    return {
        "gate_checks": gate, "flow": [dict(zip(["step", "admissions", "patients"], r)) for r in flow],
        "analysis_sets": sets,
        "layer2_by_split": l2.groupby("split").agg(rows=("hadm_id", "size"), acute_mi_after=("set_primary", lambda s: int((s.astype(bool) & (l2.loc[s.index, "acute_mi"] == 1)).sum()))).to_dict("index"),
        "machine_measure_coverage": {"qrs": round(float(l2["machine_qrs_ms"].notna().mean()), 4), "rr": round(float(l2["machine_rr_ms"].notna().mean()), 4)},
        "quality_ok_share": round(float(l2["quality_ok"].mean()), 4),
        "beat_detection_ratio_quantiles": l2["beat_detection_ratio"].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(3).to_dict(),
        "latent_source": l2["latent_source"].value_counts().to_dict(),
        "layer1_columns": list(l1.columns), "layer2_n_columns": l2.shape[1],
    }


if __name__ == "__main__":
    s = main()
    (DOC / "p3_03_build.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps({k: v for k, v in s.items() if k != "layer1_columns"}, indent=2, default=str))
