"""Phase 2, step 2 — link every ECG to an admission and classify timing relative to MI onset.

Timing rule (admission context, NOT troponin — roadmap "Changes after audit"):
  ECG in an acute-MI admission:
    after      : MI is the principal diagnosis (seq_num 1) and the admission was unplanned
                 -> the MI is the reason for coming in, so onset precedes arrival and every ECG.
    uncertain  : MI coded in a secondary position, or a planned (elective / same-day surgical /
                 procedure) admission -> MI may have begun in hospital.
  ECG in a non-MI admission:
    before     : the patient has an acute MI on a LATER admission (negative-control pool)
    control    : otherwise
  ECG outside any admission window: unlinked.

Outputs (git-ignored): data/roadmap/phase2/ecg_links.parquet, admission_ecg.parquet
Summary:               roadmap/phase2_data_audit/p2_02_ecg_linkage.json
"""
import json

import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS

OUT = DATA_OUT / "phase2"


def link() -> pd.DataFrame:
    adm = pd.read_parquet(OUT / "admissions.parquet", columns=[
        "subject_id", "hadm_id", "arrival_time", "dischtime", "acute_mi", "acute_mi_principal",
        "planned_admission", "future_acute_mi"])
    ecg = pd.read_csv(PATHS["raw_ecg_record_list"], usecols=["subject_id", "study_id", "ecg_time"])
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"])

    # Latest admission whose arrival <= ecg_time (same patient), then require ecg_time <= discharge.
    e = ecg.sort_values("ecg_time")
    a = adm.sort_values("arrival_time")
    m = pd.merge_asof(e, a, left_on="ecg_time", right_on="arrival_time", by="subject_id", direction="backward")
    inside = m["hadm_id"].notna() & (m["ecg_time"] <= m["dischtime"])
    m.loc[~inside, ["hadm_id", "arrival_time", "dischtime", "acute_mi", "acute_mi_principal",
                    "planned_admission", "future_acute_mi"]] = pd.NA
    m["hours_from_arrival"] = (m["ecg_time"] - m["arrival_time"]).dt.total_seconds() / 3600

    # Feature / legacy-master availability
    feat_ids = pd.read_csv(PATHS["ecg_features_full"], usecols=["study_id"])["study_id"]
    m["has_features"] = m["study_id"].isin(set(feat_ids)).astype("int8")
    master_ids = pd.concat([pd.read_parquet(PATHS[f"master_{s}"], columns=["study_id"])["study_id"]
                            for s in ("train", "validation", "test")])
    m["in_master"] = m["study_id"].isin(set(master_ids)).astype("int8")
    legacy = pd.read_csv(PATHS["legacy_troponin_labels"], usecols=["study_id", "hadm_id", "Label_Troponin"])
    legacy = legacy.drop_duplicates("study_id").rename(columns={"hadm_id": "legacy_hadm_id"})
    m = m.merge(legacy, on="study_id", how="left")

    linked = m["hadm_id"].notna()
    mi = linked & (m["acute_mi"] == 1)
    after = mi & (m["acute_mi_principal"] == 1) & (m["planned_admission"] == 0)
    m["timing_class"] = "unlinked"
    m.loc[linked & (m["acute_mi"] == 0), "timing_class"] = "control"
    m.loc[linked & (m["acute_mi"] == 0) & (m["future_acute_mi"] == 1), "timing_class"] = "before"
    m.loc[mi, "timing_class"] = "uncertain"
    m.loc[after, "timing_class"] = "after"
    return m


def admission_level(m: pd.DataFrame) -> pd.DataFrame:
    l = m[m["hadm_id"].notna()].sort_values(["hadm_id", "ecg_time", "study_id"])
    first_any = l.groupby("hadm_id").head(1).set_index("hadm_id")
    first_feat = l[l["has_features"] == 1].groupby("hadm_id").head(1).set_index("hadm_id")
    out = pd.DataFrame({
        "n_ecg": l.groupby("hadm_id").size(),
        "n_ecg_with_features": l.groupby("hadm_id")["has_features"].sum(),
        "first_ecg_study_id": first_any["study_id"],
        "first_ecg_has_features": first_any["has_features"],
        "first_feat_study_id": first_feat["study_id"],
        "first_feat_hours_from_arrival": first_feat["hours_from_arrival"],
        "first_feat_in_master": first_feat["in_master"],
        "first_feat_legacy_label": first_feat["Label_Troponin"],
        "timing_class": first_any["timing_class"],
    })
    out.index = out.index.astype("int64")
    return out.reset_index()


def summarize(m: pd.DataFrame, ae: pd.DataFrame) -> dict:
    adm = pd.read_parquet(OUT / "admissions.parquet", columns=["hadm_id", "acute_mi", "acute_mi_principal",
                                                                "planned_admission", "chest_pain"])
    x = adm.merge(ae, on="hadm_id", how="left")
    has_ecg = x["n_ecg"].notna()
    has_feat = x["first_feat_study_id"].notna()
    s = {
        "ecgs_total": len(m),
        "ecgs_linked_to_admission": int(m["hadm_id"].notna().sum()),
        "ecgs_with_features": int(m["has_features"].sum()),
        "ecgs_with_features_linked": int((m["has_features"].eq(1) & m["hadm_id"].notna()).sum()),
        "ecgs_in_legacy_master": int(m["in_master"].sum()),
        "ecg_timing_class_counts": m["timing_class"].value_counts().to_dict(),
        "legacy_hadm_agreement_where_both_present": round(float(
            (m["hadm_id"] == m["legacy_hadm_id"])[m["hadm_id"].notna() & m["legacy_hadm_id"].notna()].mean()), 4),
        "admissions_total": len(x),
        "admissions_with_any_ecg": int(has_ecg.sum()),
        "admissions_with_ecg_with_features": int(has_feat.sum()),
        "admissions_where_first_ecg_lacks_features_but_later_has": int(
            (has_feat & x["first_ecg_has_features"].eq(0)).sum()),
    }
    by = {}
    for name, mask in {"acute_mi": x["acute_mi"] == 1, "no_acute_mi": x["acute_mi"] == 0,
                       "chest_pain_no_mi": (x["chest_pain"] == 1) & (x["acute_mi"] == 0)}.items():
        sub = x[mask]
        by[name] = {"admissions": len(sub), "with_any_ecg": round(float(sub["n_ecg"].notna().mean()), 4),
                    "with_feature_ecg": round(float(sub["first_feat_study_id"].notna().mean()), 4)}
    s["ecg_coverage_by_group"] = by

    mi = x[(x["acute_mi"] == 1) & has_feat]
    s["mi_admissions_with_feature_ecg_by_timing"] = mi["timing_class"].value_counts().to_dict()
    s["mi_after_onset_first_feature_ecg_in_legacy_master"] = round(float(
        mi.loc[mi["timing_class"] == "after", "first_feat_in_master"].mean()), 4)
    s["mi_first_feature_ecg_legacy_label"] = mi["first_feat_legacy_label"].value_counts(dropna=False).to_dict()
    s["mi_after_onset_hours_from_arrival_quantiles"] = (
        mi.loc[mi["timing_class"] == "after", "first_feat_hours_from_arrival"]
        .quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict())
    ctrl = x[(x["acute_mi"] == 0) & has_feat]
    s["control_first_feature_ecg_in_legacy_master"] = round(float(ctrl["first_feat_in_master"].mean()), 4)
    s["control_first_feature_ecg_legacy_label"] = ctrl["first_feat_legacy_label"].value_counts(dropna=False).to_dict()
    s["before_pool_admissions_with_feature_ecg"] = int(
        ((x["timing_class"] == "before") & has_feat).sum())
    return s


if __name__ == "__main__":
    m = link()
    m.to_parquet(OUT / "ecg_links.parquet", index=False)
    ae = admission_level(m)
    ae.to_parquet(OUT / "admission_ecg.parquet", index=False)
    s = summarize(m, ae)
    (PHASE_DIRS[2] / "p2_02_ecg_linkage.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))
