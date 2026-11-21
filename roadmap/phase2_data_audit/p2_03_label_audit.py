"""Phase 2, step 3 — MI label audit.

1. Agreement between the roadmap outcome (`acute_mi`, hosp diagnoses_icd) and the legacy master
   label (`is_MI` from the ECG-linked ICD file + troponin timing).
2. Troponin testing per admission (was "Control_Symptomatic" really "troponin tested"?).
3. Legacy troponin label vs. the new admission-context timing class.
4. MI subtype counts among Layer 2-eligible admissions (power for the STEMI/NSTEMI check).

Summary: roadmap/phase2_data_audit/p2_03_label_audit.json
"""
import json

import duckdb
import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS

OUT = DATA_OUT / "phase2"


def troponin_by_admission(adm: pd.DataFrame) -> tuple[pd.Series, list]:
    con = duckdb.connect(str(PATHS["mimic_duckdb"]), read_only=True)
    items = con.sql("select itemid, label, fluid from d_labitems where label ilike '%troponin%'").df()
    ids = ",".join(str(i) for i in items["itemid"])
    con.register("adm", adm[["subject_id", "hadm_id", "arrival_time", "dischtime"]])
    # Link by patient + time window (ED labs often lack hadm_id).
    tested = con.sql(f"""
        select distinct a.hadm_id
        from labevents l join adm a
          on l.subject_id = a.subject_id
         and l.charttime between a.arrival_time and a.dischtime
        where l.itemid in ({ids})
    """).df()["hadm_id"]
    return adm["hadm_id"].isin(set(tested)).astype("int8"), items.to_dict("records")


def main() -> dict:
    adm = pd.read_parquet(OUT / "admissions.parquet")
    ae = pd.read_parquet(OUT / "admission_ecg.parquet")
    x = adm.merge(ae, on="hadm_id", how="left")
    x["troponin_tested"], items = troponin_by_admission(x)
    has_feat = x["first_feat_study_id"].notna()

    legacy = pd.read_csv(PATHS["legacy_troponin_labels"], usecols=["hadm_id", "is_MI"]).dropna(subset=["hadm_id"])
    leg_adm = legacy.groupby("hadm_id")["is_MI"].max().astype(int)
    both = x.set_index("hadm_id").join(leg_adm.rename("legacy_is_MI"), how="inner")

    s = {"troponin_items": items}
    s["legacy_vs_acute_mi_admissions"] = {
        "admissions_in_both": len(both),
        "crosstab_acute_mi_rows_legacy_cols": pd.crosstab(both["acute_mi"], both["legacy_is_MI"]).to_dict(),
        "agreement": round(float((both["acute_mi"] == both["legacy_is_MI"]).mean()), 4),
        "legacy_MI_but_no_acute_code_with_old_or_subsequent_code": int(
            ((both["legacy_is_MI"] == 1) & (both["acute_mi"] == 0)
             & ((both["old_mi"] == 1) | (both["mi_subsequent_episode_icd9"] == 1))).sum()),
    }
    grp = {
        "acute_mi": x["acute_mi"] == 1,
        "no_acute_mi": x["acute_mi"] == 0,
        "no_acute_mi_with_feature_ecg": (x["acute_mi"] == 0) & has_feat,
        "chest_pain_no_mi_with_feature_ecg": (x["acute_mi"] == 0) & (x["chest_pain"] == 1) & has_feat,
        "acute_mi_with_feature_ecg": (x["acute_mi"] == 1) & has_feat,
    }
    s["troponin_tested_share"] = {k: round(float(x.loc[v, "troponin_tested"].mean()), 4) for k, v in grp.items()}

    mi = x[(x["acute_mi"] == 1) & has_feat]
    s["timing_class_vs_legacy_label_first_feature_ecg"] = pd.crosstab(
        mi["timing_class"], mi["first_feat_legacy_label"].fillna("missing")).to_dict()
    sub = {}
    for tc in ["after", "uncertain"]:
        t = mi[mi["timing_class"] == tc]
        sub[tc] = {k: int(t[k].sum()) for k in ["mi_stemi", "mi_nstemi", "mi_type2", "mi_other_unspecified"]}
        sub[tc]["n"] = len(t)
    s["mi_subtypes_with_feature_ecg_by_timing"] = sub
    return s


if __name__ == "__main__":
    s = main()
    (PHASE_DIRS[2] / "p2_03_label_audit.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))
