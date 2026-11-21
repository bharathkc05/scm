"""Phase 7, step 3 — BMI sensitivity (Phase 4 decision Q5).

BMI = most recent plausible OMR value (10-80 kg/m2) recorded on or before the admission date, within 3 years.
Within admissions that have a BMI, estimates WITHOUT and WITH BMI category indicators are compared on the same
rows (categories: <18.5, 25-30, 30-35, >=35; reference 18.5-25), so the difference isolates BMI confounding.
Layer 1: diabetes, hypertension (BMI is part of the shared metabolic cause). Layer 2: primary set.
"""
import json

import duckdb
import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS
from roadmap.phase7_stress_test.common import L1_MAIN, P3, compare_l2, l1_rr, l2_att

BMI_CATS = ["bmi_lt18_5", "bmi_25_30", "bmi_30_35", "bmi_ge35"]


def link_bmi(adm: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("adm", adm[["subject_id", "hadm_id", "admittime"]])
    b = con.sql(f"""
        with omr as (
          select subject_id, cast(chartdate as date) d, try_cast(result_value as double) bmi
          from read_csv_auto('{PATHS["raw_omr"].as_posix()}')
          where result_name in ('BMI (kg/m2)', 'BMI') and try_cast(result_value as double) between 10 and 80)
        select a.hadm_id, arg_max(o.bmi, o.d) as bmi
        from adm a join omr o on o.subject_id = a.subject_id
         and o.d <= cast(a.admittime as date) and o.d >= cast(a.admittime as date) - interval 1095 day
        group by a.hadm_id""").df()
    out = adm[["hadm_id"]].merge(b, on="hadm_id", how="left")
    v = out["bmi"]
    out["bmi_lt18_5"] = (v < 18.5).astype(int)
    out["bmi_25_30"] = ((v >= 25) & (v < 30)).astype(int)
    out["bmi_30_35"] = ((v >= 30) & (v < 35)).astype(int)
    out["bmi_ge35"] = (v >= 35).astype(int)
    return out


def main():
    adm = pd.read_parquet(DATA_OUT / "phase2/admissions.parquet", columns=["subject_id", "hadm_id", "admittime"])
    bmi = link_bmi(adm)
    l1 = pd.read_parquet(P3 / "layer1.parquet").merge(bmi, on="hadm_id")
    l2 = pd.read_parquet(P3 / "layer2.parquet")
    l2 = l2[l2["set_primary"] == 1].merge(bmi, on="hadm_id")
    s = {"layer1_bmi_coverage": float(l1["bmi"].notna().mean()), "layer2_bmi_coverage": float(l2["bmi"].notna().mean()),
         "layer2_bmi_coverage_mi": float(l2.loc[l2["acute_mi"] == 1, "bmi"].notna().mean()),
         "layer2_bmi_median_mi_vs_control": [float(l2.loc[l2["acute_mi"] == 1, "bmi"].median()),
                                             float(l2.loc[l2["acute_mi"] == 0, "bmi"].median())]}
    l1b = l1[l1["bmi"].notna()]
    rows = []
    for i, q in enumerate([q for q in L1_MAIN if q["exposure"] in ("diabetes_primary", "hypertension_primary")]):
        a = l1_rr(l1b, q["exposure"], q["covs"], seed=10 + i)
        b = l1_rr(l1b, q["exposure"], q["covs"] + BMI_CATS, seed=20 + i)
        rows.append({"label": q["label"], "n": a["n"], "rr_without_bmi": a["rr"], "ci_without": [a["rr_ci_low"], a["rr_ci_high"]],
                     "rr_with_bmi": b["rr"], "ci_with": [b["rr_ci_low"], b["rr_ci_high"]]})
    s["layer1"] = rows
    l2b = l2[l2["bmi"].notna()]
    r0 = l2_att(l2b, seed=1)
    r1 = l2_att(l2b, seed=2, covs_extra=BMI_CATS)
    r0.to_csv(PHASE_DIRS[7] / "BMI_LAYER2_WITHOUT.csv", index=False)
    r1.to_csv(PHASE_DIRS[7] / "BMI_LAYER2_WITH.csv", index=False)
    s["layer2_rows"] = len(l2b)
    s["layer2_exposed"] = int(l2b["acute_mi"].sum())
    s["layer2_with_bmi_vs_without_same_rows"] = compare_l2(r1, primary=r0)
    s["layer2_without_bmi_vs_full_primary"] = compare_l2(r0)
    return s


if __name__ == "__main__":
    s = main()
    (PHASE_DIRS[7] / "p7_03_bmi.json").write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps(s, indent=2, default=float))
