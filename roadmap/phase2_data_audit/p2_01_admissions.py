"""Phase 2, step 1 — admission-level table with exposure, outcome and arrival context.

Output (git-ignored): data/roadmap/phase2/admissions.parquet
Summary:            roadmap/phase2_data_audit/p2_01_admissions.json
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS
from roadmap.definitions import age_at_admission, flag_codes

OUT = DATA_OUT / "phase2"
PLANNED_TYPES = {"ELECTIVE", "SURGICAL SAME DAY ADMISSION", "AMBULATORY OBSERVATION"}
PLANNED_LOCATIONS = {"PROCEDURE SITE", "PACU", "AMBULATORY SURGERY TRANSFER"}
CHRONIC = ["diabetes", "hypertension", "ckd"]


def build() -> pd.DataFrame:
    adm = pd.read_parquet(PATHS["admissions_chronology"], columns=[
        "subject_id", "hadm_id", "admittime", "dischtime", "edregtime", "admission_type",
        "admission_location", "admission_seq", "hospital_expire_flag"])
    pt = pd.read_parquet(PATHS["patients_demographics"], columns=["subject_id", "gender", "anchor_age", "anchor_year"])
    adm = adm.merge(pt, on="subject_id", how="left", validate="many_to_one")
    for c in ["admittime", "dischtime", "edregtime"]:
        adm[c] = pd.to_datetime(adm[c])
    adm["age"] = age_at_admission(adm["admittime"], adm["anchor_age"], adm["anchor_year"])
    adm["sex_male"] = (adm["gender"] == "M").astype("int8")

    # Arrival = ED registration when valid (before discharge, not after admission), else admission time.
    ed_ok = adm["edregtime"].notna() & (adm["edregtime"] < adm["dischtime"]) & (adm["edregtime"] <= adm["admittime"])
    adm["arrival_time"] = adm["admittime"].where(~ed_ok, adm["edregtime"])
    adm["arrived_via_ed"] = ed_ok.astype("int8")
    adm["planned_admission"] = (adm["admission_type"].isin(PLANNED_TYPES)
                                | adm["admission_location"].isin(PLANNED_LOCATIONS)).astype("int8")

    # Current-admission code flags
    dx = pd.read_parquet(PATHS["diagnoses_all"], columns=["hadm_id", "icd_code", "icd_version", "seq_num"])
    f = flag_codes(dx)
    flags = ["acute_mi", "mi_stemi", "mi_nstemi", "mi_type2", "mi_other_unspecified",
             "mi_subsequent_episode_icd9", "old_mi", "chest_pain"] + CHRONIC
    cur = f.groupby("hadm_id")[flags].any().astype("int8")
    first_mi_seq = f.loc[f["acute_mi"]].groupby("hadm_id")["seq_num"].min().rename("acute_mi_min_seq")
    cur = cur.join(first_mi_seq)
    adm = adm.merge(cur, left_on="hadm_id", right_index=True, how="left")
    adm["has_any_dx"] = adm["acute_mi"].notna().astype("int8")
    adm[flags] = adm[flags].fillna(0).astype("int8")
    adm["acute_mi_principal"] = (adm["acute_mi_min_seq"] == 1).astype("int8")

    # History flags: cumulative over the patient's EARLIER admissions (strict) and + current (primary)
    adm = adm.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)
    g = adm.groupby("subject_id", sort=False)
    sid = adm["subject_id"]
    for c in CHRONIC + ["acute_mi", "old_mi"]:
        upto = g[c].cummax()                                   # through current admission
        adm[f"{c}_earlier"] = upto.groupby(sid).shift(1, fill_value=0).astype("int8")
    for c in CHRONIC:
        adm[f"{c}_strict"] = adm[f"{c}_earlier"]
        adm[f"{c}_primary"] = (adm[f"{c}_earlier"] | adm[c]).astype("int8")
    adm["prior_mi_strict"] = (adm["acute_mi_earlier"] | adm["old_mi_earlier"]).astype("int8")
    adm["prior_mi_primary"] = (adm["acute_mi_earlier"] | adm["old_mi_earlier"] | adm["old_mi"]).astype("int8")
    adm["n_prior_admissions"] = g.cumcount().astype("int32")

    # Does this patient have an acute MI on a LATER admission? (negative-control pool)
    rev = adm.iloc[::-1]
    from_here_on = rev.groupby("subject_id", sort=False)["acute_mi"].cummax().iloc[::-1]
    adm["future_acute_mi"] = from_here_on.groupby(sid).shift(-1, fill_value=0).astype("int8")
    return adm


def summarize(adm: pd.DataFrame) -> dict:
    s = {"admissions": len(adm), "patients": int(adm["subject_id"].nunique()),
         "admissions_without_any_dx": int((adm["has_any_dx"] == 0).sum()),
         "acute_mi": int(adm["acute_mi"].sum()),
         "acute_mi_principal": int(adm["acute_mi_principal"].sum()),
         "acute_mi_planned_admission": int((adm["acute_mi"] & adm["planned_admission"]).sum()),
         "arrived_via_ed": int(adm["arrived_via_ed"].sum()),
         "age_missing": int(adm["age"].isna().sum()), "sex_missing": int(adm["gender"].isna().sum()),
         "seq_num_check_first_admission_is_0_prior": bool((adm.loc[adm["admission_seq"] == 1, "n_prior_admissions"] == 0).all())}
    hist = {}
    for c in CHRONIC + ["prior_mi"]:
        p, st = adm[f"{c}_primary"], adm[f"{c}_strict"]
        hist[c] = {"primary_prevalence": round(float(p.mean()), 4), "strict_prevalence": round(float(st.mean()), 4),
                   "primary_positive": int(p.sum()),
                   "share_of_primary_supported_by_earlier_admission": round(float(st[p == 1].mean()), 4)}
    s["history_flags"] = hist
    first = adm["n_prior_admissions"] == 0
    s["first_admissions"] = int(first.sum())
    s["acute_mi_rate_first_vs_repeat_admission"] = [round(float(adm.loc[first, "acute_mi"].mean()), 4),
                                                    round(float(adm.loc[~first, "acute_mi"].mean()), 4)]
    ctx = adm[adm["acute_mi"] == 1].groupby(["planned_admission", "acute_mi_principal"]).size()
    s["acute_mi_by_context"] = {f"planned={a}|principal={b}": int(n) for (a, b), n in ctx.items()}
    return s


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    adm = build()
    adm.to_parquet(OUT / "admissions.parquet", index=False)
    s = summarize(adm)
    (PHASE_DIRS[2] / "p2_01_admissions.json").write_text(json.dumps(s, indent=2, default=int))
    print(json.dumps(s, indent=2, default=int))
