"""Reproduce the counts quoted in ESTIMANDS.md.

Run from the repo root with the project env:
    C:/Users/Acer/Desktop/Capstone/Capstone/cuda_env/Scripts/python.exe -m roadmap.phase1_questions.phase1_counts
Writes roadmap/phase1_questions/phase1_counts.json.
"""
import json

import pandas as pd

from roadmap.config import PATHS, PHASE_DIRS
from roadmap.definitions import age_at_admission, flag_codes

COLS = ["acute_mi", "mi_stemi", "mi_nstemi", "mi_type2", "mi_other_unspecified",
        "mi_subsequent_episode_icd9", "old_mi", "diabetes", "hypertension", "ckd", "chest_pain"]


def main() -> dict:
    dx = pd.read_parquet(PATHS["diagnoses_all"], columns=["hadm_id", "icd_code", "icd_version"])
    f = flag_codes(dx)
    adm = f.groupby("hadm_id")[COLS].any()

    ch = pd.read_parquet(PATHS["admissions_chronology"],
                         columns=["subject_id", "hadm_id", "admittime", "admission_seq"])
    pt = pd.read_parquet(PATHS["patients_demographics"], columns=["subject_id", "anchor_age", "anchor_year"])
    ch = ch.merge(pt, on="subject_id", how="left")
    ch["age"] = age_at_admission(ch["admittime"], ch["anchor_age"], ch["anchor_year"])

    out = {
        "admissions_total": int(len(ch)),
        "admissions_with_dx": int(len(adm)),
        "admissions_under_18": int((ch["age"] < 18).sum()),
        "admissions_with_prior_admission": int((ch["admission_seq"] > 1).sum()),
        **{k: int(v) for k, v in adm.sum().items()},
        "acute_mi_by_icd_version": {int(k): int(v) for k, v in
                                    f[f["acute_mi"]].groupby("icd_version")["hadm_id"].nunique().items()},
        "subsequent_episode_only": int((adm["mi_subsequent_episode_icd9"] & ~adm["acute_mi"]).sum()),
        "acute_mi_without_subtype": int((adm["acute_mi"] & ~(adm["mi_stemi"] | adm["mi_nstemi"]
                                         | adm["mi_type2"] | adm["mi_other_unspecified"])).sum()),
        "chest_pain_without_acute_mi": int((adm["chest_pain"] & ~adm["acute_mi"]).sum()),
    }
    (PHASE_DIRS[1] / "phase1_counts.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
