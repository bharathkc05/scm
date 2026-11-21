"""Phase 3, step 1 — one split per patient for ALL 180,733 patients.

Rule: patients in the legacy split_manifest keep their split (never moved). Patients not in it are
assigned 70/15/15 with a fixed seed, stratified by whether they ever have an acute MI admission.

Output (git-ignored): data/roadmap/phase3/patient_splits.parquet
Summary:             roadmap/phase3_dataset/p3_01_splits.json
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS, SEED

OUT = DATA_OUT / "phase3"
SHARES = {"train": 0.70, "validation": 0.15, "test": 0.15}


def main() -> dict:
    adm = pd.read_parquet(DATA_OUT / "phase2/admissions.parquet", columns=["subject_id", "acute_mi"])
    patients = adm.groupby("subject_id")["acute_mi"].max().rename("ever_acute_mi").reset_index()

    man = pd.read_csv(PATHS["split_manifest"], usecols=["SubjectID", "split"]).drop_duplicates()
    per_patient = man.groupby("SubjectID")["split"].nunique()
    assert (per_patient == 1).all(), "legacy manifest splits a patient across splits"
    man = man.rename(columns={"SubjectID": "subject_id"})

    p = patients.merge(man, on="subject_id", how="left")
    p["split_source"] = np.where(p["split"].notna(), "legacy_manifest", "new_assignment")
    rng = np.random.default_rng(SEED)
    for stratum, grp in p[p["split"].isna()].groupby("ever_acute_mi"):
        ids = np.sort(grp["subject_id"].to_numpy())
        rng.shuffle(ids)
        cut = np.cumsum([int(round(len(ids) * SHARES[k])) for k in ("train", "validation")])
        labels = np.array(["train"] * cut[0] + ["validation"] * (cut[1] - cut[0]) + ["test"] * (len(ids) - cut[1]))
        p.loc[p["subject_id"].isin(ids), "split"] = pd.Series(labels, index=ids).reindex(
            p.loc[p["subject_id"].isin(ids), "subject_id"]).to_numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    p.to_parquet(OUT / "patient_splits.parquet", index=False)
    return {
        "patients": len(p),
        "by_source": p["split_source"].value_counts().to_dict(),
        "by_split": p["split"].value_counts().to_dict(),
        "by_split_and_source": {f"{a}|{b}": int(n) for (a, b), n in p.groupby(["split_source", "split"]).size().items()},
        "ever_acute_mi_rate_by_split": p.groupby("split")["ever_acute_mi"].mean().round(4).to_dict(),
        "rule": "legacy manifest kept; new patients 70/15/15 by seed, stratified by ever_acute_mi",
        "seed": SEED,
    }


if __name__ == "__main__":
    s = main()
    (PHASE_DIRS[3] / "p3_01_splits.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))
