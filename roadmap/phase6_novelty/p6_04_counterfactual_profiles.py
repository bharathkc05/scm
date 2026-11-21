"""Phase 6, step 4 — model-implied counterfactual ECGs (exploratory appendix).

For illustrative patient profiles, the fitted outcome models give the expected ECG feature with MI (m1)
and without MI (m0). These are MODEL-IMPLIED conditional expectations, not observed counterfactuals.
Uncertainty: 200 patient-level bootstrap refits. Also summarizes per-MI-patient model-implied deviations
(observed - cross-fitted m0) from the saved Phase 5 nuisances.

Outputs: COUNTERFACTUAL_PROFILES.csv, p6_04_counterfactual.json
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import SPECS, Design, ols, ols_predict
from roadmap.phase5_estimation.queries import L2_COVS

REPS = 200
OUTCOMES = ["w_ST_lateral_mean", "w_ST_max_depression", "w_ST_max_elevation", "w_ST_Elev_I", "w_ST_Elev_aVL",
            "w_ST_Elev_V5", "w_T_Amp_V5", "w_T_Amp_V6", "w_T_Amp_aVR", "w_T_Amp_V4"]
PROFILES = pd.DataFrame([
    {"profile": "55y man, no comorbidity", "age": 55, "sex_male": 1, "diabetes_primary": 0, "hypertension_primary": 0, "ckd_primary": 0, "prior_mi_primary": 0},
    {"profile": "65y man, diabetes + HTN", "age": 65, "sex_male": 1, "diabetes_primary": 1, "hypertension_primary": 1, "ckd_primary": 0, "prior_mi_primary": 0},
    {"profile": "75y woman, HTN + CKD", "age": 75, "sex_male": 0, "diabetes_primary": 0, "hypertension_primary": 1, "ckd_primary": 1, "prior_mi_primary": 0},
    {"profile": "70y man, D + H + CKD + prior MI", "age": 70, "sex_male": 1, "diabetes_primary": 1, "hypertension_primary": 1, "ckd_primary": 1, "prior_mi_primary": 1},
])


def fit_predict(df, spec, y):
    des = Design(spec, L2_COVS).fit(df)
    x, xp = des.transform(df), des.transform(PROFILES)
    t = df["acute_mi"].to_numpy() == 1
    return ols_predict(ols(x[~t], y[~t]), xp), ols_predict(ols(x[t], y[t]), xp)


def main():
    spec_name = json.loads((PHASE_DIRS[5] / "p5_01_model_selection.json").read_text())["chosen"]["layer2"]
    spec = next(s for s in SPECS if s.name == spec_name)
    l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet")
    df = l2[l2["set_primary"] == 1].reset_index(drop=True)
    y = df[OUTCOMES].to_numpy(float)
    sd = y[df["acute_mi"].to_numpy() == 0].std(0, ddof=1)
    m0, m1 = fit_predict(df, spec, y)

    pats = df["subject_id"].unique()
    rng = np.random.default_rng(SEED + 600)
    rows_by_pat = df.groupby("subject_id").indices
    diffs = np.empty((REPS, len(PROFILES), len(OUTCOMES)))
    for b in range(REPS):
        idx = np.concatenate([rows_by_pat[p] for p in rng.choice(pats, len(pats), replace=True)])
        bm0, bm1 = fit_predict(df.iloc[idx], spec, y[idx])
        diffs[b] = bm1 - bm0

    rows = []
    for i, prof in PROFILES.iterrows():
        for j, o in enumerate(OUTCOMES):
            lo, hi = np.percentile(diffs[:, i, j], [2.5, 97.5])
            rows.append({"profile": prof["profile"], "outcome": o.replace("w_", ""), "expected_without_mi": m0[i, j],
                         "expected_with_mi": m1[i, j], "model_implied_mi_effect": m1[i, j] - m0[i, j],
                         "ci_low": lo, "ci_high": hi, "effect_sd": (m1[i, j] - m0[i, j]) / sd[j]})
    prof = pd.DataFrame(rows)

    # Per-patient model-implied deviations among MI admissions (cross-fitted m0 from Phase 5)
    nu = pd.read_parquet(DATA_OUT / "phase5/layer2_nuisance.parquet", columns=["hadm_id"] + [f"m0__{o}" for o in OUTCOMES])
    mi = df[df["acute_mi"] == 1].merge(nu, on="hadm_id")
    dev = {o.replace("w_", ""): {
        "median_deviation_sd": float(np.median((mi[o] - mi[f"m0__{o}"]) / sd[j])),
        "share_beyond_1sd_in_effect_direction": float(np.mean(
            np.sign(prof.loc[prof["outcome"] == o.replace("w_", ""), "model_implied_mi_effect"].mean())
            * (mi[o] - mi[f"m0__{o}"]) / sd[j] > 1))} for j, o in enumerate(OUTCOMES)}
    return prof, {"profiles": PROFILES["profile"].tolist(), "bootstrap_reps": REPS,
                  "per_patient_deviation_among_mi": dev,
                  "caveat": "model-implied conditional expectations; not observed counterfactuals"}


if __name__ == "__main__":
    prof, s = main()
    prof.to_csv(PHASE_DIRS[6] / "COUNTERFACTUAL_PROFILES.csv", index=False)
    (PHASE_DIRS[6] / "p6_04_counterfactual.json").write_text(json.dumps(s, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(prof[prof["outcome"].isin(["ST_lateral_mean", "T_Amp_V5", "ST_max_depression"])].round(4).to_string())
    print(json.dumps(s["per_patient_deviation_among_mi"], indent=2))
