"""Phase 7, step 2 — ECG-selection sensitivity (inverse probability of selection weighting).

Target population: every Layer 1 admission that WOULD enter the Layer 2 primary set if it had a feature ECG
(after-onset-eligible MI: acute MI, principal, unplanned; or a valid control). Selection = has a feature ECG.
Selection model (cross-fitted logistic, patient folds): age spline, sex, diabetes, hypertension, CKD, prior MI,
acute MI, arrived via ED, planned admission, admission-type indicators, look-back (any earlier admission).
Stabilized weights are applied to the Phase 5 cross-fitted ATT scores (nuisances reused, not refitted).
Valid only if selection is ignorable given these variables (no unmeasured acuity driving both ECG and features).
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import Design, Spec, att_terms, ess, logistic, patient_bootstrap_ratio, patient_folds, summarize
from roadmap.phase5_estimation.queries import l2_outcomes
from roadmap.phase7_stress_test.common import REPS, compare_l2


def main():
    l1 = pd.read_parquet(DATA_OUT / "phase3/layer1.parquet")
    elig = (((l1["acute_mi"] == 1) & (l1["acute_mi_principal"] == 1) & (l1["planned_admission"] == 0))
            | ((l1["acute_mi"] == 0) & (l1["recent_mi_followup"] == 0)))
    t = l1[elig].reset_index(drop=True)
    for a in t["admission_type"].dropna().unique():
        t[f"atype_{a.replace(' ', '_').replace('.', '')}"] = (t["admission_type"] == a).astype(int)
    t["any_prior"] = (t["n_prior_admissions"] > 0).astype(int)
    covs = ["sex_male", "diabetes_primary", "hypertension_primary", "ckd_primary", "prior_mi_primary", "acute_mi",
            "arrived_via_ed", "planned_admission", "any_prior"] + sorted(c for c in t if c.startswith("atype_"))[1:]
    folds = patient_folds(t["subject_id"], SEED)
    s_hat = np.empty(len(t))
    sel = t["has_feature_ecg"].to_numpy()
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        des = Design(Spec(5, False), covs).fit(t[tr])
        s_hat[te] = logistic(des.transform(t[tr]), sel[tr]).predict_proba(des.transform(t[te]))[:, 1]
    t["p_select"] = s_hat
    rate = sel.mean()

    # Weighted ATT using Phase 5 nuisances on the primary set
    l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet")
    df = l2[l2["set_primary"] == 1].reset_index(drop=True)
    fam = l2_outcomes()
    cols = fam["point"] + fam["st_composite"]
    nu = pd.read_parquet(DATA_OUT / "phase5/layer2_nuisance.parquet", columns=["hadm_id", "e"] + [f"m0__{c}" for c in cols])
    df = df.merge(nu, on="hadm_id", validate="one_to_one").merge(t[["hadm_id", "p_select"]], on="hadm_id", how="left")
    w = rate / df["p_select"].clip(lower=0.01).to_numpy()
    x = df["acute_mi"].to_numpy(float)
    a, d = att_terms(x, df[cols].to_numpy(float), df["e"].to_numpy(), df[[f"m0__{c}" for c in cols]].to_numpy(float))
    num, den = a * w[:, None], d * w
    est = num.sum(0) / den.sum()
    boot = patient_bootstrap_ratio(df["subject_id"].to_numpy(), num, den, REPS, SEED + 900)
    sm = summarize(est, boot)
    sd = df.loc[x == 0, cols].std(ddof=1).to_numpy()
    res = pd.DataFrame({"outcome": [c.replace("w_", "") for c in cols], "att_sd": est / sd,
                        "att_sd_ci_low": sm["ci_low"] / sd, "att_sd_ci_high": sm["ci_high"] / sd, "att_p": sm["p_value"]})
    summary = {
        "target_admissions": len(t), "selected": int(sel.sum()), "selection_rate": float(rate),
        "p_select_quantiles": np.percentile(s_hat, [0, 1, 50, 99, 100]).round(3).tolist(),
        "weight_quantiles_selected": np.percentile(w, [0, 1, 50, 99, 100]).round(3).tolist(),
        "ess_selected": ess(w), "n_selected_primary": len(df),
        "missing_p_select": int(df["p_select"].isna().sum()),
        **compare_l2(res)}
    return res, summary


if __name__ == "__main__":
    res, s = main()
    res.to_csv(PHASE_DIRS[7] / "SELECTION_WEIGHTED_LAYER2.csv", index=False)
    (PHASE_DIRS[7] / "p7_02_selection.json").write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps(s, indent=2, default=float))
