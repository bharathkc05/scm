"""Phase 6, step 1 — Layer 2 ATT across the pre-specified analysis sets (Gate 1 condition + stability).

Same estimator, specification, covariates and outcomes as Phase 5; only the analysis set changes.
Pre-declared criteria (before running):
  Gate 1 condition (NSTEMI-only): >= 90% of the ST and T features significant in the primary analysis keep
      their sign, and ST+T mean |ATT| keeps >= 50% of the primary magnitude.
  Stability (each sensitivity set): >= 90% sign agreement over the primary BH-significant point features.
Negative control: exposure = acute MI on a LATER admission, among controls without prior MI. Reported as an
upper bound on residual confounding + selection + pre-existing coronary disease.

Outputs: SENSITIVITY_ESTIMATES.csv, p6_01_sensitivity.json
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import SPECS, bh, patient_folds
from roadmap.phase5_estimation.p5_03_layer2 import estimate
from roadmap.phase5_estimation.queries import l2_outcomes

SETS = {  # name: (flag column, exposure)
    "nstemi_only": ("set_nstemi", "acute_mi"),
    "active_comparator_chest_pain": ("set_active_comparator", "acute_mi"),
    "after_strict_timing": ("set_after_strict", "acute_mi"),
    "type2_excluded": ("set_no_type2", "acute_mi"),
    "first_ecg_has_features": ("set_first_ecg_has_features", "acute_mi"),
    "negative_control_future_mi": ("set_negative_control", "future_acute_mi"),
}


def main():
    spec_name = json.loads((PHASE_DIRS[5] / "p5_01_model_selection.json").read_text())["chosen"]["layer2"]
    spec = next(s for s in SPECS if s.name == spec_name)
    l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet")
    fam = l2_outcomes()
    cols = fam["point"] + fam["st_composite"]
    family_of = {c: ("point" if c in fam["point"] else "st_composite") for c in cols}
    prim = pd.read_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv").set_index("outcome")

    frames, summary = [], {}
    for i, (name, (flag, expo)) in enumerate(SETS.items()):
        df = l2[l2[flag] == 1].reset_index(drop=True)
        res, _ = estimate(df, cols, family_of, spec, patient_folds(df["subject_id"], SEED), 400 + 10 * i,
                          exposure=expo)
        pt = res["family"] == "point"
        res.loc[pt, "att_q"] = bh(res.loc[pt, "att_p"].to_numpy())
        res.insert(0, "analysis_set", name)
        frames.append(res)

        r = res[pt].set_index("outcome")
        p = prim.loc[r.index]
        sig = p["att_q"] < 0.05
        stt = sig & p["group"].isin(["ST", "T"])
        agree = np.sign(r["att"]) == np.sign(p["att"])
        mag_ratio = r.loc[p["group"].isin(["ST", "T"]).values, "att_sd"].abs().mean() / \
            p.loc[p["group"].isin(["ST", "T"]), "att_sd"].abs().mean()
        summary[name] = {
            "n": len(df), "exposed": int(df[expo].sum()),
            "sign_agreement_primary_significant": float(agree[sig].mean()),
            "sign_agreement_ST_T_significant": float(agree[stt].mean()),
            "ST_T_mean_abs_att_ratio_to_primary": float(mag_ratio),
            "bh_significant_point": int((r["att_q"] < 0.05).sum()),
            "corr_with_primary_att_sd": float(np.corrcoef(r["att_sd"], p["att_sd"])[0, 1]),
            "max_abs_att_sd": float(r["att_sd"].abs().max()),
            "composites_att_sd": res[~pt].set_index("outcome")["att_sd"].round(3).to_dict(),
        }
    s = summary["nstemi_only"]
    gate1 = s["sign_agreement_ST_T_significant"] >= 0.90 and s["ST_T_mean_abs_att_ratio_to_primary"] >= 0.50
    stability = {k: v["sign_agreement_primary_significant"] >= 0.90 for k, v in summary.items()
                 if k != "negative_control_future_mi"}
    return pd.concat(frames, ignore_index=True), {"gate1_nstemi_condition_pass": gate1,
                                                  "stability_pass_by_set": stability, "by_set": summary}


if __name__ == "__main__":
    est, s = main()
    est.to_csv(PHASE_DIRS[6] / "SENSITIVITY_ESTIMATES.csv", index=False)
    (PHASE_DIRS[6] / "p6_01_sensitivity.json").write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps(s, indent=2, default=float))
