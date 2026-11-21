"""Phase 6, step 3 — deconfounded ECG effects of MI: which ECG signs are attributable to MI?

Per point feature (and ST composite), combine Phase 5 (crude vs adjusted ATT) with Phase 6 step 1
(NSTEMI-only, negative control). Classification rule (pre-declared):
  crude significance: Welch z on the crude difference, BH within the 53 point features
  MI-attributable       adjusted BH-significant, same sign as crude, |adjusted| >= 50% of |crude|
  partly confounded     adjusted BH-significant, same sign as crude, |adjusted| <  50% of |crude|
  explained by confounding   crude significant, adjusted not significant
  revealed by adjustment     adjusted significant, crude not significant or opposite sign
  no association        neither significant
Robustness flags: same sign in NSTEMI-only; negative-control ratio |NC| / |primary| < 0.5.

Outputs: DECONFOUNDED_ECG_EFFECTS.csv, FEATURE_GROUP_SUMMARY.csv, p6_03_deconfounded.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from roadmap.config import DATA_OUT, PHASE_DIRS
from roadmap.estimators import bh
from roadmap.phase5_estimation.queries import l2_outcomes


def crude_tests(cols) -> pd.DataFrame:
    l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet", columns=cols + ["acute_mi", "set_primary"])
    df = l2[l2["set_primary"] == 1]
    t, c = df[df["acute_mi"] == 1][cols], df[df["acute_mi"] == 0][cols]
    se = np.sqrt(t.var() / len(t) + c.var() / len(c))
    z = (t.mean() - c.mean()) / se
    return pd.DataFrame({"crude_p": 2 * norm.sf(np.abs(z.to_numpy()))}, index=z.index)


def classify(r) -> str:
    adj_sig, cr_sig = r["att_q"] < 0.05, r["crude_q"] < 0.05
    same = np.sign(r["att"]) == np.sign(r["crude_diff"])
    if adj_sig and cr_sig and same:
        return "MI-attributable" if abs(r["att_sd"]) >= 0.5 * abs(r["crude_sd"]) else "partly confounded"
    if adj_sig:
        return "revealed by adjustment"
    if cr_sig:
        return "explained by confounding"
    return "no association"


def main():
    fam = l2_outcomes()
    cols = fam["point"] + fam["st_composite"]
    p5 = pd.read_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv").set_index("outcome").loc[cols]
    sens = pd.read_csv(PHASE_DIRS[6] / "SENSITIVITY_ESTIMATES.csv")
    ns = sens[sens["analysis_set"] == "nstemi_only"].set_index("outcome").loc[cols]
    nc = sens[sens["analysis_set"] == "negative_control_future_mi"].set_index("outcome").loc[cols]
    ac = sens[sens["analysis_set"] == "active_comparator_chest_pain"].set_index("outcome").loc[cols]

    d = p5[["family", "group", "control_sd", "crude_diff", "crude_sd", "att", "att_sd", "att_sd_ci_low",
            "att_sd_ci_high", "att_q"]].copy()
    d = d.join(crude_tests(cols))
    d["crude_q"] = np.nan
    pt = d["family"] == "point"
    d.loc[pt, "crude_q"] = bh(d.loc[pt, "crude_p"].to_numpy())
    d.loc[~pt, "crude_q"] = d.loc[~pt, "crude_p"]
    d.loc[~pt, "att_q"] = p5.loc[~pt, "att_p"]          # composites: nominal (5 pre-specified outcomes)
    d["adjusted_share_of_crude"] = d["att_sd"] / d["crude_sd"]
    d["nstemi_att_sd"] = ns["att_sd"]
    d["nstemi_same_sign"] = np.sign(ns["att"]) == np.sign(d["att"])
    d["chest_pain_comparator_att_sd"] = ac["att_sd"]
    d["negative_control_att_sd"] = nc["att_sd"]
    d["negative_control_ratio"] = (nc["att_sd"].abs() / d["att_sd"].abs()).round(3)
    d["class"] = d.apply(classify, axis=1)
    d["robust_mi_sign"] = (d["class"].isin(["MI-attributable", "partly confounded", "revealed by adjustment"])
                           & d["nstemi_same_sign"] & (d["negative_control_ratio"] < 0.5))
    d = d.reset_index().rename(columns={"index": "outcome"})
    d["outcome"] = d["outcome"].str.replace("w_", "", regex=False)

    pts = d[d["family"] == "point"]
    grp = pts.groupby("group").agg(
        features=("outcome", "size"), mi_attributable=("class", lambda s: int((s == "MI-attributable").sum())),
        partly_confounded=("class", lambda s: int((s == "partly confounded").sum())),
        explained_by_confounding=("class", lambda s: int((s == "explained by confounding").sum())),
        revealed=("class", lambda s: int((s == "revealed by adjustment").sum())),
        robust=("robust_mi_sign", "sum"), mean_abs_crude_sd=("crude_sd", lambda s: s.abs().mean()),
        mean_abs_att_sd=("att_sd", lambda s: s.abs().mean()),
        mean_abs_negative_control_sd=("negative_control_att_sd", lambda s: s.abs().mean())).round(3)
    s = {"class_counts_point": pts["class"].value_counts().to_dict(),
         "robust_point_features": pts.loc[pts["robust_mi_sign"], "outcome"].tolist(),
         "explained_by_confounding": pts.loc[pts["class"] == "explained by confounding", "outcome"].tolist(),
         "revealed_by_adjustment": pts.loc[pts["class"] == "revealed by adjustment", "outcome"].tolist(),
         "composites": d[d["family"] == "st_composite"].set_index("outcome")[
             ["crude_sd", "att_sd", "nstemi_att_sd", "negative_control_att_sd", "class", "robust_mi_sign"]].round(3).to_dict("index")}
    return d, grp, s


if __name__ == "__main__":
    d, grp, s = main()
    d.to_csv(PHASE_DIRS[6] / "DECONFOUNDED_ECG_EFFECTS.csv", index=False)
    grp.to_csv(PHASE_DIRS[6] / "FEATURE_GROUP_SUMMARY.csv")
    (PHASE_DIRS[6] / "p6_03_deconfounded.json").write_text(json.dumps(s, indent=2, default=str))
    print(grp.to_string())
    print(json.dumps(s, indent=2, default=str))
