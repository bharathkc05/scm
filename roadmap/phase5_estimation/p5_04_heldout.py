"""Phase 5, step 4 — single use of the TEST split: held-out independence check and stability check.

Pre-declared pass criteria (Gate 2, held-out component):
  H1  revised graph's implication A _||_ S holds on test Layer 1: |SMD| < 0.10
  H2  Layer 1: every primary-estimand RR on the test split lies on the same side of 1 as the full estimate
  H3  Layer 2 ATT (53 point features): sign agreement >= 90% among full-data BH-significant features,
      and median |test - full| <= 0.05 control SD
Nuisances are fitted on train+validation and applied to test rows only (no test data in fitting).
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import (SPECS, Design, ate_binary_terms, att_terms, logistic, ols, ols_predict,
                                patient_bootstrap_ratio)
from roadmap.phase5_estimation.queries import L1_QUERIES, L2_COVS, l2_outcomes

P3 = DATA_OUT / "phase3"
REPS = 500


def spec_of(layer: str):
    name = json.loads((PHASE_DIRS[5] / "p5_01_model_selection.json").read_text())["chosen"][layer]
    return next(s for s in SPECS if s.name == name)


def layer1(full: pd.DataFrame) -> list[dict]:
    l1 = pd.read_parquet(P3 / "layer1.parquet")
    fit, test = l1[l1["split"] != "test"], l1[l1["split"] == "test"]
    spec, out = spec_of("layer1"), []
    for i, q in enumerate([q for q in L1_QUERIES]):
        des = Design(spec, q["covs"]).fit(fit)
        xf, xt = des.transform(fit), des.transform(test)
        x_f, x_t = fit[q["exposure"]].to_numpy(), test[q["exposure"]].to_numpy(float)
        y_f, y_t = fit["acute_mi"].to_numpy(), test["acute_mi"].to_numpy(float)
        e = logistic(xf, x_f).predict_proba(xt)[:, 1]
        m1 = logistic(xf[x_f == 1], y_f[x_f == 1]).predict_proba(xt)[:, 1]
        m0 = logistic(xf[x_f == 0], y_f[x_f == 0]).predict_proba(xt)[:, 1]
        ov = (e >= 0.01) & (e <= 0.99)                                  # primary estimand: ATE in overlap
        t = ate_binary_terms(x_t[ov], y_t[ov], e[ov], m0[ov], m1[ov])
        b = patient_bootstrap_ratio(test["subject_id"].to_numpy()[ov], np.column_stack([t["mu1"], t["mu0"]]),
                                    np.ones(ov.sum()), REPS, SEED + 300 + i)
        rr = t["mu1"].mean() / t["mu0"].mean()
        f = full[(full["label"] == q["label"]) & (full["estimand"] == "ATE_overlap")].iloc[0]
        out.append({"label": q["label"], "role": q["role"], "n_test": int(ov.sum()), "rr_test": rr,
                    "rr_test_ci": [float(np.percentile(b[:, 0] / b[:, 1], 2.5)), float(np.percentile(b[:, 0] / b[:, 1], 97.5))],
                    "rr_full": float(f["rr"]), "same_side_of_1": bool((rr - 1) * (f["rr"] - 1) > 0)})
    return out


def layer2(full: pd.DataFrame) -> dict:
    l2 = pd.read_parquet(P3 / "layer2.parquet")
    df = l2[l2["set_primary"] == 1]
    fit, test = df[df["split"] != "test"], df[df["split"] == "test"]
    cols = l2_outcomes()["point"]
    des = Design(spec_of("layer2"), L2_COVS).fit(fit)
    xf, xt = des.transform(fit), des.transform(test)
    x_f, x_t = fit["acute_mi"].to_numpy(), test["acute_mi"].to_numpy(float)
    e = logistic(xf, x_f).predict_proba(xt)[:, 1]
    m0 = ols_predict(ols(xf[x_f == 0], fit[cols].to_numpy()[x_f == 0]), xt)
    a, d = att_terms(x_t, test[cols].to_numpy(float), e, m0)
    att_test = a.sum(0) / d.sum()
    f = full.set_index("outcome").loc[cols]
    sd = f["control_sd"].to_numpy()
    diff = np.abs(att_test / sd - f["att_sd"].to_numpy())
    sig = (f["att_q"] < 0.05).to_numpy()
    agree = np.sign(att_test) == np.sign(f["att"].to_numpy())
    per = pd.DataFrame({"outcome": cols, "att_sd_full": f["att_sd"].to_numpy(), "att_sd_test": att_test / sd,
                        "significant_full": sig, "sign_agree": agree})
    per.to_csv(PHASE_DIRS[5] / "LAYER2_TEST_STABILITY.csv", index=False)
    return {"n_test": len(test), "exposed_test": int(x_t.sum()),
            "sign_agreement_among_significant": float(agree[sig].mean()), "n_significant": int(sig.sum()),
            "sign_agreement_all": float(agree.mean()),
            "median_abs_diff_sd": float(np.median(diff)), "max_abs_diff_sd": float(diff.max()),
            "corr_full_vs_test": float(np.corrcoef(att_test / sd, f["att_sd"].to_numpy())[0, 1])}


def main() -> dict:
    l1 = pd.read_parquet(P3 / "layer1.parquet", columns=["split", "age", "sex_male"])
    t = l1[l1["split"] == "test"]
    m, w = t.loc[t["sex_male"] == 1, "age"], t.loc[t["sex_male"] == 0, "age"]
    smd = float((m.mean() - w.mean()) / np.sqrt((m.var() + w.var()) / 2))
    full1 = pd.read_csv(PHASE_DIRS[5] / "LAYER1_ESTIMATES.csv")
    full2 = pd.read_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv")
    r1, r2 = layer1(full1), layer2(full2)
    checks = {"H1_age_sex_smd_test": smd, "H1_pass": abs(smd) < 0.10,
              "H2_pass": all(r["same_side_of_1"] for r in r1),
              "H3_pass": r2["sign_agreement_among_significant"] >= 0.90 and r2["median_abs_diff_sd"] <= 0.05}
    return {"checks": checks, "layer1_test": r1, "layer2_test": r2}


if __name__ == "__main__":
    s = main()
    (PHASE_DIRS[5] / "p5_04_heldout.json").write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps(s, indent=2, default=float))
