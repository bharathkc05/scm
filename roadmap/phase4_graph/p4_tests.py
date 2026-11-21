"""Phase 4 — test the graph's implied independencies on the TRAINING split, judged by effect size.

Pre-declared thresholds (set before running; p-values reported but not used for the verdict):
  T1  A _||_ S             |standardized mean difference of age by sex| < 0.10
  T2  D _||_ H | A,S        adjusted OR within [0.80, 1.25] and |partial correlation| < 0.05
Descriptive graph checks (not implications): look-back C = number of earlier admissions.

Writes INDEPENDENCE_TESTS.csv and p4_tests.json. Test split is NOT used here (reserved for Phase 5).
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from roadmap.config import DATA_OUT, PHASE_DIRS

P3 = DATA_OUT / "phase3"
OR_BAND, PR_MAX, SMD_MAX = (0.80, 1.25), 0.05, 0.10


def smd_age_by_sex(d: pd.DataFrame) -> float:
    m, f = d.loc[d["sex_male"] == 1, "age"], d.loc[d["sex_male"] == 0, "age"]
    return float((m.mean() - f.mean()) / np.sqrt((m.var() + f.var()) / 2))


def adjusted_or(d: pd.DataFrame, y: str, x: str, extra: str = "") -> tuple[float, float, float, float]:
    fit = smf.logit(f"{y} ~ {x} + bs(age, df=5) + sex_male{extra}", data=d).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["subject_id"]})
    b, se = fit.params[x], fit.bse[x]
    return float(np.exp(b)), float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se)), float(fit.pvalues[x])


def partial_r(d: pd.DataFrame, a: str, b: str) -> float:
    ra = smf.ols(f"{a} ~ bs(age, df=5) + sex_male", data=d).fit().resid
    rb = smf.ols(f"{b} ~ bs(age, df=5) + sex_male", data=d).fit().resid
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> tuple[pd.DataFrame, dict]:
    l1 = pd.read_parquet(P3 / "layer1.parquet")
    l2 = pd.read_parquet(P3 / "layer2.parquet", columns=[
        "subject_id", "split", "set_primary", "age", "sex_male", "diabetes_primary", "hypertension_primary",
        "diabetes_strict", "hypertension_strict"])
    samples = {"layer1_train": l1[l1["split"] == "train"],
               "layer2_primary_train": l2[(l2["split"] == "train") & (l2["set_primary"] == 1)]}
    # Layer 2 rows are selected on R (has a feature ECG); A -> R <- S, so A _||_ S is not implied there.
    implied = {"layer1_train": True, "layer2_primary_train": False}
    rows = []
    for sname, d in samples.items():
        smd = smd_age_by_sex(d)
        rows.append({"test": "T1 A _||_ S", "sample": sname, "flags": "-", "n": len(d), "effect": "SMD age (M-F)",
                     "estimate": round(smd, 4), "ci_low": None, "ci_high": None, "p_value": None,
                     "partial_r": None, "threshold": f"|SMD|<{SMD_MAX}", "implied_in_this_sample": implied[sname],
                     "holds": abs(smd) < SMD_MAX})
        for ver in ("primary", "strict"):
            dd, hh = f"diabetes_{ver}", f"hypertension_{ver}"
            o, lo, hi, p = adjusted_or(d, hh, dd)
            pr = partial_r(d, dd, hh)
            rows.append({"test": "T2 D _||_ H | A,S", "sample": sname, "flags": ver, "n": len(d),
                         "effect": "adjusted OR (H on D)", "estimate": round(o, 3), "ci_low": round(lo, 3),
                         "ci_high": round(hi, 3), "p_value": p, "partial_r": round(pr, 4),
                         "threshold": f"OR in {OR_BAND} and |r|<{PR_MAX}", "implied_in_this_sample": implied[sname],
                         "holds": OR_BAND[0] <= o <= OR_BAND[1] and abs(pr) < PR_MAX})
    tests = pd.DataFrame(rows)

    # Descriptive: look-back C (any earlier admission) vs MI and vs recorded history, given age and sex
    d = samples["layer1_train"].assign(any_prior=lambda x: (x["n_prior_admissions"] > 0).astype(int))
    desc = {"C_vs_acute_mi_adjusted_OR": adjusted_or(d, "acute_mi", "any_prior")[:3]}
    for c in ("diabetes", "hypertension", "ckd"):
        desc[f"C_vs_{c}_primary_OR"] = adjusted_or(d, f"{c}_primary", "any_prior")[:3]
    desc = {k: [round(v, 3) for v in vals] for k, vals in desc.items()}
    # Strict flags are 0 by construction without an earlier admission (structural, not estimable as an OR)
    desc["strict_flag_positive_share_when_no_prior_admission"] = {
        c: float(d.loc[d["any_prior"] == 0, f"{c}_strict"].mean()) for c in ("diabetes", "hypertension", "ckd")}
    # D-H dependence restricted to admissions WITH look-back (removes the shared structural zeros)
    w = d[d["any_prior"] == 1]
    desc["T2_strict_flags_among_admissions_with_prior_admission_OR"] = [
        round(v, 3) for v in adjusted_or(w, "hypertension_strict", "diabetes_strict")[:3]]
    return tests, desc


if __name__ == "__main__":
    tests, desc = main()
    tests.to_csv(PHASE_DIRS[4] / "INDEPENDENCE_TESTS.csv", index=False)
    (PHASE_DIRS[4] / "p4_tests.json").write_text(json.dumps({"tests": tests.to_dict("records"),
                                                             "lookback_descriptive_OR_[est,lo,hi]": desc}, indent=2, default=str))
    print(tests.drop(columns=["threshold"]).to_string())
    print(json.dumps(desc, indent=2))
