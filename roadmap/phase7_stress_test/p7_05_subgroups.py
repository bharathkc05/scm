"""Phase 7, step 5 — subgroups by age (<65 / >=65) and sex, both layers.

Each subgroup is re-estimated separately with the Phase 5 estimators. Interaction = difference between the
two subgroup estimates, with a normal-approximation CI from the two bootstrap SEs (patients rarely cross
age groups; sex groups are disjoint). Layer 2 summarized on the 16 robust measures; BH across them.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from roadmap.config import PHASE_DIRS
from roadmap.estimators import bh
from roadmap.phase7_stress_test.common import L1_MAIN, P3, ROBUST, compare_l2, l1_rr, l2_att

SUBGROUPS = {"age_lt65": lambda d: d["age"] < 65, "age_ge65": lambda d: d["age"] >= 65,
             "female": lambda d: d["sex_male"] == 0, "male": lambda d: d["sex_male"] == 1}
PAIRS = {"age": ("age_lt65", "age_ge65"), "sex": ("female", "male")}


def main():
    l1 = pd.read_parquet(P3 / "layer1.parquet")
    rows = []
    for gi, (g, f) in enumerate(SUBGROUPS.items()):
        for qi, q in enumerate(L1_MAIN):
            covs = [c for c in q["covs"] if not (g in ("female", "male") and c == "sex_male")]
            rows.append({"subgroup": g, "label": q["label"], **l1_rr(l1[f(l1)], q["exposure"], covs, seed=200 + 10 * gi + qi)})
    l1res = pd.DataFrame(rows)

    l2 = pd.read_parquet(P3 / "layer2.parquet")
    l2 = l2[l2["set_primary"] == 1]
    res, summ = {}, {}
    for gi, (g, f) in enumerate(SUBGROUPS.items()):
        sub = l2[f(l2)]
        r = l2_att(sub, seed=60 + gi)
        r["subgroup"] = g
        res[g] = r.set_index("outcome")
        summ[g] = {"n": len(sub), "exposed": int(sub["acute_mi"].sum()), **compare_l2(r)}
    inter = []
    for name, (a, b) in PAIRS.items():
        ra, rb = res[a].loc[ROBUST], res[b].loc[ROBUST]
        se = lambda r: (r["att_sd_ci_high"] - r["att_sd_ci_low"]) / (2 * 1.96)  # noqa: E731
        diff = rb["att_sd"] - ra["att_sd"]
        z = diff / np.sqrt(se(ra) ** 2 + se(rb) ** 2)
        p = 2 * norm.sf(np.abs(z.to_numpy()))
        inter.append(pd.DataFrame({"modifier": name, "outcome": ROBUST, f"att_sd_{a}": ra["att_sd"].to_numpy(),
                                   f"att_sd_{b}": rb["att_sd"].to_numpy(), "difference": diff.to_numpy(),
                                   "p": p, "q": bh(p)}))
    allres = pd.concat([r.reset_index() for r in res.values()], ignore_index=True)
    return l1res, allres, pd.concat(inter, ignore_index=True), summ


if __name__ == "__main__":
    l1res, l2res, inter, summ = main()
    l1res.to_csv(PHASE_DIRS[7] / "SUBGROUPS_LAYER1.csv", index=False)
    l2res.to_csv(PHASE_DIRS[7] / "SUBGROUPS_LAYER2.csv", index=False)
    inter.to_csv(PHASE_DIRS[7] / "SUBGROUP_INTERACTIONS_LAYER2.csv", index=False)
    s = {"layer2_by_subgroup": summ,
         "layer2_interactions_bh05": inter.loc[inter["q"] < 0.05, ["modifier", "outcome", "difference"]].round(3).to_dict("records"),
         "layer1": l1res.round(4).to_dict("records")}
    (PHASE_DIRS[7] / "p7_05_subgroups.json").write_text(json.dumps(s, indent=2, default=float))
    print(l1res[["subgroup", "label", "rr", "rr_ci_low", "rr_ci_high"]].round(3).to_string())
    print(json.dumps({k: s[k] for k in ("layer2_by_subgroup", "layer2_interactions_bh05")}, indent=2, default=float))
