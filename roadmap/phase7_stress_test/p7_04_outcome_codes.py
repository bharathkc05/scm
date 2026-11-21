"""Phase 7, step 4 — alternative MI code definitions.

Layer 1 outcome:  narrow  = acute MI coded as the principal diagnosis
                  broad   = any 410 / I21 / I22 code, including ICD-9 410.x2 follow-up admissions
                  type 1  = acute MI excluding type 2 (I21.A1); type-2 admissions dropped
Layer 2 exposure: broad   = after-onset acute MI or 410.x2 follow-up admission (follow-ups otherwise excluded)
"""
import json

import pandas as pd

from roadmap.config import PHASE_DIRS
from roadmap.phase7_stress_test.common import L1_MAIN, P3, compare_l2, l1_rr, l2_att


def main():
    l1 = pd.read_parquet(P3 / "layer1.parquet")
    l1["mi_narrow"] = ((l1["acute_mi"] == 1) & (l1["acute_mi_principal"] == 1)).astype(int)
    l1["mi_broad"] = ((l1["acute_mi"] == 1) | (l1["mi_subsequent_episode_icd9"] == 1)).astype(int)
    variants = {"narrow_principal": (l1, "mi_narrow"), "broad_any_410_I21_I22": (l1, "mi_broad"),
                "type1_only": (l1[l1["mi_type2"] == 0], "acute_mi")}
    rows = []
    for vi, (name, (df, outc)) in enumerate(variants.items()):
        for qi, q in enumerate(L1_MAIN):
            r = l1_rr(df, q["exposure"], q["covs"], outcome=outc, seed=100 + 10 * vi + qi)
            rows.append({"variant": name, "label": q["label"], **r})
    l1res = pd.DataFrame(rows)

    l2 = pd.read_parquet(P3 / "layer2.parquet")
    df = l2[(l2["set_primary"] == 1) | ((l2["recent_mi_followup"] == 1) & (l2["has_any_dx"] == 1))].copy()
    df["mi_broad"] = ((df["set_primary"] == 1) & (df["acute_mi"] == 1) | (df["recent_mi_followup"] == 1)).astype(int)
    r2 = l2_att(df, exposure="mi_broad", seed=40)
    r2.to_csv(PHASE_DIRS[7] / "OUTCOME_CODES_LAYER2_BROAD.csv", index=False)
    return l1res, {"layer2_broad": {"n": len(df), "exposed": int(df["mi_broad"].sum()), **compare_l2(r2)}}


if __name__ == "__main__":
    l1res, s = main()
    l1res.to_csv(PHASE_DIRS[7] / "OUTCOME_CODES_LAYER1.csv", index=False)
    s["layer1"] = l1res.round(4).to_dict("records")
    (PHASE_DIRS[7] / "p7_04_outcome_codes.json").write_text(json.dumps(s, indent=2, default=float))
    print(l1res[["variant", "label", "outcome_events", "rr", "rr_ci_low", "rr_ci_high"]].round(3).to_string())
    print(json.dumps(s["layer2_broad"], indent=2, default=float))
