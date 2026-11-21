"""Phase 7, step 6 — Layer 1 history-definition sensitivities (Phase 1 / Phase 4 Q6).

strict_flags     : exposure and covariates use prior-admission-only flags
lookback_only    : primary flags, restricted to admissions with >= 1 earlier admission
lookback_covariate: primary flags + number of earlier admissions (1, 2-4, 5+; reference 0)
"""
import json

import pandas as pd

from roadmap.config import PHASE_DIRS
from roadmap.phase7_stress_test.common import L1_MAIN, P3, l1_rr

LB = ["prior_adm_1", "prior_adm_2_4", "prior_adm_5plus"]


def strict(c: str) -> str:
    return c.replace("_primary", "_strict")


def main():
    l1 = pd.read_parquet(P3 / "layer1.parquet")
    n = l1["n_prior_admissions"]
    l1["prior_adm_1"] = (n == 1).astype(int)
    l1["prior_adm_2_4"] = n.between(2, 4).astype(int)
    l1["prior_adm_5plus"] = (n >= 5).astype(int)
    rows = []
    for qi, q in enumerate(L1_MAIN):
        rows.append({"variant": "strict_flags", "label": q["label"],
                     **l1_rr(l1, strict(q["exposure"]), [strict(c) for c in q["covs"]], seed=300 + qi)})
        rows.append({"variant": "lookback_only", "label": q["label"],
                     **l1_rr(l1[n > 0], q["exposure"], q["covs"], seed=320 + qi)})
        rows.append({"variant": "lookback_covariate", "label": q["label"],
                     **l1_rr(l1, q["exposure"], q["covs"] + LB, seed=340 + qi)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = main()
    r.to_csv(PHASE_DIRS[7] / "LAYER1_HISTORY_VARIANTS.csv", index=False)
    (PHASE_DIRS[7] / "p7_06_layer1_history.json").write_text(json.dumps(r.round(4).to_dict("records"), indent=2, default=float))
    print(r[["variant", "label", "n", "exposed", "rr", "rr_ci_low", "rr_ci_high"]].round(3).to_string())
