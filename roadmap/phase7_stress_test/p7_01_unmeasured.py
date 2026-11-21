"""Phase 7, step 1 — unmeasured-confounding bounds (E-values).

Layer 1: E-value for the risk ratio (outcome rare, 2.6%) and for the CI limit nearest 1.
Layer 2: continuous outcomes; standardized effect d converted to an approximate RR = exp(0.91 d)
(VanderWeele & Ding 2017), then the E-value. Benchmarked against measured associations (Layer 1 RRs).
E-values say how strongly an unmeasured confounder would need to relate to both exposure and outcome
(on the RR scale) to move the estimate (or its CI) to the null. They do not address selection.
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import PHASE_DIRS
from roadmap.phase7_stress_test.common import ROBUST


def evalue(rr: float) -> float:
    if np.isnan(rr):
        return np.nan
    rr = rr if rr >= 1 else 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def ci_evalue(lo: float, hi: float) -> float:
    if lo <= 1 <= hi:
        return 1.0
    return evalue(lo if lo > 1 else hi)


def main():
    l1 = pd.read_csv(PHASE_DIRS[5] / "LAYER1_ESTIMATES.csv")
    l1 = l1[l1["estimand"] == "ATE_overlap"]
    r1 = pd.DataFrame({"label": l1["label"], "role": l1["role"], "rr": l1["rr"],
                       "rr_ci": l1.apply(lambda r: f"{r['rr_ci_low']:.2f}-{r['rr_ci_high']:.2f}", axis=1),
                       "evalue_point": l1["rr"].map(evalue),
                       "evalue_ci": [ci_evalue(a, b) for a, b in zip(l1["rr_ci_low"], l1["rr_ci_high"])]})
    l2 = pd.read_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv")
    l2 = l2[l2["family"].isin(["point", "st_composite"])]          # winsorized (primary) versions only
    l2["outcome"] = l2["outcome"].str.replace("w_", "", regex=False)
    l2 = l2[l2["outcome"].isin(ROBUST)].set_index("outcome").loc[ROBUST].reset_index()
    near = np.where(l2["att_sd"] > 0, l2["att_sd_ci_low"], l2["att_sd_ci_high"])
    r2 = pd.DataFrame({"outcome": l2["outcome"], "att_sd": l2["att_sd"],
                       "approx_rr": np.exp(0.91 * l2["att_sd"].abs()),
                       "evalue_point": [evalue(np.exp(0.91 * abs(d))) for d in l2["att_sd"]],
                       "evalue_ci": [evalue(np.exp(0.91 * abs(c))) if np.sign(c) == np.sign(d) else 1.0
                                     for c, d in zip(near, l2["att_sd"])]})
    out = pd.concat([r1.assign(layer=1), r2.assign(layer=2)], ignore_index=True)
    s = {"layer1": r1.round(3).to_dict("records"),
         "layer2_evalue_point_range": [float(r2["evalue_point"].min()), float(r2["evalue_point"].max())],
         "layer2_evalue_ci_range": [float(r2["evalue_ci"].min()), float(r2["evalue_ci"].max())],
         "benchmark_measured_rr": {"diabetes": float(l1.iloc[0]["rr"]), "prior_mi": float(l1[l1["label"].str.startswith("prior")]["rr"].iloc[0])}}
    return out, s


if __name__ == "__main__":
    out, s = main()
    out.to_csv(PHASE_DIRS[7] / "EVALUES.csv", index=False)
    (PHASE_DIRS[7] / "p7_01_unmeasured.json").write_text(json.dumps(s, indent=2, default=float))
    print(out.round(3).to_string())
