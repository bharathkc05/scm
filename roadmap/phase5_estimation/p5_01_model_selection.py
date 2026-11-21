"""Phase 5, step 1 — choose the nuisance-model specification on the VALIDATION split.

Fit on train, score on validation (roadmap: "Validation chooses the model specification and tuning").
Layer 1: mean validation log loss of propensity + arm-specific outcome models over the primary queries.
Layer 2: propensity log loss and mean validation MSE of m0 over the 53 standardized point features.
The test split is not touched.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from roadmap.config import DATA_OUT, PHASE_DIRS
from roadmap.estimators import SPECS, Design, logistic, ols, ols_predict
from roadmap.phase5_estimation.queries import L1_QUERIES, L2_COVS, l2_outcomes

P3 = DATA_OUT / "phase3"


def layer1(spec) -> float:
    l1 = pd.read_parquet(P3 / "layer1.parquet")
    tr, va = l1[l1["split"] == "train"], l1[l1["split"] == "validation"]
    losses = []
    for q in [q for q in L1_QUERIES if q["role"] == "primary"]:
        des = Design(spec, q["covs"]).fit(tr)
        xtr, xva = des.transform(tr), des.transform(va)
        e = logistic(xtr, tr[q["exposure"]]).predict_proba(xva)[:, 1]
        losses.append(log_loss(va[q["exposure"]], e))
        for arm in (0, 1):
            s_tr, s_va = tr[q["exposure"]].to_numpy() == arm, va[q["exposure"]].to_numpy() == arm
            m = logistic(xtr[s_tr], tr["acute_mi"].to_numpy()[s_tr]).predict_proba(xva[s_va])[:, 1]
            losses.append(log_loss(va["acute_mi"].to_numpy()[s_va], m, labels=[0, 1]))
    return float(np.mean(losses))


def layer2(spec, l2: pd.DataFrame) -> dict:
    tr, va = l2[l2["split"] == "train"], l2[l2["split"] == "validation"]
    point = l2_outcomes()["point"]
    sd = tr.loc[tr["acute_mi"] == 0, point].std().to_numpy()
    des = Design(spec, L2_COVS).fit(tr)
    xtr, xva = des.transform(tr), des.transform(va)
    e = logistic(xtr, tr["acute_mi"]).predict_proba(xva)[:, 1]
    c_tr, c_va = tr["acute_mi"].to_numpy() == 0, va["acute_mi"].to_numpy() == 0
    m0 = ols_predict(ols(xtr[c_tr], tr[point].to_numpy()[c_tr]), xva[c_va])
    mse = (((va[point].to_numpy()[c_va] - m0) / sd) ** 2).mean()
    return {"propensity_logloss": float(log_loss(va["acute_mi"], e)), "outcome_mse_sd_units": float(mse)}


if __name__ == "__main__":
    l2 = pd.read_parquet(P3 / "layer2.parquet")
    l2 = l2[l2["set_primary"] == 1]
    res = {s.name: {"layer1_mean_logloss": layer1(s), **{f"layer2_{k}": v for k, v in layer2(s, l2).items()}}
           for s in SPECS}
    best_l1 = min(res, key=lambda k: res[k]["layer1_mean_logloss"])
    best_l2 = min(res, key=lambda k: res[k]["layer2_propensity_logloss"] + res[k]["layer2_outcome_mse_sd_units"])
    out = {"by_spec": res, "chosen": {"layer1": best_l1, "layer2": best_l2}}
    (PHASE_DIRS[5] / "p5_01_model_selection.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
