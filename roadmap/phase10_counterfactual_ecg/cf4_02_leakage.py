"""Phase 10, CF4 step 2 — where the MI information in z comes from (selected model, validation split).

CF-A2 needs z independent of the MI state GIVEN the condition (covariates), not marginally. The roadmap gate tests z
alone; this script also measures how much z adds to the covariates for predicting MI (gradient-boosting probes fitted
on train, scored on validation, patient-level bootstrap of the AUROC difference with fitted probes held fixed).

Output: cf4_02_leakage.json
"""
import json

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from roadmap.config import REPO, SEED
from roadmap.phase10_counterfactual_ecg import cf4_01_train as t
from roadmap.phase10_counterfactual_ecg.cf4_model import MI_STATES, CondVAE

REPS = 300


def main() -> dict:
    data = t.Data(torch.device("cuda"))
    ck = torch.load(REPO / "models/cf_vae/cvae_selected.pt", map_location="cuda", weights_only=False)
    model = CondVAE(ck["z_dim"]).cuda()
    model.load_state_dict(ck["state_dict"])
    tr, va = data.t["train"], data.t["validation"]
    ztr, zva = t.latents(model, tr), t.latents(model, va)
    k = len(MI_STATES)
    ctr, cva = tr["c"].cpu().numpy()[:, k:], va["c"].cpu().numpy()[:, k:]          # covariates, no MI state
    ytr, yva = (tr["y"].cpu().numpy() > 0).astype(int), (va["y"].cpu().numpy() > 0).astype(int)

    def fit(a, b):
        return HistGradientBoostingClassifier(max_iter=300, random_state=SEED, class_weight="balanced").fit(
            a, ytr).predict_proba(b)[:, 1]

    p_z, p_c, p_cz = fit(ztr, zva), fit(ctr, cva), fit(np.c_[ctr, ztr], np.c_[cva, zva])
    sid = va["df"].subject_id.to_numpy()
    uniq, codes = np.unique(sid, return_inverse=True)
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(REPS):
        ix = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        boot.append(roc_auc_score(yva[ix], p_cz[ix]) - roc_auc_score(yva[ix], p_c[ix]))
    df = va["df"]
    out = {"model": ck.get("beta") and f"cvae_b{ck['beta']}_l{ck['lam']:g}",
           "auroc_z_only": round(roc_auc_score(yva, p_z), 4),
           "auroc_covariates_only": round(roc_auc_score(yva, p_c), 4),
           "auroc_covariates_plus_z": round(roc_auc_score(yva, p_cz), 4),
           "delta_auroc_z_given_covariates": round(roc_auc_score(yva, p_cz) - roc_auc_score(yva, p_c), 4),
           "delta_ci95": np.round(np.percentile(boot, [2.5, 97.5]), 4).tolist(),
           "z_only_score_correlation": {c: round(float(np.corrcoef(p_z, df[c])[0, 1]), 3)
                                        for c in ["age_z", "hr_z", "sex_male", "prior_mi_any"]}}
    (t.DOC / "cf4_02_leakage.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
