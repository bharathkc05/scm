"""Phase 6, step 2 — causal audit of the 64 beta-VAE latents (Layer 2 primary set).

Per latent (cross-fitted, out-of-fold R^2 with the Phase 5 specification and patient folds):
  R2_base   : variance explained by age, sex, diabetes, hypertension, CKD, prior MI
  R2_mi     : variance explained by MI alone (crude)
  dR2_mi    : extra variance explained by MI on top of the baseline model (adjusted)
  retained  : dR2_mi / R2_mi  (share of MI's crude signal that survives adjustment)
plus the Phase 5 ATT (SD units, BH q) and the most correlated handcrafted point feature (interpretation).
Aggregate: out-of-fold AUC for MI from baseline-residualized latents vs residualized handcrafted features.

Outputs: LATENT_AUDIT.csv, p6_02_latent_audit.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import SPECS, Design, logistic, ols, ols_predict, patient_folds
from roadmap.phase5_estimation.queries import L2_COVS, l2_outcomes


def oof_r2(df, y, folds, spec, with_mi: bool, base: bool) -> np.ndarray:
    pred = np.empty_like(y)
    mi = df["acute_mi"].to_numpy(float)[:, None]
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        if base:
            des = Design(spec, L2_COVS).fit(df[tr])
            xtr, xte = des.transform(df[tr]), des.transform(df[te])
        else:
            xtr, xte = np.empty((tr.sum(), 0)), np.empty((te.sum(), 0))
        if with_mi:
            xtr, xte = np.column_stack([xtr, mi[tr]]), np.column_stack([xte, mi[te]])
        pred[te] = ols_predict(ols(xtr, y[tr]), xte)
    return 1 - ((y - pred) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def residualize(df, y, folds, spec) -> np.ndarray:
    res = np.empty_like(y)
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        des = Design(spec, L2_COVS).fit(df[tr])
        res[te] = y[te] - ols_predict(ols(des.transform(df[tr]), y[tr]), des.transform(df[te]))
    return res


def oof_auc(x: np.ndarray, d: np.ndarray, folds: np.ndarray) -> float:
    p = np.empty(len(d))
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        p[te] = logistic((x[tr] - mu) / sd, d[tr]).predict_proba((x[te] - mu) / sd)[:, 1]
    return float(roc_auc_score(d, p))


def main():
    spec_name = json.loads((PHASE_DIRS[5] / "p5_01_model_selection.json").read_text())["chosen"]["layer2"]
    spec = next(s for s in SPECS if s.name == spec_name)
    l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet")
    df = l2[l2["set_primary"] == 1].reset_index(drop=True)
    folds = patient_folds(df["subject_id"], SEED)
    fam = l2_outcomes()
    zc, pc = fam["latent"], fam["point"]
    z = df[zc].to_numpy(float)

    r2_base = oof_r2(df, z, folds, spec, with_mi=False, base=True)
    r2_full = oof_r2(df, z, folds, spec, with_mi=True, base=True)
    r2_mi = oof_r2(df, z, folds, spec, with_mi=True, base=False)
    p5 = pd.read_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv").set_index("outcome")
    corr = np.corrcoef(np.column_stack([z, df[pc].to_numpy(float)]).T)[:64, 64:]
    top = np.abs(corr).argmax(1)

    audit = pd.DataFrame({
        "latent": zc, "r2_baseline": r2_base, "r2_mi_alone": r2_mi, "delta_r2_mi_adjusted": r2_full - r2_base,
        "retained_share": np.where(r2_mi > 1e-6, (r2_full - r2_base) / r2_mi, np.nan),
        "att_sd": p5.loc[zc, "att_sd"].to_numpy(), "att_q": p5.loc[zc, "att_q"].to_numpy(),
        "crude_sd": p5.loc[zc, "crude_sd"].to_numpy(),
        "top_handcrafted_feature": [pc[i].replace("w_", "") for i in top],
        "top_feature_corr": corr[np.arange(64), top]})
    audit["mi_to_baseline_ratio"] = audit["delta_r2_mi_adjusted"] / audit["r2_baseline"].clip(lower=1e-6)

    d = df["acute_mi"].to_numpy()
    auc = {"latents_residualized": oof_auc(residualize(df, z, folds, spec), d, folds),
           "handcrafted_point_residualized": oof_auc(residualize(df, df[pc].to_numpy(float), folds, spec), d, folds),
           "baseline_covariates_only": oof_auc(Design(spec, L2_COVS).fit(df).transform(df), d, folds)}
    s = {
        "latents": 64, "bh_significant": int((audit["att_q"] < 0.05).sum()),
        "median_r2_baseline": float(audit["r2_baseline"].median()),
        "median_delta_r2_mi": float(audit["delta_r2_mi_adjusted"].median()),
        "max_delta_r2_mi": float(audit["delta_r2_mi_adjusted"].max()),
        "latents_where_mi_explains_more_than_baseline": int((audit["delta_r2_mi_adjusted"] > audit["r2_baseline"]).sum()),
        "median_retained_share": float(audit["retained_share"].median()),
        "top10_by_delta_r2": audit.nlargest(10, "delta_r2_mi_adjusted")[
            ["latent", "delta_r2_mi_adjusted", "r2_baseline", "att_sd", "top_handcrafted_feature", "top_feature_corr"]
        ].round(4).to_dict("records"),
        "oof_auc_for_mi": auc,
    }
    return audit, s


if __name__ == "__main__":
    audit, s = main()
    audit.to_csv(PHASE_DIRS[6] / "LATENT_AUDIT.csv", index=False)
    (PHASE_DIRS[6] / "p6_02_latent_audit.json").write_text(json.dumps(s, indent=2, default=float))
    print(json.dumps(s, indent=2, default=float))
