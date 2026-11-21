"""Phase 5, step 3 — Layer 2: acute MI (after onset) -> ECG outcomes. ATT primary, ATE secondary.

Population: set_primary. One cross-fitting pass (5 patient-grouped folds) for all outcome families;
1,000 patient-level bootstrap resamples (nuisances fixed); BH FDR within each family; effects reported in
raw units and in control-SD units. Nuisances saved for Phases 6-7.

Outputs: LAYER2_ESTIMATES.csv, LAYER2_DIAGNOSTICS.json (committed); data/roadmap/phase5/layer2_nuisance.parquet
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import (SPECS, ate_cont_terms, att_terms, bh, crossfit_continuous, ess,
                                patient_bootstrap_ratio, patient_folds, summarize, weighted_smd)
from roadmap.phase5_estimation.queries import L2_COVS, L2_MACHINE, l2_outcomes

OUT = DATA_OUT / "phase5"
REPS = 1000
FDR_FAMILIES = ["point", "iqr", "latent"]


def group_of(col: str) -> str:
    c = col.replace("w_", "").replace("z_ecg_", "LATENT_")
    for p in ("ST_", "QRS", "RR", "Q_", "R_", "S_", "T_", "LATENT_", "machine"):
        if c.startswith(p):
            return {"ST_": "ST", "QRS": "QRS", "RR": "RR", "Q_": "Q", "R_": "R", "S_": "S", "T_": "T",
                    "LATENT_": "latent", "machine": "machine"}[p]
    return "other"


def estimate(df, cols, family_of, spec, folds, seed_offset, save_name=None):
    y = df[cols].to_numpy(float)
    x = df["acute_mi"].to_numpy(float)
    sid = df["subject_id"].to_numpy()
    nu = crossfit_continuous(df, "acute_mi", y, L2_COVS, spec, folds, need_m1=True)
    a, d = att_terms(x, y, nu["e"], nu["m0"])
    att = a.sum(0) / d.sum()
    att_boot = patient_bootstrap_ratio(sid, a, d, REPS, SEED + seed_offset)
    ate_t = ate_cont_terms(x, y, nu["e"], nu["m0"], nu["m1"])
    ate = ate_t.mean(0)
    ate_boot = patient_bootstrap_ratio(sid, ate_t, np.ones(len(x)), REPS, SEED + seed_offset + 1)
    ctrl = x == 0
    sd = y[ctrl].std(0, ddof=1)
    crude = y[x == 1].mean(0) - y[ctrl].mean(0)
    s_att, s_ate = summarize(att, att_boot), summarize(ate, ate_boot)
    res = pd.DataFrame({"outcome": cols, "family": [family_of[c] for c in cols], "group": [group_of(c) for c in cols],
                        "control_sd": sd, "crude_diff": crude, "crude_sd": crude / sd,
                        "att": s_att["estimate"], "att_ci_low": s_att["ci_low"], "att_ci_high": s_att["ci_high"],
                        "att_p": s_att["p_value"], "att_sd": att / sd, "att_sd_ci_low": s_att["ci_low"] / sd,
                        "att_sd_ci_high": s_att["ci_high"] / sd,
                        "ate": s_ate["estimate"], "ate_ci_low": s_ate["ci_low"], "ate_ci_high": s_ate["ci_high"],
                        "ate_p": s_ate["p_value"], "ate_sd": ate / sd})
    if save_name:
        nuis = pd.DataFrame({"hadm_id": df["hadm_id"].to_numpy(), "fold": folds, "e": nu["e"].astype("float32")})
        nuis = pd.concat([nuis, pd.DataFrame(nu["m0"].astype("float32"), columns=[f"m0__{c}" for c in cols]),
                          pd.DataFrame(nu["m1"].astype("float32"), columns=[f"m1__{c}" for c in cols])], axis=1)
        nuis.to_parquet(OUT / save_name, index=False)
    return res, nu


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = json.loads((PHASE_DIRS[5] / "p5_01_model_selection.json").read_text())["chosen"]["layer2"]
    spec = next(s for s in SPECS if s.name == chosen)
    l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet")
    df = l2[l2["set_primary"] == 1].reset_index(drop=True)
    folds = patient_folds(df["subject_id"], SEED)

    fam = l2_outcomes()
    family_of = {c: f for f, cs in fam.items() for c in cs}
    cols = [c for f in ["point", "iqr", "st_composite", "raw_point", "latent"] for c in fam[f]]
    res, nu = estimate(df, cols, family_of, spec, folds, 0, save_name="layer2_nuisance.parquet")

    # Machine-measured validation outcomes (rows with a valid measurement only)
    mres = []
    for i, c in enumerate(L2_MACHINE):
        sub = df[df[c].notna()].reset_index(drop=True)
        r, _ = estimate(sub, [c], {c: "machine"}, spec, patient_folds(sub["subject_id"], SEED), 10 + 2 * i)
        mres.append(r)
    res = pd.concat([res] + mres, ignore_index=True)

    res["att_q"] = np.nan
    res["ate_q"] = np.nan
    for f in FDR_FAMILIES:
        m = res["family"] == f
        res.loc[m, "att_q"] = bh(res.loc[m, "att_p"].to_numpy())
        res.loc[m, "ate_q"] = bh(res.loc[m, "ate_p"].to_numpy())

    # Diagnostics
    x = df["acute_mi"].to_numpy(float)
    e = nu["e"]
    w_att = np.where(x == 1, 1.0, np.clip(e, None, 0.99) / (1 - np.clip(e, None, 0.99)))
    ec = np.clip(e, 0.01, 0.99)
    w_ate = np.where(x == 1, 1 / ec, 1 / (1 - ec))
    bal = ["age"] + L2_COVS
    diag = {
        "spec": chosen, "n": len(df), "exposed": int(x.sum()), "patients": int(df["subject_id"].nunique()),
        "propensity_quantiles_exposed": np.percentile(e[x == 1], [0, 1, 50, 99, 100]).round(4).tolist(),
        "propensity_quantiles_control": np.percentile(e[x == 0], [0, 1, 50, 99, 100]).round(4).tolist(),
        "att_control_weight_max": float(w_att[x == 0].max()), "att_ess_controls": ess(w_att[x == 0]),
        "ate_clipped_share": float(((e < 0.01) | (e > 0.99)).mean()),
        "smd_unweighted": weighted_smd(df, bal, x, np.ones(len(x))),
        "smd_att_weighted": weighted_smd(df, bal, x, w_att),
        "smd_ate_weighted": weighted_smd(df, bal, x, w_ate),
    }
    for f in FDR_FAMILIES:
        m = res["family"] == f
        diag[f"bh05_significant_{f}"] = {"att": int((res.loc[m, "att_q"] < 0.05).sum()),
                                         "ate": int((res.loc[m, "ate_q"] < 0.05).sum()), "tests": int(m.sum())}
    return res, diag


if __name__ == "__main__":
    res, diag = main()
    res.to_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv", index=False)
    (PHASE_DIRS[5] / "LAYER2_DIAGNOSTICS.json").write_text(json.dumps(diag, indent=2, default=float))
    print(json.dumps(diag, indent=2, default=float))
    pd.set_option("display.width", 250)
    top = res[res["family"] == "point"].reindex(res[res["family"] == "point"]["att_sd"].abs().sort_values(ascending=False).index)
    print(top[["outcome", "crude_sd", "att_sd", "att_sd_ci_low", "att_sd_ci_high", "att_q", "ate_sd"]].head(15).round(3).to_string())
    print(res[res["family"].isin(["st_composite", "machine"])][["outcome", "att", "att_ci_low", "att_ci_high", "att_sd", "ate_sd"]].round(3).to_string())
