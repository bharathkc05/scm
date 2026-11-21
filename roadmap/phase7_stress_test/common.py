"""Shared helpers for Phase 7: rerun the exact Phase 5 estimators on variant data/definitions."""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import SPECS, ate_binary_terms, crossfit_binary, patient_bootstrap_ratio, patient_folds
from roadmap.phase5_estimation.p5_03_layer2 import estimate
from roadmap.phase5_estimation.queries import L1_QUERIES, l2_outcomes

REPS = 500
P3 = DATA_OUT / "phase3"
L1_MAIN = [q for q in L1_QUERIES if q["role"] in ("primary", "bracket")]
ROBUST = ["Q_Amp_I", "ST_Elev_I", "T_Amp_II", "ST_Elev_III", "T_Amp_aVR", "ST_Elev_aVR", "Q_Amp_aVL",
          "ST_Elev_aVL", "T_Amp_V4", "ST_Elev_V4", "T_Amp_V5", "ST_Elev_V5", "T_Amp_V6", "ST_Elev_V6",
          "ST_lateral_mean", "ST_max_depression"]                      # Phase 6, S2


def spec(layer: str):
    name = json.loads((PHASE_DIRS[5] / "p5_01_model_selection.json").read_text())["chosen"][layer]
    return next(s for s in SPECS if s.name == name)


def l1_rr(df: pd.DataFrame, exposure: str, covs: list, outcome: str = "acute_mi", seed: int = 0) -> dict:
    """Phase 5 primary Layer 1 estimand: ATE in the overlap region (propensity 0.01-0.99)."""
    df = df.reset_index(drop=True)
    folds = patient_folds(df["subject_id"], SEED)
    nu = crossfit_binary(df, exposure, outcome, covs, spec("layer1"), folds)
    ov = (nu["e"] >= 0.01) & (nu["e"] <= 0.99)
    x, y = df[exposure].to_numpy(float)[ov], df[outcome].to_numpy(float)[ov]
    t = ate_binary_terms(x, y, nu["e"][ov], nu["m0"][ov], nu["m1"][ov])
    b = patient_bootstrap_ratio(df["subject_id"].to_numpy()[ov], np.column_stack([t["mu1"], t["mu0"]]),
                                np.ones(ov.sum()), REPS, SEED + 700 + seed)
    m1, m0 = t["mu1"].mean(), t["mu0"].mean()
    return {"n": int(ov.sum()), "exposed": int(x.sum()), "outcome_events": int(y.sum()),
            "rd_pp": 100 * (m1 - m0), "rr": m1 / m0,
            "rr_ci_low": float(np.percentile(b[:, 0] / b[:, 1], 2.5)),
            "rr_ci_high": float(np.percentile(b[:, 0] / b[:, 1], 97.5))}


def l2_att(df: pd.DataFrame, exposure: str = "acute_mi", seed: int = 0, covs_extra=None) -> pd.DataFrame:
    """Phase 5 Layer 2 ATT for the 53 point features + 5 ST composites."""
    from roadmap.phase5_estimation import p5_03_layer2 as m
    df = df.reset_index(drop=True)
    fam = l2_outcomes()
    cols = fam["point"] + fam["st_composite"]
    family_of = {c: ("point" if c in fam["point"] else "st_composite") for c in cols}
    old = m.L2_COVS
    if covs_extra:
        m.L2_COVS = old + covs_extra                    # estimate() reads the module-level covariate list
    try:
        res, _ = estimate(df, cols, family_of, spec("layer2"), patient_folds(df["subject_id"], SEED),
                          800 + seed, exposure=exposure, reps=REPS)
    finally:
        m.L2_COVS = old
    res["outcome"] = res["outcome"].str.replace("w_", "", regex=False)
    return res


def compare_l2(res: pd.DataFrame, primary: pd.DataFrame | None = None) -> dict:
    """Compare a variant's ATT with the Phase 5 primary on the robust set and all ST/T features."""
    if primary is None:
        primary = pd.read_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv")
        primary = primary[primary["family"].isin(["point", "st_composite"])]     # winsorized versions only
        primary["outcome"] = primary["outcome"].str.replace("w_", "", regex=False)
    p = primary.set_index("outcome")
    r = res.set_index("outcome")
    rob = [o for o in ROBUST if o in r.index]
    agree = np.sign(r.loc[rob, "att_sd"]) == np.sign(p.loc[rob, "att_sd"])
    excl0 = (r.loc[rob, "att_sd_ci_low"] > 0) | (r.loc[rob, "att_sd_ci_high"] < 0)
    stt = [o for o in r.index if o in p.index and p.loc[o, "group"] in ("ST", "T") and p.loc[o, "family"] == "point"]
    return {"robust_sign_agreement": float(agree.mean()), "robust_ci_excludes_0": float((agree & excl0).mean()),
            "ST_T_magnitude_ratio": float(r.loc[stt, "att_sd"].abs().mean() / p.loc[stt, "att_sd"].abs().mean()),
            "lateral_ST_att_sd": float(r.loc["ST_lateral_mean", "att_sd"]),
            "lateral_ST_ci": [float(r.loc["ST_lateral_mean", "att_sd_ci_low"]), float(r.loc["ST_lateral_mean", "att_sd_ci_high"])]}
