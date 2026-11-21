"""Phase 5, step 2 — Layer 1: chronic conditions -> acute MI at admission (AIPW, ATE).

Cross-fitting over all Layer 1 admissions (5 patient-grouped folds); 1,000 patient-level bootstrap
resamples with nuisances fixed. Nuisance predictions saved for later phases.

Outputs: LAYER1_ESTIMATES.csv, LAYER1_DIAGNOSTICS.csv (committed); data/roadmap/phase5/layer1_nuisance.parquet
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import (SPECS, ate_binary_terms, att_terms, crossfit_binary, ess, patient_bootstrap_ratio,
                                patient_folds, weighted_smd)
from roadmap.phase5_estimation.queries import L1_QUERIES

OUT = DATA_OUT / "phase5"
REPS = 1000


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = json.loads((PHASE_DIRS[5] / "p5_01_model_selection.json").read_text())["chosen"]["layer1"]
    spec = next(s for s in SPECS if s.name == chosen)
    l1 = pd.read_parquet(DATA_OUT / "phase3/layer1.parquet")
    folds = patient_folds(l1["subject_id"], SEED)
    y = l1["acute_mi"].to_numpy(float)
    sid = l1["subject_id"].to_numpy()
    est_rows, diag_rows, nuis = [], [], {"hadm_id": l1["hadm_id"].to_numpy(), "fold": folds}

    for i, q in enumerate(L1_QUERIES):
        x = l1[q["exposure"]].to_numpy(float)
        nu = crossfit_binary(l1, q["exposure"], "acute_mi", q["covs"], spec, folds)
        t = ate_binary_terms(x, y, nu["e"], nu["m0"], nu["m1"])
        boot = patient_bootstrap_ratio(sid, np.column_stack([t["mu1"], t["mu0"]]), np.ones(len(y)), REPS, SEED + i)
        mu1, mu0 = t["mu1"].mean(), t["mu0"].mean()
        rd_b, rr_b = boot[:, 0] - boot[:, 1], boot[:, 0] / boot[:, 1]
        crude = y[x == 1].mean() - y[x == 0].mean()
        est_rows.append({
            "label": q["label"], "role": q["role"], "exposure": q["exposure"], "adjust_for": "age(spline)," + ",".join(q["covs"]),
            "n": len(y), "exposed": int(x.sum()), "risk_exposed_crude": y[x == 1].mean(), "risk_unexposed_crude": y[x == 0].mean(),
            "crude_rd_pp": 100 * crude, "risk1_adj": mu1, "risk0_adj": mu0,
            "rd_pp": 100 * (mu1 - mu0), "rd_ci_low_pp": 100 * np.percentile(rd_b, 2.5), "rd_ci_high_pp": 100 * np.percentile(rd_b, 97.5),
            "rr": mu1 / mu0, "rr_ci_low": np.percentile(rr_b, 2.5), "rr_ci_high": np.percentile(rr_b, 97.5),
            "note": q["note"]})
        est_rows[-1]["estimand"] = "ATE"
        e = nu["e"]
        # --- ATE restricted to the overlap region (no clipping needed inside it)
        ov = (e >= 0.01) & (e <= 0.99)
        to = ate_binary_terms(x[ov], y[ov], e[ov], nu["m0"][ov], nu["m1"][ov])
        bo = patient_bootstrap_ratio(sid[ov], np.column_stack([to["mu1"], to["mu0"]]), np.ones(ov.sum()), REPS, SEED + 100 + i)
        m1o, m0o = to["mu1"].mean(), to["mu0"].mean()
        est_rows.append({**est_rows[-1], "estimand": "ATE_overlap", "n": int(ov.sum()), "exposed": int(x[ov].sum()),
                         "risk1_adj": m1o, "risk0_adj": m0o, "rd_pp": 100 * (m1o - m0o),
                         "rd_ci_low_pp": 100 * np.percentile(bo[:, 0] - bo[:, 1], 2.5),
                         "rd_ci_high_pp": 100 * np.percentile(bo[:, 0] - bo[:, 1], 97.5),
                         "rr": m1o / m0o, "rr_ci_low": np.percentile(bo[:, 0] / bo[:, 1], 2.5),
                         "rr_ci_high": np.percentile(bo[:, 0] / bo[:, 1], 97.5)})
        # --- ATT (effect among the exposed)
        a, dd = att_terms(x, y[:, None], e, nu["m0"][:, None])
        ba = patient_bootstrap_ratio(sid, np.column_stack([x * y, a[:, 0]]), dd, REPS, SEED + 200 + i)
        r1 = y[x == 1].mean()
        th = a[:, 0].sum() / dd.sum()
        est_rows.append({**est_rows[-1], "estimand": "ATT", "n": len(y), "exposed": int(x.sum()),
                         "risk1_adj": r1, "risk0_adj": r1 - th, "rd_pp": 100 * th,
                         "rd_ci_low_pp": 100 * np.percentile(ba[:, 1], 2.5), "rd_ci_high_pp": 100 * np.percentile(ba[:, 1], 97.5),
                         "rr": r1 / (r1 - th), "rr_ci_low": np.percentile(ba[:, 0] / (ba[:, 0] - ba[:, 1]), 2.5),
                         "rr_ci_high": np.percentile(ba[:, 0] / (ba[:, 0] - ba[:, 1]), 97.5)})

        bal_covs = ["age"] + q["covs"]
        ec = np.clip(e, 0.01, 0.99)
        weights = {
            "ATE": (np.ones(len(x), bool), np.where(x == 1, 1 / ec, 1 / (1 - ec))),
            "ATE_overlap": (ov, np.where(x == 1, 1 / e, 1 / (1 - e))),
            "ATT": (np.ones(len(x), bool), np.where(x == 1, 1.0, np.clip(e, None, 0.99) / (1 - np.clip(e, None, 0.99)))),
        }
        for estimand, (rows_, w) in weights.items():
            sub, xs, ws = l1[rows_], x[rows_], w[rows_]
            smd_w = weighted_smd(sub, bal_covs, xs, ws)
            diag_rows.append({
                "label": q["label"], "estimand": estimand, "n": int(rows_.sum()),
                "e_min": e.min(), "e_p01": np.percentile(e, 1), "e_p99": np.percentile(e, 99), "e_max": e.max(),
                "share_outside_overlap": float((~ov).mean()), "clipped_share": t["clipped_share"] if estimand == "ATE" else 0.0,
                "ess_exposed": ess(ws[xs == 1]), "ess_unexposed": ess(ws[xs == 0]),
                "max_abs_smd_unweighted": max(abs(v) for v in weighted_smd(sub, bal_covs, xs, np.ones(len(xs))).values()),
                "max_abs_smd_weighted": max(abs(v) for v in smd_w.values()),
                "smd_weighted": json.dumps({k: round(v, 4) for k, v in smd_w.items()})})
        key = q["label"].split(" |")[0] + f"_{i}"
        nuis.update({f"{key}_e": nu["e"], f"{key}_m0": nu["m0"], f"{key}_m1": nu["m1"]})

    pd.DataFrame(nuis).to_parquet(OUT / "layer1_nuisance.parquet", index=False)
    return pd.DataFrame(est_rows), pd.DataFrame(diag_rows)


if __name__ == "__main__":
    est, diag = main()
    est.to_csv(PHASE_DIRS[5] / "LAYER1_ESTIMATES.csv", index=False)
    diag.to_csv(PHASE_DIRS[5] / "LAYER1_DIAGNOSTICS.csv", index=False)
    pd.set_option("display.width", 250)
    print(est[["label", "estimand", "n", "exposed", "crude_rd_pp", "rd_pp", "rd_ci_low_pp", "rd_ci_high_pp", "rr", "rr_ci_low", "rr_ci_high"]].round(3).to_string())
    print(diag[["label", "estimand", "n", "share_outside_overlap", "ess_exposed", "ess_unexposed", "max_abs_smd_weighted"]].round(4).to_string())
