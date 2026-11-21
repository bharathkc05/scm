"""Phase 9, step 1 — export the three demo parameter files (aggregate only, no patient rows).

demo_params.json (committed):
  structural  : no-MI outcome model m0 for the 16 eligible measures, fitted on the TRAINING split controls
                (Phase 5 specification: 5-df age spline, binaries, age x binary, binary x binary; scaler; OLS)
  preprocessing: training-split winsorization bounds, plausible-input limits, beat-count limits, lead-label note
  effects     : Phase 5 ATT (raw units and control-SD units) with 95% CI for the same measures
Local verification files (git-ignored, data/roadmap/phase9/): held-out test cases and a fresh file in the raw
training format, with Python reference values for the acceptance tests.
"""
import json

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS, SEED
from roadmap.estimators import Design, ols
from roadmap.phase5_estimation.queries import L2_COVS
from roadmap.phase7_stress_test.common import ROBUST, spec

OUT = DATA_OUT / "phase9"
DOC = PHASE_DIRS[9]
POINT_INPUTS = json.loads((PHASE_DIRS[2] / "ecg_feature_list.json").read_text())["point_53"]
COUNT = "QRS_Dur_Global_count"
LATERAL = ["I", "aVL", "V5", "V6"]
NON_AVR = ["I", "II", "III", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GROUP = lambda m: "ST" if m.startswith("ST") else ("T" if m.startswith("T_") else "Q")  # noqa: E731


def spline_kept_columns(des: Design) -> list[int]:
    """Which of the full B-spline basis columns sklearn keeps (include_bias=False drops one)."""
    t, k = des.spline.bsplines_[0].t, des.spline.bsplines_[0].k
    n = len(t) - k - 1
    grid = np.linspace(t[k] + 1e-6, t[n] - 1e-6, 50)
    full = np.column_stack([BSpline(t, np.eye(n)[j], k)(grid) for j in range(n)])
    sk = des.spline.transform(grid[:, None])
    kept = [int(np.argmin(np.abs(full - sk[:, [c]]).sum(0))) for c in range(sk.shape[1])]
    assert np.allclose(full[:, kept], sk, atol=1e-10)
    return kept


def main():
    l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet")
    df = l2[l2["set_primary"] == 1].reset_index(drop=True)
    train = df[df["split"] == "train"]
    ctrl = train[train["acute_mi"] == 0]
    cols = [f"w_{m}" for m in ROBUST]
    des = Design(spec("layer2"), L2_COVS).fit(train)            # same fitting rows as the scaler/knots
    beta = ols(des.transform(ctrl), ctrl[cols].to_numpy(float))  # (1 + p) x 16
    b = des.spline.bsplines_[0]

    bounds = json.loads((PHASE_DIRS[3] / "winsorization_bounds.json").read_text())["bounds"]
    p5 = pd.read_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv")
    p5 = p5[p5["family"].isin(["point", "st_composite"])].set_index("outcome")
    params = {
        "version": "phase9-v1", "research_prototype": True,
        "structural": {
            "covariates": L2_COVS, "age_spline": {"knots": b.t.tolist(), "degree": int(b.k),
                                                  "kept_columns": spline_kept_columns(des), "extrapolation": "constant"},
            "design_order": "age_spline, binaries, age_spline x each binary (in covariate order), binary pairs (i<j)",
            "scaler_mean": des.scaler.mean_.tolist(), "scaler_scale": des.scaler.scale_.tolist(),
            "measures": ROBUST, "intercept": beta[0].tolist(), "coef": beta[1:].T.tolist(),
            "fitted_on": f"layer2 primary set, training split, controls (n={len(ctrl)})"},
        "preprocessing": {
            "point_inputs": POINT_INPUTS, "count_input": COUNT,
            "composites": {"ST_lateral_mean": {"mean_of": [f"ST_Elev_{x}" for x in LATERAL]},
                           "ST_max_depression": {"min_of": [f"ST_Elev_{x}" for x in NON_AVR]}},
            "winsor_bounds": {m: bounds[m] for m in POINT_INPUTS + ["ST_lateral_mean", "ST_max_depression"]},
            "training_range_p001_p999": {m: [float(train[m].quantile(0.001)), float(train[m].quantile(0.999))]
                                         for m in POINT_INPUTS},
            "training_medians": {m: float(train[m].median()) for m in POINT_INPUTS + [COUNT]},  # example input only
            "age_limits": [18, 100], "beat_count_min": float(train[COUNT].quantile(0.01)),
            "beat_ratio_limits": [0.8, 1.25],
            "ecgdeli_raw_file_lead_swap": "Raw ECGdeli CSVs from src/matlab/extract_ecg_features.m label aVL/aVF swapped; "
                                          "the demo swaps them back when 'raw ECGdeli file' is selected."},
        "effects": {m: {"group": GROUP(m), "att": float(p5.loc[f"w_{m}", "att"]),
                        "att_ci": [float(p5.loc[f"w_{m}", "att_ci_low"]), float(p5.loc[f"w_{m}", "att_ci_high"])],
                        "att_sd": float(p5.loc[f"w_{m}", "att_sd"]), "control_sd": float(p5.loc[f"w_{m}", "control_sd"])}
                    for m in ROBUST},
    }
    (DOC / "demo_params.json").write_text(json.dumps(params, indent=1))

    # ---- local acceptance-test fixtures (patient-level; git-ignored) ----
    OUT.mkdir(parents=True, exist_ok=True)
    test = df[df["split"] == "test"].sample(25, random_state=SEED)
    expected = des.transform(test) @ beta[1:] + beta[0]
    cases = [{"hadm_id": int(r.hadm_id), "study_id": int(r.study_id), "age": float(r.age),
              **{c: int(getattr(r, c)) for c in L2_COVS},
              "features": {m: float(getattr(r, m)) for m in POINT_INPUTS + [COUNT]},
              "expected_without_mi": dict(zip(ROBUST, map(float, e)))}
             for r, e in zip(test.itertuples(), expected)]
    (OUT / "test_cases.json").write_text(json.dumps(cases))
    raw = pd.read_csv(PATHS["ecg_features_full"], nrows=0).columns.tolist()
    fresh_ids = set(test["study_id"].head(5))
    chunks = pd.read_csv(PATHS["ecg_features_full"], chunksize=100_000)
    fresh = pd.concat([c[c["study_id"].isin(fresh_ids)] for c in chunks])[raw]   # raw format, swapped labels
    fresh.to_csv(OUT / "fresh_file_raw_format.csv", index=False)
    return {"measures": len(ROBUST), "design_columns": len(params["structural"]["scaler_mean"]),
            "train_controls": len(ctrl), "test_cases": len(cases), "fresh_rows": len(fresh)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
