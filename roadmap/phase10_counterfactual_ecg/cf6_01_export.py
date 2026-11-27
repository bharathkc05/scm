"""Phase 10, CF6 step 1 — export what the MCP server needs for individual reasoning (aggregate parameters only).

local_mcp/assets/cf_track_a.json  (T3)  Track A (CF3, model A2): portable design (age spline knots, kept columns,
                                        scaler), m0 / m1 coefficients with MI-type dummies, control SDs, rho and
                                        split-conformal 90% half-widths per baseline kind and direction, input rules
                                        (ST artefact limit, winsorization bounds, composites), validation metrics.
local_mcp/assets/cf_clinical.json (T3)  Layer 1 do() models (Phase 5 design, Phase 3 training split, same fit as the
                                        patient explorer): arm-specific logistic coefficients and 200 draws from the
                                        patient-clustered sandwich covariance, per primary query plus the diabetes
                                        bracket set.
local_mcp/assets/figures/cf_pair_validation.png (T3)  Pair validation figure for render_figure.

No patient rows are exported. Parity of the portable formulas with the fitted models is tested in
local_mcp/tests/test_counterfactual_tools.py.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, REPO, SEED
from roadmap.estimators import Design, logistic
from roadmap.phase5_estimation.queries import L1_QUERIES, L2_COVS
from roadmap.phase7_stress_test.common import spec
from roadmap.phase9_demo.p9_01_export import spline_kept_columns
from roadmap.phase10_counterfactual_ecg import cf3_01_track_a as A

OUT = DATA_OUT / "phase10"
ASSETS = REPO / "local_mcp" / "assets"
DOC = PHASE_DIRS[10]
DRAWS = 200
L1_EXPORT = {("diabetes_primary", "primary"): "diabetes:primary", ("diabetes_primary", "bracket"): "diabetes:alternative",
             ("hypertension_primary", "primary"): "hypertension:primary", ("ckd_primary", "primary"): "ckd:primary"}
INPUTS = [f"ST_Elev_{l}" for l in A.LEADS] + ["T_Amp_II", "T_Amp_aVR", "T_Amp_V4", "T_Amp_V5", "T_Amp_V6",
                                              "Q_Amp_I", "Q_Amp_aVL"]


def design_spec(des: Design, covs: list[str]) -> dict:
    b = des.spline.bsplines_[0]
    return {"covariates": covs, "age_spline": {"knots": b.t.tolist(), "degree": int(b.k),
                                               "kept_columns": spline_kept_columns(des)},
            "age_by_all": des.spec.age_by_all, "interactions": des.spec.interactions,
            "scaler_mean": des.scaler.mean_.tolist(), "scaler_scale": des.scaler.scale_.tolist(),
            "spec": des.spec.name}


def export_track_a() -> dict:
    with open(OUT / "cf3_track_a_model.pkl", "rb") as fh:
        saved = pickle.load(fh)
    m = saved["model"]
    bounds = json.loads((PHASE_DIRS[3] / "winsorization_bounds.json").read_text())["bounds"]
    s = json.loads((DOC / "cf3_01_track_a.json").read_text())
    tuned = {side: {d: {"rho": t["rho"].tolist(), "q90": t["q"]["A2_shrunk"].tolist(), "n_val_pairs": t["n_val"]}
                    for d, t in dd.items()} for side, dd in saved["tuned"].items()}
    out = {
        "version": "cf3-A2-v1", "research_prototype": True,
        "measures": A.ROBUST, "st_t_measures": A.ST_T, "types": A.TYPES, "type_reference": A.TYPES[0],
        "design": design_spec(m.des, L2_COVS),
        "m0": {"intercept": m.b0[0].tolist(), "coef": m.b0[1:].T.tolist()},
        "m1": {"intercept": m.b1[0].tolist(), "coef": m.b1[1:].T.tolist(),
               "type_dummies": A.TYPES[1:], "note": "coef columns: design, then one dummy per non-reference type"},
        "control_sd": saved["sd"].tolist(),
        "tuned": tuned,
        "inputs": {"features": INPUTS, "st_max_abs_mv": A.ST_MAX_ABS_MV,
                   "composites": {"ST_lateral_mean": {"mean_of": [f"ST_Elev_{l}" for l in A.LATERAL]},
                                  "ST_max_depression": {"min_of": [f"ST_Elev_{l}" for l in A.NON_AVR]}},
                   "winsor_bounds": {k: bounds[k] for k in A.ROBUST}},
        "gate_cf_g1": s["gate_cf_g1"], "composite_st_t_primary": s["composite_st_t"]["primary"],
        "fitted_on": {"training": s["training"], "pairs": s["pairs"]},
    }
    (ASSETS / "cf_track_a.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def sandwich_draws(xt: np.ndarray, y: np.ndarray, groups: np.ndarray, model, rng) -> np.ndarray:
    """Draws of [intercept, coef] from N(beta, V) with V the patient-clustered sandwich covariance."""
    x1 = np.column_stack([np.ones(len(xt)), xt])
    beta = np.concatenate([model.intercept_, model.coef_[0]])
    p = 1 / (1 + np.exp(-x1 @ beta))
    h = (x1 * (p * (1 - p))[:, None]).T @ x1
    codes, _ = pd.factorize(groups)
    sg = np.zeros((codes.max() + 1, x1.shape[1]))
    np.add.at(sg, codes, x1 * (y - p)[:, None])
    hi = np.linalg.pinv(h)
    v = hi @ (sg.T @ sg) @ hi
    return rng.multivariate_normal(beta, (v + v.T) / 2, size=DRAWS, method="eigh")


def export_clinical() -> dict:
    l1 = pd.read_parquet(DATA_OUT / "phase3/layer1.parquet")
    tr = l1[l1.split == "train"].reset_index(drop=True)
    rng = np.random.default_rng(SEED + 60)
    queries = {}
    for q in L1_QUERIES:
        key = L1_EXPORT.get((q["exposure"], q["role"]))
        if key is None:
            continue
        des = Design(spec("layer1"), q["covs"]).fit(tr)
        xt = des.transform(tr)
        x, y = tr[q["exposure"]].to_numpy(), tr["acute_mi"].to_numpy()
        arms = {}
        for a in (0, 1):
            sel = x == a
            mdl = logistic(xt[sel], y[sel])
            arms[str(a)] = {"intercept": float(mdl.intercept_[0]), "coef": mdl.coef_[0].tolist(),
                            "draws": np.round(sandwich_draws(xt[sel], y[sel], tr.subject_id.to_numpy()[sel], mdl,
                                                             rng), 6).tolist(),
                            "n": int(sel.sum())}
        queries[key] = {"exposure": q["exposure"], "label": q["label"], "role": q["role"], "note": q["note"],
                        "design": design_spec(des, q["covs"]), "arms": arms}
    out = {"version": "l1-do-v1", "outcome": "acute-MI admission (Phase 5 Layer 1)",
           "fitted_on": f"Phase 3 training split, {len(tr)} admissions",
           "uncertainty": f"{DRAWS} draws per arm from the patient-clustered sandwich covariance (fixed seed)",
           "queries": queries}
    (ASSETS / "cf_clinical.json").write_text(json.dumps(out), encoding="utf-8")
    return out


def export_figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = pd.read_csv(DOC / "CF3_PAIR_VALIDATION.csv")
    v = v[(v["set"] == "primary") & v["group"].isin(["ST", "T"])]          # = figure__cf_pair_validation.sql
    agg = v.groupby(["direction", "method"]).mae_sd.mean()
    comp = {d: {"mae_sd": agg[d].to_dict(), "n_pairs": int(v[v.direction == d].n.max())} for d in ("add", "remove")}
    methods = [("A2_shrunk", "Track A, A2 (used by the tools)"), ("A1_additive", "Track A, A1 (pre-specified)"),
               ("population_mean", "Population mean (explain_ecg)"), ("no_change", "No change (copy source ECG)")]
    ink, ink2, accent, grid = "#191919", "#666666", "#257de0", "#e6e6e6"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (direction, title) in zip(axes, (("add", "Add MI: before ECG → during-MI ECG"),
                                             ("remove", "Remove MI: during-MI ECG → before ECG"))):
        vals = [comp[direction]["mae_sd"][k] for k, _ in methods]
        ys = np.arange(len(methods))[::-1]
        for y, v, (k, _) in zip(ys, vals, methods):
            c = accent if k == "A2_shrunk" else ink2
            ax.scatter([v], [y], s=64, color=c, zorder=3, edgecolor="white", linewidth=2)
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=9, color=ink)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.set_xlim(0.5, 1.12)
        ax.set_yticks(ys, [lbl for _, lbl in methods], fontsize=9, color=ink)
        ax.tick_params(axis="x", labelsize=8, colors=ink2)
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.supxlabel("Absolute error in control SD, averaged over the 14 ST/T measures; lower is better", fontsize=9,
                  color=ink2)
    n = comp["add"]["n_pairs"]
    fig.suptitle(f"Individual counterfactuals checked against real within-patient pairs ({n:,} held-out pairs)",
                 fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ASSETS / "figures" / "cf_pair_validation.png", dpi=110, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    a = export_track_a()
    c = export_clinical()
    export_figure()
    print("exported cf_track_a.json (", len(a["measures"]), "measures ), cf_clinical.json (",
          list(c["queries"]), "), cf_pair_validation.png")
