"""Portable arithmetic for the Phase 10 individual-reasoning tools (no numpy, no sklearn, no patient data).

Parameters come from local_mcp/assets/cf_track_a.json and cf_clinical.json (exported by
roadmap/phase10_counterfactual_ecg/cf6_01_export.py). Parity with the fitted models is tested to 1e-9.
"""
import math

from local_mcp.tools.ecg_explainer import bspline_all


def design_row(cov: dict, d: dict) -> list[float]:
    """Same columns and order as roadmap.estimators.Design._raw, then standardized."""
    full = bspline_all(float(cov["age"]), d["age_spline"]["knots"], d["age_spline"]["degree"])
    age = [full[j] for j in d["age_spline"]["kept_columns"]]
    b = [float(cov[c]) for c in d["covariates"]]
    row = age + b
    if d["age_by_all"] and b:
        for bi in b:
            row += [a * bi for a in age]
    if d["interactions"] and b:
        if "sex_male" in d["covariates"] and not d["age_by_all"]:
            s = float(cov["sex_male"])
            row += [a * s for a in age]
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                row.append(b[i] * b[j])
    if len(row) != len(d["scaler_mean"]):
        raise ValueError("design length mismatch")
    return [(v - d["scaler_mean"][i]) / d["scaler_scale"][i] for i, v in enumerate(row)]


def _linear(intercept: list[float], coef: list[list[float]], x: list[float]) -> list[float]:
    return [intercept[j] + sum(c * xi for c, xi in zip(coef[j], x)) for j in range(len(intercept))]


# ------------------------------------------------------------------ Track A
def track_a_means(cov: dict, mi_type: str, p: dict) -> tuple[list[float], list[float]]:
    """(m0, m1): expected measures without MI and with an MI of `mi_type`, for this covariate profile."""
    x = design_row(cov, p["design"])
    m0 = _linear(p["m0"]["intercept"], p["m0"]["coef"], x)
    dummies = [1.0 if mi_type == t else 0.0 for t in p["m1"]["type_dummies"]]
    m1 = _linear(p["m1"]["intercept"], p["m1"]["coef"], x + dummies)
    return m0, m1


def composites(features: dict, p: dict) -> dict[str, float]:
    """The 16 measures from per-lead inputs (ST already checked against the artefact limit), winsorized."""
    comp = p["inputs"]["composites"]
    raw = dict(features)
    lat = comp["ST_lateral_mean"]["mean_of"]
    raw["ST_lateral_mean"] = sum(features[k] for k in lat) / len(lat)
    raw["ST_max_depression"] = min(features[k] for k in comp["ST_max_depression"]["min_of"])
    out = {}
    for m in p["measures"]:
        lo, hi = p["inputs"]["winsor_bounds"][m]
        out[m] = min(max(float(raw[m]), lo), hi)
    return out


def counterfactual(observed: list[float], cov: dict, direction: str, mi_type: str, baseline: str,
                   p: dict) -> dict:
    """Model A2: y_cf = m_target + rho * (y_obs - m_source); 90% split-conformal interval."""
    m0, m1 = track_a_means(cov, mi_type, p)
    src, tgt = (m1, m0) if direction == "remove" else (m0, m1)
    t = p["tuned"][baseline][direction]
    cf = [tg + r * (y - s) for y, s, tg, r in zip(observed, src, tgt, t["rho"])]
    return {"counterfactual": cf, "low": [c - q for c, q in zip(cf, t["q90"])],
            "high": [c + q for c, q in zip(cf, t["q90"])], "expected_target": tgt, "expected_source": src,
            "rho": t["rho"]}


# ------------------------------------------------------------------ clinical do()
def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def arm_risk(x: list[float], arm: dict) -> float:
    return _sigmoid(arm["intercept"] + sum(c * v for c, v in zip(arm["coef"], x)))


def risk_ratio(cov: dict, q: dict) -> dict:
    """Profile-specific RR of do(exposure=1) vs do(exposure=0), with a 95% interval from the coefficient draws.
    Absolute risks are computed but deliberately not returned by the tool."""
    x = design_row(cov, q["design"])
    r0, r1 = arm_risk(x, q["arms"]["0"]), arm_risk(x, q["arms"]["1"])
    rr_draws = []
    for d0, d1 in zip(q["arms"]["0"]["draws"], q["arms"]["1"]["draws"]):
        a0 = _sigmoid(d0[0] + sum(c * v for c, v in zip(d0[1:], x)))
        a1 = _sigmoid(d1[0] + sum(c * v for c, v in zip(d1[1:], x)))
        rr_draws.append(a1 / a0)
    rr_draws.sort()
    n = len(rr_draws)
    lo, hi = rr_draws[int(round(0.025 * (n - 1)))], rr_draws[int(round(0.975 * (n - 1)))]
    return {"rr": r1 / r0, "rr_ci95": [lo, hi], "_r0": r0, "_r1": r1}


def probability_of_necessity(rr: float) -> float:
    """PN = 1 - 1/RR under monotonicity and no unmeasured confounding (Tian & Pearl); 0 when RR <= 1."""
    return max(0.0, 1 - 1 / rr)
