"""Python port of roadmap/phase9_demo/explainer.js (same parameters, same arithmetic, same order of operations).

Parity with the browser explainer is tested to 1e-9 on the Phase 9 held-out cases. It never produces a
probability, a diagnosis or advice.
"""
import math

CAVEAT = ("Research prototype: not a diagnosis, not medical advice. The expected ECG is a model-implied "
          "comparison for a similar patient without MI, not this patient's true counterfactual. Effects hold only "
          "under the study's stated assumptions (hospitalized adults, ECG recorded after MI onset).")


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def bspline_all(x: float, t: list[float], k: int) -> list[float]:
    """Cox-de Boor B-spline basis with constant extrapolation (clamp to the boundary knots)."""
    n = len(t) - k - 1
    lo, hi = t[k], t[n]
    xc = min(max(x, lo), hi)
    if xc >= hi:
        xc = hi - 1e-12 * max(1, abs(hi))          # right end: use the left limit
    b = [1.0 if t[i] <= xc < t[i + 1] else 0.0 for i in range(len(t) - 1)]
    for d in range(1, k + 1):
        nxt = []
        for i in range(len(t) - 1 - d):
            a, c = t[i + d] - t[i], t[i + d + 1] - t[i + 1]
            nxt.append(((xc - t[i]) / a * b[i] if a > 0 else 0) + ((t[i + d + 1] - xc) / c * b[i + 1] if c > 0 else 0))
        b = nxt
    return b[:n]


def design_row(patient: dict, s: dict) -> list[float]:
    full = bspline_all(float(patient["age"]), s["age_spline"]["knots"], s["age_spline"]["degree"])
    age = [full[j] for j in s["age_spline"]["kept_columns"]]
    b = [float(patient[c]) for c in s["covariates"]]
    row = age + b
    for i in range(len(b)):
        for a in age:
            row.append(a * b[i])
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            row.append(b[i] * b[j])
    if len(row) != len(s["scaler_mean"]):
        raise ValueError("design length mismatch")
    return [(v - s["scaler_mean"][i]) / s["scaler_scale"][i] for i, v in enumerate(row)]


def linear(intercept: list[float], coef: list[list[float]], x: list[float], measures: list[str]) -> dict[str, float]:
    out = {}
    for j, m in enumerate(measures):
        v = intercept[j]
        for i in range(len(x)):
            v += coef[j][i] * x[i]
        out[m] = v
    return out


def expected_without_mi(patient: dict, p: dict) -> dict[str, float]:
    s = p["structural"]
    return linear(s["intercept"], s["coef"], design_row(patient, s), s["measures"])


def swap_lead_labels(features: dict) -> dict:
    out = {}
    for k, v in features.items():
        nk = k.replace("_aVL", "_aVF", 1) if "_aVL" in k else (k.replace("_aVF", "_aVL", 1) if "_aVF" in k else k)
        out[nk] = v
    return out


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def validate(patient: dict, features: dict, p: dict) -> tuple[list[str], list[str]]:
    errors, outside, pre = [], [], p["preprocessing"]
    amin, amax = pre["age_limits"]
    age = _num(patient.get("age"))
    if age is None:
        errors.append("Age is missing or not a number.")
    elif age < amin or age > amax:
        errors.append(f"Age {patient['age']} is outside {amin}-{amax}.")
    for c in p["structural"]["covariates"]:
        if _num(patient.get(c)) not in (0.0, 1.0):
            errors.append(f"{c} must be 0 or 1.")
    for f in pre["point_inputs"] + [pre["count_input"]]:
        raw = features.get(f)
        if raw is None or raw == "":
            errors.append(f"Missing ECG field: {f}.")
            continue
        v = _num(raw)
        if v is None:
            errors.append(f"ECG field {f} is not a number ({raw}).")
            continue
        r = pre["training_range_p001_p999"].get(f)
        if r:
            w = r[1] - r[0]
            if v < r[0] - w or v > r[1] + w:
                errors.append(f"ECG field {f} = {v} is far outside the training range ({r[0]:.3f} to {r[1]:.3f}).")
            elif v < r[0] or v > r[1]:
                outside.append(f)
    if (_num(features.get("RR_Mean_Global")) or 0) <= 0:
        errors.append("RR_Mean_Global must be positive.")
    if (_num(features.get(pre["count_input"])) or 0) <= 0:
        errors.append(f"{pre['count_input']} must be positive.")
    return errors, outside


def explain(patient: dict, features_in: dict, p: dict, timing: str = "unknown", raw_ecgdeli: bool = False) -> dict:
    """Same structure as explainer.js explain(): caveat, flags, rejected, errors, rows, groups."""
    features = swap_lead_labels(features_in) if raw_ecgdeli else dict(features_in)
    errors, outside = validate(patient, features, p)
    result = {"caveat": CAVEAT, "flags": [], "rejected": bool(errors), "errors": errors, "rows": [], "groups": []}
    if timing != "after_onset":
        result["flags"].append("Unknown ECG timing: effects apply to ECGs recorded after MI onset.")
    if errors:
        return result
    pre = p["preprocessing"]
    f = {k: float(features[k]) for k in pre["point_inputs"]}
    count = float(features[pre["count_input"]])
    ratio = count / (10000 / f["RR_Mean_Global"])
    if count < pre["beat_count_min"] or ratio < pre["beat_ratio_limits"][0] or ratio > pre["beat_ratio_limits"][1]:
        result["flags"].append(f"Low or inconsistent beat count ({count:g} beats detected; {ratio * 100:.0f}% of expected).")
    if outside:
        result["flags"].append(f"Inputs outside the range seen in training: {', '.join(outside)}.")
    comp = pre["composites"]
    obs_raw = dict(f)
    lat = comp["ST_lateral_mean"]["mean_of"]
    obs_raw["ST_lateral_mean"] = sum(f[k] for k in lat) / len(lat)
    obs_raw["ST_max_depression"] = min(f[k] for k in comp["ST_max_depression"]["min_of"])
    expected = expected_without_mi(patient, p)
    for m in p["structural"]["measures"]:
        e = p["effects"][m]
        lo, hi = pre["winsor_bounds"][m]
        observed = min(max(obs_raw[m], lo), hi)
        diff = observed - expected[m]
        result["rows"].append({"measure": m, "group": e["group"], "observed": observed,
                               "expected_without_mi": expected[m], "difference": diff,
                               "difference_sd": diff / e["control_sd"], "mi_effect": e["att"], "mi_effect_ci": e["att_ci"],
                               "mi_effect_sd": e["att_sd"], "in_mi_effect_direction": _sign(diff) == _sign(e["att"])})
    for g in ("ST", "T", "Q"):
        rs = [r for r in result["rows"] if r["group"] == g]
        result["groups"].append({"group": g, "measures": len(rs),
                                 "in_mi_direction": sum(r["in_mi_effect_direction"] for r in rs),
                                 "mean_aligned_difference_sd": sum(_sign(r["mi_effect"]) * r["difference_sd"] for r in rs) / len(rs)})
    return result


def profile(patient: dict, p: dict, model: dict) -> list[dict]:
    """Expected value with and without MI for a profile, with bootstrap 95% intervals for the model-implied effect."""
    s = p["structural"]
    x = design_row(patient, s)
    m0 = linear(s["intercept"], s["coef"], x, s["measures"])
    m1 = linear(model["m1_intercept"], model["m1_coef"], x, model["measures"])
    rows = []
    for j, m in enumerate(model["measures"]):
        draws = sorted(d[j][0] + sum(c * xi for c, xi in zip(d[j][1:], x)) for d in model["effect_draws"])
        n = len(draws)
        lo, hi = draws[int(round(0.025 * (n - 1)))], draws[int(round(0.975 * (n - 1)))]
        e = p["effects"][m]
        rows.append({"measure": m, "group": e["group"], "expected_without_mi": m0[m], "expected_with_mi": m1[m],
                     "effect": m1[m] - m0[m], "effect_ci": [lo, hi], "effect_sd": (m1[m] - m0[m]) / e["control_sd"],
                     "study_att_sd": e["att_sd"], "study_att_ci_sd": [e["att_ci"][0] / e["control_sd"],
                                                                      e["att_ci"][1] / e["control_sd"]]})
    return rows


def synthetic_features(patient: dict, p: dict, shift: bool) -> dict:
    """A model-generated ECG (training medians, with the eligible leads set to the expected no-MI value, plus the
    study's average MI effect when shift=True). Never a real patient."""
    pre = p["preprocessing"]
    feats = dict(pre["training_medians"])
    expected = expected_without_mi(patient, p)
    for m in p["structural"]["measures"]:
        if m in pre["point_inputs"]:
            feats[m] = expected[m] + (p["effects"][m]["att"] if shift else 0.0)
    return feats
