"""Robustness cards: one result against every stress test, judged by pre-declared rules.

Rules (fixed in Phases 6-7, applied identically to every result):
  * a check passes if its estimate is on the same side of the null as the primary estimate
    (Layer 1: RR vs 1; Layer 2: ATT vs 0); confidence intervals are reported but not required, because
    sensitivity subsets are smaller than the primary cohort;
  * Layer 2 negative control passes if |negative control| / |primary| < 0.5;
  * Layer 1 strict prior-only history flags are shown but not counted (known artifact, Phase 7);
  * Layer 2 also requires the primary confidence interval to exclude 0.
"""
from pydantic import BaseModel, Field

from local_mcp.envelope import fail
from local_mcp.runtime import ServerContext
from local_mcp.tools.common import (MEASUREMENT_NOTES, evalue_rr, evalue_rr_ci, evalue_sd, r4, require,
                                    resolve_variable)

L1_CHECKS = {
    ("outcome_codes", "narrow_principal"): "MI coded as the principal diagnosis",
    ("outcome_codes", "broad_any_410_I21_I22"): "Broad MI codes (including 410.x2 follow-ups)",
    ("outcome_codes", "type1_only"): "Type 2 MI excluded",
    ("subgroup", "age_lt65"): "Age < 65",
    ("subgroup", "age_ge65"): "Age >= 65",
    ("subgroup", "female"): "Female",
    ("subgroup", "male"): "Male",
    ("history_definition", "lookback_only"): "Only admissions with an earlier admission",
    ("history_definition", "lookback_covariate"): "Number of earlier admissions adjusted",
    ("history_definition", "strict_flags"): "Strict prior-only history flags",
}
NOT_COUNTED = {("history_definition", "strict_flags"):
               "Shown, not counted: strict flags are 0 on every first admission, where MI is more common (Phase 7)."}
L2_SETS = ["nstemi", "chest_pain_comparator", "after_strict", "no_type2", "first_ecg_has_features",
           "selection_weighted", "bmi_adjusted", "broad_codes"]
L2_SET_LABELS = {
    "nstemi": "NSTEMI only (no ST-elevation coding)", "chest_pain_comparator": "Chest-pain comparator",
    "after_strict": "After-strict timing", "no_type2": "Type 2 MI excluded",
    "first_ecg_has_features": "First ECG of the admission", "selection_weighted": "ECG-selection weighted",
    "bmi_adjusted": "BMI adjusted (BMI-available admissions)", "broad_codes": "Broad MI codes",
}
SUBGROUP_LABELS = {"age_lt65": "Age < 65", "age_ge65": "Age >= 65", "female": "Female", "male": "Male"}


class EValue(BaseModel):
    point: float
    ci_limit: float = Field(description="E-value for the confidence limit nearest the null")


class Check(BaseModel):
    check: str
    group: str
    estimate: float
    ci95: list[float]
    same_direction: bool
    ci_excludes_null: bool
    counted_in_verdict: bool
    passed: bool
    rule: str
    ratio_to_primary: float | None = None
    note: str | None = None


class Primary(BaseModel):
    scale: str
    estimate: float
    ci95: list[float]


class Card(BaseModel):
    result_id: str
    layer: int
    label: str
    primary: Primary
    checks: list[Check]
    e_value: EValue
    verdict: str = Field(description="survives | survives (descriptive) | fragile | not robust | no primary effect | "
                                     "not stress-tested; may add a measurement note")
    failed_checks: list[str]
    what_would_overturn_it: str
    measurement_note: str | None = None


# ------------------------------------------------------------------ Layer 1
def l1_card(ctx: ServerContext, result_id: str) -> Card:
    require(ctx, "l1_effects", "l1_checks")
    prim = ctx.sandbox.query("risk__effect", [result_id, "ate_overlap"])
    if not prim:
        fail("INVALID_ARGUMENT", f"Unknown Layer 1 result {result_id!r}.",
             valid_values=[r["result_id"] for r in ctx.sandbox.query("risk__labels")])
    p = prim[0]
    up = p["rr"] > 1
    checks: list[Check] = []
    for c in ctx.sandbox.query("risk__checks", [result_id]):
        key = (c["check_group"], c["check_name"])
        counted = key not in NOT_COUNTED
        same = (c["rr"] > 1) == up
        checks.append(Check(check=L1_CHECKS.get(key, c["check_name"]), group=c["check_group"], estimate=r4(c["rr"]),
                            ci95=[r4(c["rr_ci_low"]), r4(c["rr_ci_high"])], same_direction=same,
                            ci_excludes_null=not (c["rr_ci_low"] <= 1 <= c["rr_ci_high"]), counted_in_verdict=counted,
                            passed=same, rule="risk ratio on the same side of 1 as the primary estimate",
                            note=NOT_COUNTED.get(key)))
    labels = {r["label"]: r["result_id"] for r in ctx.sandbox.query("risk__labels")}
    for b in ctx.registry.json("roadmap/phase7_stress_test/p7_03_bmi.json")["layer1"]:
        if labels.get(b["label"]) == result_id:
            same = (b["rr_with_bmi"] > 1) == up
            checks.append(Check(check="BMI adjusted (BMI-available admissions)", group="bmi", estimate=r4(b["rr_with_bmi"]),
                                ci95=[r4(x) for x in b["ci_with"]], same_direction=same,
                                ci_excludes_null=not (b["ci_with"][0] <= 1 <= b["ci_with"][1]), counted_in_verdict=True,
                                passed=same, rule="risk ratio on the same side of 1 as the primary estimate",
                                note=f"Same admissions without BMI: {b['rr_without_bmi']:.2f}"))
    failed = [c.check for c in checks if c.counted_in_verdict and not c.passed]
    ev = EValue(point=r4(evalue_rr(p["rr"])), ci_limit=r4(evalue_rr_ci(p["rr_ci_low"], p["rr_ci_high"])))
    if not checks:
        verdict = "not stress-tested"
    elif failed:
        verdict = "fragile"
    else:
        verdict = "survives (descriptive)" if p["exposure"] == "prior_mi" else "survives"
    exposure = p["exposure"].replace("_", " ")
    overturn = (f"Already fragile: fails {', '.join(failed)}." if failed else
                f"An unmeasured confounder associated with both {exposure} and MI at RR >= {ev.ci_limit:.2f}, beyond "
                f"the adjustment set, could move the confidence interval to 1.")
    return Card(result_id=result_id, layer=1, label=p["label"],
                primary=Primary(scale="risk ratio (ATE in overlap)", estimate=r4(p["rr"]),
                                ci95=[r4(p["rr_ci_low"]), r4(p["rr_ci_high"])]),
                checks=checks, e_value=ev, verdict=verdict, failed_checks=failed, what_would_overturn_it=overturn)


# ------------------------------------------------------------------ Layer 2
def l2_card(ctx: ServerContext, measure: str) -> Card:
    require(ctx, "l2_effects", "l2_subgroups")
    rows = {r["analysis_set"]: r for r in ctx.sandbox.query("ecg__all_sets", [measure])}
    if "primary" not in rows or rows["primary"]["family"] not in ("point", "st_composite"):
        fail("NOT_SUPPORTED", f"Stress tests were run for the 53 point ECG features and 5 ST composites, not {measure!r}.")
    p = rows["primary"]
    neg = p["att_sd"] < 0
    checks: list[Check] = []
    for s in L2_SETS:
        r = rows.get(s)
        if r is None:
            continue
        same = (r["att_sd"] < 0) == neg
        note = None
        if s == "bmi_adjusted" and "bmi_subset_unadjusted" in rows:
            note = f"Same admissions without BMI: {rows['bmi_subset_unadjusted']['att_sd']:+.3f} SD"
        checks.append(Check(check=L2_SET_LABELS[s], group="analysis_set", estimate=r4(r["att_sd"]),
                            ci95=[r4(r["att_sd_ci_low"]), r4(r["att_sd_ci_high"])], same_direction=same,
                            ci_excludes_null=r["att_sd_ci_low"] > 0 or r["att_sd_ci_high"] < 0, counted_in_verdict=True,
                            passed=same, rule="effect on the same side of 0 as the primary estimate", note=note))
    for g in ctx.sandbox.query("ecg__subgroups", [measure]):
        same = (g["att_sd"] < 0) == neg
        checks.append(Check(check=SUBGROUP_LABELS.get(g["subgroup"], g["subgroup"]), group="subgroup",
                            estimate=r4(g["att_sd"]), ci95=[r4(g["att_sd_ci_low"]), r4(g["att_sd_ci_high"])],
                            same_direction=same, ci_excludes_null=g["att_sd_ci_low"] > 0 or g["att_sd_ci_high"] < 0,
                            counted_in_verdict=True, passed=same,
                            rule="effect on the same side of 0 as the primary estimate"))
    nc = rows.get("negative_control")
    if nc is not None:
        ratio = abs(nc["att_sd"]) / abs(p["att_sd"]) if p["att_sd"] else float("inf")
        checks.append(Check(check="Negative control (ECG before a later MI)", group="negative_control",
                            estimate=r4(nc["att_sd"]), ci95=[r4(nc["att_sd_ci_low"]), r4(nc["att_sd_ci_high"])],
                            same_direction=(nc["att_sd"] < 0) == neg,
                            ci_excludes_null=nc["att_sd_ci_low"] > 0 or nc["att_sd_ci_high"] < 0,
                            counted_in_verdict=True, passed=ratio < 0.5, rule="|negative control| / |primary| < 0.5",
                            ratio_to_primary=r4(ratio)))
    stored = ctx.sandbox.query("ecg__evalue", [measure])
    if stored:
        ev = EValue(point=r4(stored[0]["evalue_point"]), ci_limit=r4(stored[0]["evalue_ci"]))
    else:
        pt, ci = evalue_sd(p["att_sd"], p["att_sd_ci_low"], p["att_sd_ci_high"])
        ev = EValue(point=r4(pt), ci_limit=r4(ci))
    primary_ok = p["att_sd_ci_low"] > 0 or p["att_sd_ci_high"] < 0
    failed = [c.check for c in checks if c.counted_in_verdict and not c.passed]
    note = MEASUREMENT_NOTES.get(measure)
    if not primary_ok:
        verdict, overturn = "no primary effect", "The primary confidence interval already includes 0."
    elif failed:
        verdict, overturn = "not robust", f"Already not robust: fails {', '.join(failed)}."
    else:
        verdict = "survives"
        overturn = (f"An unmeasured confounder associated with both MI and this ECG measure at RR >= {ev.ci_limit:.2f} "
                    f"(beyond age, sex and comorbidities) could move the confidence interval to 0.")
    if measure == "QRS_Dur_Global" and verdict == "survives":
        verdict = "survives (measurement not validated)"
    label = resolve_variable(ctx, measure, ("ecg_feature", "ecg_composite"))["label"]
    return Card(result_id=f"L2:{measure}", layer=2, label=label,
                primary=Primary(scale="ATT in control-SD units", estimate=r4(p["att_sd"]),
                                ci95=[r4(p["att_sd_ci_low"]), r4(p["att_sd_ci_high"])]),
                checks=checks, e_value=ev, verdict=verdict, failed_checks=failed, what_would_overturn_it=overturn,
                measurement_note=note)


def card(ctx: ServerContext, result_id: str) -> Card:
    rid = result_id.strip()
    if rid.upper().startswith("L1:"):
        parts = rid.split(":")
        rid = ":".join([parts[0].upper(), *[p.lower() for p in parts[1:]]])
        return l1_card(ctx, rid)
    if rid.upper().startswith("L2:"):
        measure = resolve_variable(ctx, rid[3:], ("ecg_feature", "ecg_composite", "latent", "validation"))["name"]
        return l2_card(ctx, measure)
    fail("INVALID_ARGUMENT", f"result_id must look like 'L1:<exposure>:<primary|alternative>' or 'L2:<measure>', "
                             f"got {result_id!r}.", hint="Call study_overview for the available result IDs.")
