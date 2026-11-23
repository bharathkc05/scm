"""Computation tools: confounding_what_if (bias arithmetic on stored estimates), cohort_summary (disclosure-
controlled aggregates from patient-level data) and explain_ecg (model-implied ECG comparison)."""
import math
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from local_mcp.envelope import Envelope, fail
from local_mcp.runtime import ServerContext, audited
from local_mcp.tools import ecg_explainer as X
from local_mcp.tools.common import (EXPOSURE_LABELS, L1_ASSUMPTIONS, L1_CAVEATS, L2_ASSUMPTIONS, L2_CAVEATS, READ_ONLY,
                                    envelope, evalue_rr, evalue_rr_ci, evalue_sd, r4, require, resolve_variable)
from local_mcp.tools.robustness import EValue

DEMO = "roadmap/phase9_demo/demo_params.json"
PROFILE = "local_mcp/assets/profile_model.json"


# ====================================================================== confounding_what_if
class Benchmark(BaseModel):
    factor: str
    rr_with_mi: float


class FrontierPoint(BaseModel):
    rr_confounder_exposure: float
    min_rr_confounder_outcome_point: float | None = Field(description="None = impossible at this exposure RR")
    min_rr_confounder_outcome_ci: float | None


class WhatIfResult(BaseModel):
    result_id: str
    label: str
    scale: str
    observed: float
    observed_ci95: list[float]
    rr_confounder_exposure: float
    rr_confounder_outcome: float
    bounding_factor: float
    worst_case: float = Field(description="Most the confounder could move the estimate toward the null, on the original scale")
    worst_case_ci_limit: float = Field(description="Same for the confidence limit nearest the null")
    explained_away_point: bool
    explained_away_ci: bool
    e_value: EValue
    benchmarks: list[Benchmark]
    frontier: list[FrontierPoint]
    method: str


def _bound(rr_ue: float, rr_ud: float) -> float:
    return rr_ue * rr_ud / (rr_ue + rr_ud - 1)


def _needed(target: float, rr_ue: float) -> float | None:
    """Smallest confounder-outcome RR that, with rr_ue, gives bounding factor >= target."""
    if target <= 1:
        return 1.0
    if rr_ue <= target:
        return None
    return target * (rr_ue - 1) / (rr_ue - target)


def confounding_what_if(ctx: ServerContext, result_id: str, rr_ue: float, rr_ud: float) -> Envelope:
    require(ctx, "l1_effects", "l2_effects")
    rid = result_id.strip()
    b = _bound(rr_ue, rr_ud)
    if rid.upper().startswith("L1:"):
        rid = ":".join([rid[:2].upper(), *[x.lower() for x in rid.split(":")[1:]]])
        rows = ctx.sandbox.query("risk__effect", [rid, "ate_overlap"])
        if not rows:
            fail("INVALID_ARGUMENT", f"Unknown Layer 1 result {result_id!r}.",
                 valid_values=[r["result_id"] for r in ctx.sandbox.query("risk__labels")])
        p = rows[0]
        rr, lo, hi = p["rr"], p["rr_ci_low"], p["rr_ci_high"]
        up = rr >= 1
        obs = rr if up else 1 / rr
        near = (lo if up else 1 / hi) if not (lo <= 1 <= hi) else 1.0
        worst, worst_ci = obs / b, near / b
        to_scale = (lambda v: v) if up else (lambda v: 1 / v)
        label, scale = p["label"], "risk ratio"
        observed, ci = rr, [lo, hi]
        ev = EValue(point=r4(evalue_rr(rr)), ci_limit=r4(evalue_rr_ci(lo, hi)))
        assumptions, caveats, views = L1_ASSUMPTIONS, L1_CAVEATS, ("layer1_estimates",)
    elif rid.upper().startswith("L2:"):
        var = resolve_variable(ctx, rid[3:], ("ecg_feature", "ecg_composite"))
        rows = ctx.sandbox.query("ecg__measure", ["primary", var["name"]])
        if not rows or rows[0]["family"] not in ("point", "st_composite"):
            fail("NOT_SUPPORTED", "Bias analysis covers Layer 1 results and the Layer 2 point features and composites.")
        p = rows[0]
        d, lo, hi = p["att_sd"], p["att_sd_ci_low"], p["att_sd_ci_high"]
        sgn = 1 if d >= 0 else -1
        obs = math.exp(0.91 * abs(d))
        near_d = lo if d > 0 else hi
        near = math.exp(0.91 * abs(near_d)) if (near_d > 0) == (d > 0) and near_d != 0 else 1.0
        worst, worst_ci = obs / b, near / b
        to_scale = lambda v: sgn * math.log(v) / 0.91          # noqa: E731  back to control-SD units
        rid, label, scale = f"L2:{var['name']}", var["label"], "ATT, control SD (via RR = exp(0.91|d|))"
        observed, ci = d, [lo, hi]
        pt, cl = evalue_sd(d, lo, hi)
        ev = EValue(point=r4(pt), ci_limit=r4(cl))
        assumptions, caveats, views = L2_ASSUMPTIONS, L2_CAVEATS, ("layer2_estimates", "evalues")
    else:
        fail("INVALID_ARGUMENT", f"result_id must start with 'L1:' or 'L2:', got {result_id!r}.")
    bench = []
    for exp in ("diabetes", "hypertension", "ckd", "prior_mi"):
        r = ctx.sandbox.query("risk__effect", [f"L1:{exp}:primary", "ate_overlap"])
        if r:
            bench.append(Benchmark(factor=f"{EXPOSURE_LABELS[exp]} (measured, adjusted)", rr_with_mi=r4(r[0]["rr"])))
    frontier = [FrontierPoint(rr_confounder_exposure=a, min_rr_confounder_outcome_point=r4(_needed(obs, a)),
                              min_rr_confounder_outcome_ci=r4(_needed(near, a)))
                for a in (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)]
    res = WhatIfResult(
        result_id=rid, label=label, scale=scale, observed=r4(observed), observed_ci95=[r4(ci[0]), r4(ci[1])],
        rr_confounder_exposure=rr_ue, rr_confounder_outcome=rr_ud, bounding_factor=r4(b),
        worst_case=r4(to_scale(worst)), worst_case_ci_limit=r4(to_scale(worst_ci)),
        explained_away_point=worst <= 1 + 1e-12, explained_away_ci=worst_ci <= 1 + 1e-12, e_value=ev,
        benchmarks=bench, frontier=frontier,
        method="Ding & VanderWeele bounding factor B = RRue*RRud/(RRue+RRud-1); worst case = observed RR / B.")
    verdict = ("would explain away the estimate itself" if res.explained_away_point else
               "would move the confidence interval to the null" if res.explained_away_ci else
               "would not explain the result away")
    if rid.startswith("L1"):
        nums = f"worst-case RR {res.worst_case:.2f} (CI limit {res.worst_case_ci_limit:.2f}) vs observed {res.observed:.2f}"
    else:
        nums = (f"worst case {res.worst_case:+.3f} SD (CI limit {res.worst_case_ci_limit:+.3f}) vs observed "
                f"{res.observed:+.3f} SD")
    summary = (f"A hidden confounder with RR {rr_ue:g} with the exposure and {rr_ud:g} with the outcome {verdict}: "
               f"{nums}; E-value {ev.point:.2f} (CI {ev.ci_limit:.2f}).")
    return envelope(ctx, "confounding_what_if", res, summary,
                    caveats=list(caveats) + ["Bounds address unmeasured confounding only, not selection or measurement error."],
                    assumptions=assumptions, views=views, phase=7)


# ====================================================================== cohort_summary
MIN_CELL = 11
ROUND_TO = 5
DIMS = {
    "acute_mi": "acute_mi",
    "sex": "CASE WHEN sex_male = 1 THEN 'male' ELSE 'female' END",
    "age_band": "CASE WHEN age < 45 THEN '18-44' WHEN age < 65 THEN '45-64' WHEN age < 80 THEN '65-79' ELSE '80+' END",
    "diabetes": "diabetes_primary", "hypertension": "hypertension_primary", "ckd": "ckd_primary",
    "prior_mi": "prior_mi_primary",
    "mi_subtype": ("CASE WHEN mi_stemi = 1 THEN 'STEMI' WHEN mi_nstemi = 1 THEN 'NSTEMI' "
                   "WHEN mi_type2 = 1 THEN 'type 2' ELSE 'other' END"),
    "timing_class": "timing_class",
}
METRICS = {
    "age_mean": "avg(age)", "age_sd": "stddev_samp(age)", "pct_male": "100 * avg(sex_male)",
    "pct_diabetes": "100 * avg(diabetes_primary)", "pct_hypertension": "100 * avg(hypertension_primary)",
    "pct_ckd": "100 * avg(ckd_primary)", "pct_prior_mi": "100 * avg(prior_mi_primary)", "pct_acute_mi": "100 * avg(acute_mi)",
}
LAYERS = {
    "layer1": ("layer1", "TRUE", "Layer 1: all adult admissions with a diagnosis code"),
    "layer2_candidates": ("layer2", "TRUE", "Layer 2 candidates: admissions with an ECG with features"),
    "layer2_primary": ("layer2", "set_primary = 1", "Layer 2 primary set: after-onset MI vs controls"),
}
Dim = Literal["acute_mi", "sex", "age_band", "diabetes", "hypertension", "ckd", "prior_mi", "mi_subtype", "timing_class"]
Metric = Literal["age_mean", "age_sd", "pct_male", "pct_diabetes", "pct_hypertension", "pct_ckd", "pct_prior_mi",
                 "pct_acute_mi"]


class Cell(BaseModel):
    strata: dict[str, str | int | None]
    suppressed: bool
    n_admissions: int | None
    n_patients: int | None
    metrics: dict[str, float | None]


class CohortResult(BaseModel):
    layer: str
    layer_description: str
    stratify_by: list[str]
    metrics: list[str]
    rows: list[Cell]
    suppressed_cells: int
    min_cell: int
    rounding: str


def suppress(cells: list[dict], dims: list[str]) -> set[int]:
    """Primary suppression (n < MIN_CELL) plus complementary suppression: within every line of the table
    (cells sharing all other dimensions; the whole table when there is one dimension) a single suppressed cell
    could be recovered from the margin, so the smallest other cell in that line is suppressed too."""
    hidden = {i for i, c in enumerate(cells) if c["n_admissions"] < MIN_CELL}
    if not dims:
        return hidden
    changed = True
    while changed:
        changed = False
        for d in dims:
            others = [o for o in dims if o != d]
            lines: dict[tuple, list[int]] = {}
            for i, c in enumerate(cells):
                lines.setdefault(tuple(c[o] for o in others), []).append(i)
            for members in lines.values():
                if len(members) > 1 and sum(i in hidden for i in members) == 1:
                    rest = sorted((i for i in members if i not in hidden), key=lambda i: cells[i]["n_admissions"])
                    hidden.add(rest[0])
                    changed = True
    return hidden


def _round(n: int) -> int:
    return int(math.floor(n / ROUND_TO + 0.5) * ROUND_TO)


def cohort_summary(ctx: ServerContext, layer: str, stratify_by: list[str], metrics: list[str]) -> Envelope:
    source, where, desc = LAYERS[layer]
    if ctx.sandbox is None or not ctx.sandbox.has(source):
        fail("DATA_NOT_PRESENT", "Patient-level analysis files are not on this machine; aggregate results tools "
                                 "still work.", needs=[source])
    if len(set(stratify_by)) != len(stratify_by) or len(stratify_by) > 2:
        fail("INVALID_ARGUMENT", "stratify_by takes at most 2 distinct dimensions.")
    if "timing_class" in stratify_by and source != "layer2":
        fail("INVALID_ARGUMENT", "timing_class exists only for the Layer 2 layers.")
    if "mi_subtype" in stratify_by:
        where = f"({where}) AND acute_mi = 1"
    fragments = {
        "dims": "".join(f"{DIMS[d]} AS {d}, " for d in stratify_by),       # whitelisted SQL only
        "metrics": "".join(f", {METRICS[m]} AS {m}" for m in metrics),
        "source": source, "where": where,
        "order": ("ORDER BY " + ", ".join(stratify_by)) if stratify_by else "",
    }
    raw = ctx.sandbox.query("cohort__grouped", fragments=fragments)
    hidden = suppress(raw, list(stratify_by))
    rows = []
    for i, c in enumerate(raw):
        hide = i in hidden
        rows.append(Cell(strata={d: c[d] for d in stratify_by}, suppressed=hide,
                         n_admissions=None if hide else _round(c["n_admissions"]),
                         n_patients=None if hide else _round(c["n_patients"]),
                         metrics={m: (None if hide or c[m] is None else round(float(c[m]), 1)) for m in metrics}))
    res = CohortResult(layer=layer, layer_description=desc, stratify_by=list(stratify_by), metrics=list(metrics),
                       rows=rows, suppressed_cells=len(hidden), min_cell=MIN_CELL,
                       rounding=f"counts rounded to the nearest {ROUND_TO}")
    summary = (f"{len(rows)} cell(s) from {desc}"
               + (f" by {' x '.join(stratify_by)}" if stratify_by else "")
               + f"; counts rounded to the nearest {ROUND_TO}; {len(hidden)} cell(s) suppressed "
                 f"(fewer than {MIN_CELL} admissions, or needed to protect one).")
    env = envelope(ctx, "cohort_summary", res, summary,
                   caveats=["Descriptive statistics only; no patient-level rows are returned.",
                            "Suppressed cells protect small groups; do not try to reconstruct them from other queries."],
                   views=(source,), phase=3, tier="aggregate_from_patient_level")
    env.disclosure.suppressed_cells = len(hidden)
    return env


# ====================================================================== explain_ecg
class Patient(BaseModel):
    age: float = Field(ge=18, le=100, description="Age in years")
    sex: Literal["male", "female"]
    diabetes: bool = False
    hypertension: bool = False
    ckd: bool = False
    prior_mi: bool = False

    def covariates(self) -> dict:
        return {"age": self.age, "sex_male": int(self.sex == "male"), "diabetes_primary": int(self.diabetes),
                "hypertension_primary": int(self.hypertension), "ckd_primary": int(self.ckd),
                "prior_mi_primary": int(self.prior_mi)}


class ExplainRow(BaseModel):
    measure: str
    label: str
    group: str
    units: str = "mV"
    observed: float | None = None
    expected_without_mi: float
    expected_with_mi: float | None = None
    difference_sd: float | None = Field(None, description="(observed - expected without MI) / control SD")
    model_implied_mi_effect: float | None = None
    model_effect_ci95: list[float] | None = None
    in_mi_effect_direction: bool | None = None
    study_mi_effect_sd: float
    study_mi_effect_ci95_sd: list[float]


class GroupRow(BaseModel):
    group: str
    measures: int
    in_mi_direction: int
    mean_aligned_difference_sd: float


class ExplainResult(BaseModel):
    mode: Literal["profile", "comparison"]
    synthetic: bool
    rows: list[ExplainRow]
    groups: list[GroupRow] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    not_computed: str = "No probability of MI, diagnosis or advice is computed by this tool."


def explain_ecg(ctx: ServerContext, patient: Patient, ecg_features: dict[str, float] | None,
                example_id: str | None, timing: str, raw_ecgdeli_labels: bool) -> Envelope:
    p = ctx.registry.json(DEMO)
    names = {e["name"]: e for e in ctx.sandbox.query("variables__names")} if ctx.sandbox else {}
    cov = patient.covariates()
    if ecg_features is not None and example_id is not None:
        fail("INVALID_ARGUMENT", "Give ecg_features or example_id, not both.")
    caveats = [X.CAVEAT, "Only the 16 ECG measures that passed every stress test are used."]
    if ecg_features is None and example_id is None:
        model = ctx.registry.json(PROFILE)
        rows = [ExplainRow(measure=r["measure"], label=names.get(r["measure"], {}).get("label", r["measure"]),
                           group=r["group"], expected_without_mi=r4(r["expected_without_mi"]),
                           expected_with_mi=r4(r["expected_with_mi"]), model_implied_mi_effect=r4(r["effect"]),
                           model_effect_ci95=[r4(v) for v in r["effect_ci"]], difference_sd=r4(r["effect_sd"]),
                           study_mi_effect_sd=r4(r["study_att_sd"]), study_mi_effect_ci95_sd=[r4(v) for v in r["study_att_ci_sd"]])
                for r in X.profile(cov, p, model)]
        res = ExplainResult(mode="profile", synthetic=False, rows=rows)
        lat = next(r for r in rows if r.measure == "ST_lateral_mean")
        summary = (f"For this profile the model expects lateral ST {lat.expected_without_mi:+.4f} mV without MI and "
                   f"{lat.expected_with_mi:+.4f} mV with MI (model-implied effect {lat.difference_sd:+.2f} SD, 95% CI "
                   f"{lat.model_effect_ci95[0]:+.4f} to {lat.model_effect_ci95[1]:+.4f} mV); study average "
                   f"{lat.study_mi_effect_sd:+.2f} SD.")
        caveats.append("Model-implied conditional expectations from training-split models; not observed counterfactuals.")
        return envelope(ctx, "explain_ecg", res, summary, caveats=caveats, assumptions=L2_ASSUMPTIONS,
                        paths=(DEMO, PROFILE), phase=9, tier="model_output")
    synthetic = example_id is not None
    feats = X.synthetic_features(cov, p, shift=example_id == "synthetic_typical_mi_shift") if synthetic else dict(ecg_features)
    out = X.explain(cov, feats, p, timing=timing, raw_ecgdeli=raw_ecgdeli_labels and not synthetic)
    if out["rejected"]:
        fail("INVALID_ARGUMENT", "ECG input rejected.", errors=out["errors"])
    rows = [ExplainRow(measure=r["measure"], label=names.get(r["measure"], {}).get("label", r["measure"]),
                       group=r["group"], observed=r4(r["observed"]), expected_without_mi=r4(r["expected_without_mi"]),
                       difference_sd=r4(r["difference_sd"]), in_mi_effect_direction=r["in_mi_effect_direction"],
                       study_mi_effect_sd=r4(r["mi_effect_sd"]),
                       study_mi_effect_ci95_sd=[r4(v / p["effects"][r["measure"]]["control_sd"]) for v in r["mi_effect_ci"]])
            for r in out["rows"]]
    groups = [GroupRow(group=g["group"], measures=g["measures"], in_mi_direction=g["in_mi_direction"],
                       mean_aligned_difference_sd=r4(g["mean_aligned_difference_sd"])) for g in out["groups"]]
    res = ExplainResult(mode="comparison", synthetic=synthetic, rows=rows, groups=groups, flags=out["flags"])
    summary = ("Compared with the model's expected ECG for this patient without MI: "
               + "; ".join(f"{g.group} {g.in_mi_direction}/{g.measures} measures in the MI-effect direction "
                           f"(mean aligned difference {g.mean_aligned_difference_sd:+.2f} SD)" for g in groups)
               + ". This is a comparison, not a diagnosis.")
    if synthetic:
        caveats.append("Synthetic ECG generated from the model (training medians plus model values), not a real patient.")
    return envelope(ctx, "explain_ecg", res, summary, caveats=caveats + out["flags"], assumptions=L2_ASSUMPTIONS,
                    paths=(DEMO,), phase=9, tier="model_output")


# ====================================================================== registration
def register(app: FastMCP, ctx: ServerContext) -> None:
    @app.tool(name="confounding_what_if", title="Hidden-confounder what-if",
              description=("How strong would an unmeasured confounder (e.g. smoking) have to be to explain a result "
                           "away? Enter its risk ratio with the exposure and with the outcome; returns the worst-case "
                           "estimate (Ding & VanderWeele bound), whether it is explained away, the E-value, measured "
                           "risk factors as benchmarks, and the frontier of confounder strengths that would suffice."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "confounding_what_if")
    def _whatif(
        result_id: Annotated[str, Field(description="e.g. 'L1:hypertension:primary' or 'L2:ST_lateral_mean'")],
        rr_confounder_exposure: Annotated[float, Field(ge=1.0, le=10.0, description="Confounder-exposure risk ratio")],
        rr_confounder_outcome: Annotated[float, Field(ge=1.0, le=10.0, description="Confounder-outcome risk ratio")],
    ) -> Envelope[WhatIfResult]:
        return confounding_what_if(ctx, result_id, rr_confounder_exposure, rr_confounder_outcome)

    @app.tool(name="cohort_summary", title="Cohort summary (patient-level data, aggregates only)",
              description=("Descriptive counts, means and percentages from the patient-level analysis data on this "
                           "machine, stratified by up to 2 fixed dimensions. Aggregates only: counts rounded to the "
                           "nearest 5 and cells under 11 admissions suppressed (with complementary suppression)."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "cohort_summary")
    def _cohort(
        layer: Annotated[Literal["layer1", "layer2_candidates", "layer2_primary"], Field(description="Which cohort")] = "layer1",
        stratify_by: Annotated[list[Dim], Field(max_length=2, description="0-2 dimensions")] = [],
        metrics: Annotated[list[Metric], Field(max_length=8, description="Metrics besides the counts")] = ["age_mean", "pct_male"],
    ) -> Envelope[CohortResult]:
        return cohort_summary(ctx, layer, stratify_by, metrics)

    @app.tool(name="explain_ecg", title="Explain an ECG against the no-MI expectation",
              description=("Compares one ECG with what the study's model expects for the same patient without MI, "
                           "beside the study's average MI effect, for the 16 robust ECG measures. Without an ECG it "
                           "returns the patient's expected values with and without MI (profile mode, with intervals). "
                           "example_id gives a synthetic, model-generated ECG for demos. Never returns a probability "
                           "or diagnosis."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "explain_ecg", redact=("patient", "ecg_features"))
    def _explain(
        patient: Annotated[Patient, Field(description="Baseline facts for the patient")],
        ecg_features: Annotated[dict[str, float] | None,
                                Field(description="The 53 ECGdeli point features + QRS_Dur_Global_count, by name")] = None,
        example_id: Annotated[Literal["synthetic_no_mi", "synthetic_typical_mi_shift"] | None,
                              Field(description="Synthetic model-generated ECG instead of ecg_features")] = None,
        timing: Annotated[Literal["after_onset", "unknown"], Field(description="Was the ECG recorded after MI onset?")] = "unknown",
        raw_ecgdeli_labels: Annotated[bool, Field(description="Features come straight from the MATLAB pipeline (aVL/aVF swapped)")] = False,
    ) -> Envelope[ExplainResult]:
        return explain_ecg(ctx, patient, ecg_features, example_id, timing, raw_ecgdeli_labels)
