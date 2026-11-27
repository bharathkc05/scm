"""Phase 10 individual-reasoning tools.

counterfactual_ecg         Track A (CF3, model A2): this ECG's 16 robust measures with MI removed or added, with a
                           90% interval and the error the same model made on real held-out within-patient pairs.
clinical_what_if           Profile-specific risk ratio of do(condition = yes) vs do(condition = no) for acute-MI
                           admission, and the probability of necessity for an MI patient with the condition.
                           Absolute probabilities of MI are never returned (server rule).
counterfactual_validation  How accurate the individual counterfactuals are (aggregate pair-validation metrics).

Inputs are bounded numbers and enums; no identifiers, file paths or waveforms are accepted. Real patients' ECG
counterfactuals are produced only by the local patient explorer (roadmap/phase10_counterfactual_ecg), never here.
"""
import ast
import math
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from local_mcp.envelope import Envelope, fail
from local_mcp.runtime import ServerContext, audited
from local_mcp.tools import cf_math as M
from local_mcp.tools.common import (EXTRA_CAVEATS, IDENTIFICATION, L1_ASSUMPTIONS, L1_CAVEATS, L2_ASSUMPTIONS,
                                    READ_ONLY, envelope, r4, require)
from local_mcp.tools.computation import Patient
from local_mcp.tools.ecg_explainer import swap_lead_labels

TRACK_A = "local_mcp/assets/cf_track_a.json"
CLINICAL = "local_mcp/assets/cf_clinical.json"
SUMMARY_JSON = "roadmap/phase10_counterfactual_ecg/cf3_01_track_a.json"
DEMO = "roadmap/phase9_demo/demo_params.json"
MIN_CELL = 11
MiType = Literal["nstemi", "stemi_anterior", "stemi_inferior", "stemi_other", "type2", "other"]
TYPE_LABEL = {"nstemi": "NSTEMI", "stemi_anterior": "anterior STEMI", "stemi_inferior": "inferior STEMI",
              "stemi_other": "other-territory STEMI", "type2": "type 2 MI", "other": "other/unspecified MI"}
CF_CAVEATS = [
    "Research prototype: not a diagnosis, not medical advice. A counterfactual is a model-based range for this "
    "patient, not their true alternative ECG.",
    "Checked against real within-patient pairs (an ECG before and during the same patient's MI): about 0.9 SD of "
    "error per ST/T measure remains when adding MI and about 0.7 SD when removing it. Each row gives that error.",
    "A different visit is an imperfect stand-in for 'the same patient without MI': ageing, drugs and disease between "
    "visits are part of the validation error.",
]
STEMI_WARNING = ("STEMI: the 90% intervals were too narrow on held-out STEMI pairs (anterior STEMI covered 77%, "
                 "not 90%); treat them as optimistic.")
CF_ASSUMPTIONS = {
    "CF-A2": "The patient's own deviation from the expected ECG does not change with MI; the share that persisted "
             "between real visits (rho, fitted on validation pairs) is carried into the counterfactual.",
    "CF-A3": "The MI effect depends only on age, sex, comorbidities and MI type, not on unmeasured traits "
             "(not identifiable; bounded by the pair-validation error).",
    "CF-A4": "Another visit's ECG is an acceptable stand-in for the patient without MI.",
}


def _assumption_ids():
    from local_mcp.tools import common
    common.ASSUMPTIONS.update(CF_ASSUMPTIONS)       # server-authored text, added once at import
    return L2_ASSUMPTIONS + tuple(CF_ASSUMPTIONS)


CF_ASSUMPTION_IDS = _assumption_ids()


# ====================================================================== counterfactual_ecg
class CFRow(BaseModel):
    measure: str
    group: str
    units: str = "mV"
    observed: float = Field(description="Input value after the study's winsorization")
    counterfactual: float
    interval90: list[float] = Field(description="90% split-conformal interval for the counterfactual value")
    expected_in_target_state: float = Field(description="Population expectation for this profile in the target "
                                                        "state (what explain_ecg-style comparisons use)")
    change: float
    change_sd: float = Field(description="(counterfactual - observed) / control SD")
    validation_error_sd: float | None = Field(description="Mean absolute error of this model on held-out pairs (SD)")
    validation_error_population_mean_sd: float | None
    validation_error_no_change_sd: float | None
    validation_coverage90: float | None


class CFGroup(BaseModel):
    group: str
    measures: int
    mean_abs_change_sd: float


class CFResult(BaseModel):
    direction: Literal["remove_mi", "add_mi"]
    mi_type: str
    baseline_calibration: str
    synthetic: bool
    rows: list[CFRow]
    groups: list[CFGroup]
    validation_st_t: dict = Field(description="Composite ST/T error of this model and both baselines on the held-out "
                                              "pairs used for calibration")
    flags: list[str] = Field(default_factory=list)
    not_computed: str = "No probability of MI, diagnosis or advice is computed by this tool."


def _features(ctx: ServerContext, p: dict, ecg_features: dict, raw_ecgdeli: bool) -> tuple[dict, list[str]]:
    feats = swap_lead_labels(ecg_features) if raw_ecgdeli else dict(ecg_features)
    demo = ctx.registry.json(DEMO)["preprocessing"]
    known = set(p["inputs"]["features"]) | set(demo["point_inputs"]) | {demo["count_input"]}
    unknown = sorted(k for k in feats if k not in known)
    if unknown:
        fail("INVALID_ARGUMENT", "Unknown ECG feature name(s).", unknown=unknown[:10],
             valid_values=p["inputs"]["features"])
    missing = [f for f in p["inputs"]["features"] if f not in feats]
    if missing:
        fail("INVALID_ARGUMENT", "Missing ECG feature(s) needed for the 16 measures.", missing=missing)
    out, errors = {}, []
    for f in p["inputs"]["features"]:
        v = feats[f]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            errors.append(f"{f} is not a finite number.")
            continue
        if f.startswith("ST_Elev") and abs(v) > p["inputs"]["st_max_abs_mv"]:
            errors.append(f"{f} = {v} mV is beyond {p['inputs']['st_max_abs_mv']} mV (a delineation artefact).")
        out[f] = float(v)
    if errors:
        fail("INVALID_ARGUMENT", "ECG input rejected.", errors=errors)
    bounds = p["inputs"]["winsor_bounds"]
    comp = M.composites(out, p)
    raw = dict(out)
    lat = p["inputs"]["composites"]
    raw["ST_lateral_mean"] = (sum(out[k] for k in lat["ST_lateral_mean"]["mean_of"])
                              / len(lat["ST_lateral_mean"]["mean_of"]))
    raw["ST_max_depression"] = min(out[k] for k in lat["ST_max_depression"]["min_of"])
    clipped = [m for m in p["measures"] if not bounds[m][0] <= raw[m] <= bounds[m][1]]
    return comp, clipped


def _validation(ctx: ServerContext, pair_set: str, direction: str) -> dict[str, dict]:
    if not ctx.sandbox or not ctx.sandbox.has("cf_pair_validation"):
        return {}
    rows = ctx.sandbox.query("cf__by_measure", [pair_set, direction])
    out: dict[str, dict] = {}
    for r in rows:
        out.setdefault(r["measure"], {})[r["method"]] = r
    return out


def counterfactual_ecg(ctx: ServerContext, patient: Patient, direction: str, mi_type: str,
                       ecg_features: dict | None, example_id: str | None, baseline: str,
                       raw_ecgdeli_labels: bool) -> Envelope:
    p = ctx.registry.json(TRACK_A)
    if (ecg_features is None) == (example_id is None):
        fail("INVALID_ARGUMENT", "Give exactly one of ecg_features or example_id.")
    d = "remove" if direction == "remove_mi" else "add"
    side = "before" if baseline == "before_mi" else "after"
    cov = patient.covariates()
    flags = []
    if example_id is not None:
        m0, m1 = M.track_a_means(cov, mi_type, p)
        obs = m1 if d == "remove" else m0
        synthetic = True
    else:
        comp, clipped = _features(ctx, p, ecg_features, raw_ecgdeli_labels)
        obs = [comp[m] for m in p["measures"]]
        synthetic = False
        if clipped:
            flags.append(f"Outside the study's range and clipped to it: {', '.join(clipped)}.")
    cf = M.counterfactual(obs, cov, d, mi_type, side, p)
    sd = p["control_sd"]
    val = _validation(ctx, "primary" if side == "before" else "all_after", d)
    rows = []
    for j, m in enumerate(p["measures"]):
        v = val.get(m, {})
        g = "ST" if m.startswith("ST") else ("T" if m.startswith("T_") else "Q")
        rows.append(CFRow(
            measure=m, group=g, observed=r4(obs[j]), counterfactual=r4(cf["counterfactual"][j]),
            interval90=[r4(cf["low"][j]), r4(cf["high"][j])], expected_in_target_state=r4(cf["expected_target"][j]),
            change=r4(cf["counterfactual"][j] - obs[j]), change_sd=r4((cf["counterfactual"][j] - obs[j]) / sd[j]),
            validation_error_sd=r4(v.get("A2_shrunk", {}).get("mae_sd")),
            validation_error_population_mean_sd=r4(v.get("population_mean", {}).get("mae_sd")),
            validation_error_no_change_sd=r4(v.get("no_change", {}).get("mae_sd")),
            validation_coverage90=r4(v.get("A2_shrunk", {}).get("coverage_90"))))
    groups = []
    for g in ("ST", "T", "Q"):
        rs = [r for r in rows if r.group == g]
        groups.append(CFGroup(group=g, measures=len(rs), mean_abs_change_sd=r4(sum(abs(r.change_sd) for r in rs) / len(rs))))
    comp_val = ctx.registry.json(SUMMARY_JSON)["composite_st_t"]["primary" if side == "before" else "all_after"][d]
    vsum = {"pairs": comp_val["n_pairs"], "model_error_sd": comp_val["mae_sd"]["A2_shrunk"],
            "population_mean_error_sd": comp_val["mae_sd"]["population_mean"],
            "no_change_error_sd": comp_val["mae_sd"]["no_change"],
            "coverage90": comp_val["coverage_90_A2_shrunk"]["mean"]}
    if mi_type.startswith("stemi"):
        flags.append(STEMI_WARNING)
    if d == "remove":
        flags.append("Assumes the ECG was recorded during the MI, after onset.")
    res = CFResult(direction=direction, mi_type=mi_type, baseline_calibration=baseline, synthetic=synthetic,
                   rows=rows, groups=groups, validation_st_t={k: r4(v) if isinstance(v, float) else v
                                                              for k, v in vsum.items()}, flags=flags)
    lat = next(r for r in rows if r.measure == "ST_lateral_mean")
    verb = f"without the {TYPE_LABEL[mi_type]}" if d == "remove" else f"with a {TYPE_LABEL[mi_type]}"
    summary = (f"{'Synthetic profile ECG' if synthetic else 'This ECG'} {verb}: lateral ST {lat.observed:+.4f} -> "
               f"{lat.counterfactual:+.4f} mV (90% interval {lat.interval90[0]:+.4f} to {lat.interval90[1]:+.4f}); "
               + "; ".join(f"{g.group} measures move {g.mean_abs_change_sd:.2f} SD on average" for g in groups)
               + f". On {vsum['pairs']:,} held-out real pairs this model's ST/T error was {vsum['model_error_sd']:.2f} SD "
                 f"(population mean {vsum['population_mean_error_sd']:.2f}, no change {vsum['no_change_error_sd']:.2f}); "
                 f"90% intervals covered {vsum['coverage90']:.0%}.")
    caveats = CF_CAVEATS + flags
    if synthetic:
        caveats.append("Synthetic: the model's expected ECG for this profile, not a real patient. Its counterfactual "
                       "equals the expected ECG in the target state.")
    return envelope(ctx, "counterfactual_ecg", res, summary, caveats=caveats, assumptions=CF_ASSUMPTION_IDS,
                    views=("cf_pair_validation",), paths=(TRACK_A, SUMMARY_JSON), phase=10, tier="model_output")


# ====================================================================== clinical_what_if
class ClinicalEstimate(BaseModel):
    adjustment_set: str
    role: Literal["primary", "alternative"]
    risk_ratio: float = Field(description="do(condition = yes) vs do(condition = no), this profile")
    risk_ratio_ci95: list[float]
    probability_of_necessity: float | None = Field(
        description="For an MI patient with the condition: P(no MI had they not had it). None if not applicable")
    probability_of_necessity_ci95: list[float] | None
    identification: str


class ClinicalResult(BaseModel):
    exposure: str
    patient_has_exposure: bool
    patient_had_mi: bool
    estimates: list[ClinicalEstimate]
    study_average_rr: float | None = Field(description="Phase 5 average effect (ATE in the overlap region)")
    outcome: str = "Admission being an acute-MI admission, among hospitalized adults (Phase 5 Layer 1)."
    not_returned: str = ("Absolute probabilities of MI are not returned: the server never gives an individual a "
                         "probability of MI. Only ratios and the probability of necessity are reported.")


def clinical_what_if(ctx: ServerContext, patient: Patient, exposure: str, had_mi: bool) -> Envelope:
    c = ctx.registry.json(CLINICAL)
    cov = patient.covariates()
    exp_col = f"{exposure}_primary"
    has = bool(cov[exp_col])
    keys = ["diabetes:primary", "diabetes:alternative"] if exposure == "diabetes" else [f"{exposure}:primary"]
    estimates = []
    for k in keys:
        q = c["queries"][k]
        r = M.risk_ratio(cov, q)
        pn = pn_ci = None
        if has and had_mi:
            pn = r4(M.probability_of_necessity(r["rr"]))
            pn_ci = [r4(M.probability_of_necessity(r["rr_ci95"][0])), r4(M.probability_of_necessity(r["rr_ci95"][1]))]
        role = "primary" if k.endswith("primary") else "alternative"
        estimates.append(ClinicalEstimate(
            adjustment_set=q["label"], role=role, risk_ratio=r4(r["rr"]), risk_ratio_ci95=[r4(v) for v in r["rr_ci95"]],
            probability_of_necessity=pn, probability_of_necessity_ci95=pn_ci,
            identification=IDENTIFICATION[f"L1:{exposure}:{role}"]))
    study = None
    if ctx.sandbox and ctx.sandbox.has("l1_effects"):
        rows = ctx.sandbox.query("risk__effect", [f"L1:{exposure}:primary", "ate_overlap"])
        study = r4(rows[0]["rr"]) if rows else None
    res = ClinicalResult(exposure=exposure, patient_has_exposure=has, patient_had_mi=had_mi, estimates=estimates,
                         study_average_rr=study)
    e0 = estimates[0]
    summary = (f"For this profile, do({exposure} = yes) vs do({exposure} = no): risk ratio {e0.risk_ratio:.2f} "
               f"(95% CI {e0.risk_ratio_ci95[0]:.2f} to {e0.risk_ratio_ci95[1]:.2f}; study average "
               f"{f'{study:.2f}' if study is not None else 'n/a'})")
    if len(estimates) > 1:
        e1 = estimates[1]
        summary += (f"; alternative adjustment set {e1.risk_ratio:.2f} ({e1.risk_ratio_ci95[0]:.2f} to "
                    f"{e1.risk_ratio_ci95[1]:.2f}): read the two as a bracket")
    if e0.probability_of_necessity is not None:
        summary += (f". Probability that {exposure} was necessary for this MI: {e0.probability_of_necessity:.2f} "
                    f"({e0.probability_of_necessity_ci95[0]:.2f} to {e0.probability_of_necessity_ci95[1]:.2f})")
    elif had_mi and not has:
        summary += f". The patient does not have {exposure}, so the probability of necessity does not apply"
    summary += ". No absolute probability of MI is given."
    caveats = list(L1_CAVEATS) + [
        "Profile-specific ratio from the Phase 5 outcome models, refitted on the training split; intervals reflect "
        "model uncertainty only.",
        "Probability of necessity assumes the condition never prevents MI (monotonicity) and no unmeasured "
        "confounding; without monotonicity it is a lower bound. It cannot be validated for an individual.",
        "Phase 7: unmeasured confounding (e.g. smoking, lipids) could move these ratios; see confounding_what_if."]
    if f"L1:{exposure}" in EXTRA_CAVEATS:
        caveats.append(EXTRA_CAVEATS[f"L1:{exposure}"])
    return envelope(ctx, "clinical_what_if", res, summary, caveats=caveats, assumptions=L1_ASSUMPTIONS,
                    views=("layer1_estimates",), paths=(CLINICAL,), phase=10, tier="model_output")


# ====================================================================== counterfactual_validation
class ValidationRow(BaseModel):
    measure: str | None = None
    subgroup: str | None = None
    level: str | None = None
    direction: str
    method: str | None = None
    n: int | None
    suppressed: bool = False
    values: dict[str, float | list[float] | dict[str, float] | None]


class ValidationResult(BaseModel):
    view: str
    pair_set: str
    gate_cf_g1: dict
    composite_st_t: dict
    rows: list[ValidationRow]
    min_cell: int = MIN_CELL


def counterfactual_validation(ctx: ServerContext, view: str, direction: str | None, pair_set: str) -> Envelope:
    require(ctx, "cf_pair_validation", "cf_subgroups")
    if view == "subgroups" and pair_set != "primary":
        fail("INVALID_ARGUMENT", "Subgroup results exist for the primary pair set only.", valid_values=["primary"])
    s = ctx.registry.json(SUMMARY_JSON)
    dirs = ["add", "remove"] if direction is None else ["add" if direction == "add_mi" else "remove"]
    rows: list[ValidationRow] = []
    if view == "by_measure":
        keep = ("mae_sd", "median_ae_sd", "direction_acc", "coverage_90", "interval_halfwidth_sd",
                "diff_vs_population_mean", "diff_vs_population_mean_ci_low", "diff_vs_population_mean_ci_high",
                "diff_vs_no_change", "diff_vs_no_change_ci_low", "diff_vs_no_change_ci_high")
        base_keep = ("mae_sd", "direction_acc")         # baselines: their error only (payload cap)
        for d in dirs:                                  # one row per measure; methods nested (egress list cap)
            by: dict[str, list[dict]] = {}
            for r in ctx.sandbox.query("cf__by_measure", [pair_set, d]):
                by.setdefault(r["measure"], []).append(r)
            for measure, rs in by.items():
                n = min(r["n"] or 0 for r in rs)
                small = n < MIN_CELL
                rows.append(ValidationRow(measure=measure, direction=d, n=n, suppressed=small, values={} if small else {
                    r["method"]: {k: r4(r[k]) for k in (keep if r["method"].startswith("A") else base_keep)
                                  if k in r and r[k] is not None} for r in rs}))
    elif view == "subgroups":
        for r in ctx.sandbox.query("cf__subgroups"):
            if r["direction"] not in dirs:
                continue
            small = (r["n_pairs"] or 0) < MIN_CELL
            vals = {}
            for k, v in r.items():
                if k in ("subgroup", "level", "direction", "n_pairs"):
                    continue
                if isinstance(v, str) and v.startswith("["):
                    v = [r4(x) for x in ast.literal_eval(v)]
                elif isinstance(v, (int, float)):
                    v = r4(v)
                vals[k] = v
            rows.append(ValidationRow(subgroup=r["subgroup"], level=str(r["level"]), direction=r["direction"],
                                      n=r["n_pairs"], suppressed=small, values={} if small else vals))
    comp = {d: s["composite_st_t"][pair_set][d] for d in dirs}
    res = ValidationResult(view=view, pair_set=pair_set, gate_cf_g1=s["gate_cf_g1"], composite_st_t=comp, rows=rows)
    parts = []
    for d in dirs:
        c = comp[d]["mae_sd"]
        parts.append(f"{'adding' if d == 'add' else 'removing'} MI: model A2 error {c['A2_shrunk']:.2f} SD vs "
                     f"population mean {c['population_mean']:.2f} and no change {c['no_change']:.2f}, 90% coverage "
                     f"{comp[d]['coverage_90_A2_shrunk']['mean']:.0%}")
    summary = (f"Individual counterfactuals checked on {comp[dirs[0]]['n_pairs']:,} held-out within-patient pairs "
               f"({pair_set}): " + "; ".join(parts) + ". Gate CF-G1 passed for A2, failed for the pre-specified A1.")
    return envelope(ctx, "counterfactual_validation", res, summary,
                    caveats=["Aggregate validation on held-out patients; groups under 11 pairs are suppressed.",
                             "Errors are mean absolute errors in control-SD units of each measure.",
                             "STEMI intervals were too narrow (anterior STEMI 77% coverage)."],
                    assumptions=CF_ASSUMPTION_IDS, views=("cf_pair_validation", "cf_subgroups"),
                    paths=(SUMMARY_JSON,), phase=10, tier="aggregate")


# ====================================================================== registration
def register(app: FastMCP, ctx: ServerContext) -> None:
    @app.tool(name="counterfactual_ecg", title="Individual ECG counterfactual (MI removed or added)",
              description=("Feature-level individual counterfactual for one ECG: the 16 robust ECG measures with the "
                           "MI removed (ECG recorded during an MI) or added (ECG without MI, choose the MI type), "
                           "keeping the share of the patient's own deviation that persists between visits. Returns "
                           "each value with a 90% interval and the error this model made on real held-out "
                           "before/during-MI pairs. Takes per-lead ECGdeli features or a synthetic profile ECG; no "
                           "identifiers or waveforms. Never a diagnosis."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "counterfactual_ecg", redact=("patient", "ecg_features"))
    def _cf(
        patient: Annotated[Patient, Field(description="Baseline facts for the patient")],
        direction: Annotated[Literal["remove_mi", "add_mi"], Field(description="remove_mi: ECG recorded during an "
                                                                               "MI; add_mi: ECG without MI")],
        mi_type: Annotated[MiType, Field(description="Type of the MI being removed, or of the MI to add")],
        ecg_features: Annotated[dict[str, float] | None,
                                Field(description="ST_Elev for all 12 leads, T_Amp_II/aVR/V4/V5/V6, Q_Amp_I/aVL (mV); "
                                                  "other ECGdeli point features are accepted and ignored")] = None,
        example_id: Annotated[Literal["synthetic_expected"] | None,
                              Field(description="Use the model's expected ECG for this profile instead")] = None,
        baseline: Annotated[Literal["before_mi", "after_mi"],
                            Field(description="Which real pairs calibrate the interval: baselines recorded before "
                                              "the MI (primary) or after it")] = "before_mi",
        raw_ecgdeli_labels: Annotated[bool, Field(description="Features come straight from the legacy MATLAB "
                                                              "pipeline (aVL/aVF swapped)")] = False,
    ) -> Envelope[CFResult]:
        return counterfactual_ecg(ctx, patient, direction, mi_type, ecg_features, example_id, baseline,
                                  raw_ecgdeli_labels)

    @app.tool(name="clinical_what_if", title="Clinical what-if for one profile (risk ratio and probability of necessity)",
              description=("For one patient profile: the risk ratio for an acute-MI admission under do(condition = "
                           "yes) vs do(condition = no) for diabetes, hypertension or CKD, with a 95% interval, beside "
                           "the study average; and, for an MI patient who has the condition, the probability that it "
                           "was necessary for the MI. Never returns an absolute probability of MI."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "clinical_what_if", redact=("patient",))
    def _clin(
        patient: Annotated[Patient, Field(description="Baseline facts for the patient")],
        exposure: Annotated[Literal["diabetes", "hypertension", "ckd"], Field(description="Condition to intervene on")],
        had_mi: Annotated[bool, Field(description="The patient had an acute MI (needed for the probability of "
                                                  "necessity)")] = False,
    ) -> Envelope[ClinicalResult]:
        return clinical_what_if(ctx, patient, exposure, had_mi)

    @app.tool(name="counterfactual_validation", title="Accuracy of individual counterfactuals",
              description=("How accurate are individual ECG counterfactuals? Aggregate results from real held-out "
                           "within-patient pairs (ECG before vs during the same patient's MI): the model's error "
                           "against two baselines, interval coverage, per measure or by MI type and time gap."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "counterfactual_validation")
    def _val(
        view: Annotated[Literal["summary", "by_measure", "subgroups"], Field(description="Level of detail")] = "summary",
        direction: Annotated[Literal["add_mi", "remove_mi"] | None, Field(description="One direction or both")] = None,
        pair_set: Annotated[Literal["primary", "all_before", "all_after"],
                            Field(description="primary: after-onset MI with a clean before-MI baseline")] = "primary",
    ) -> Envelope[ValidationResult]:
        return counterfactual_validation(ctx, view, direction, pair_set)
