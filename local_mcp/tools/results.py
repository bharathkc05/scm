"""Results tools over the pre-computed roadmap estimates (tier T2):
get_risk_factor_effect, get_ecg_effect, get_subgroup_effects, robustness_card.
"""
import math
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from scipy.stats import norm

from local_mcp.envelope import Envelope, fail
from local_mcp.runtime import ServerContext, audited
from local_mcp.tools.common import (ADJUST_LABELS, EXPOSURE_LABELS, EXTRA_CAVEATS, IDENTIFICATION, L1_ASSUMPTIONS,
                                    L1_CAVEATS, L2_ASSUMPTIONS, L2_CAVEATS, MEASUREMENT_NOTES, READ_ONLY, SETS,
                                    envelope, evalue_rr, evalue_rr_ci, r4, require, resolve_variable, sig)
from local_mcp.tools.robustness import Card, EValue, card, l1_card

L1_VIEWS = ("layer1_estimates", "outcome_codes_layer1", "subgroups_layer1", "layer1_history_variants", "evalues")
L2_VIEWS = ("layer2_estimates", "deconfounded_ecg_effects", "sensitivity_estimates", "selection_weighted_layer2",
            "bmi_layer2_with", "bmi_layer2_without", "outcome_codes_layer2_broad", "subgroups_layer2",
            "subgroup_interactions_layer2", "evalues")
ECG_CATEGORIES = ("ecg_feature", "ecg_composite", "latent", "validation")
AnalysisSet = Literal["primary", "nstemi", "chest_pain_comparator", "after_strict", "no_type2", "first_ecg_has_features",
                      "selection_weighted", "bmi_adjusted", "broad_codes", "negative_control"]


# ====================================================================== get_risk_factor_effect
class RiskFactorResult(BaseModel):
    result_id: str
    exposure: str
    adjustment: str
    estimand: str
    adjusted_for: list[str]
    n_admissions: int
    n_exposed: int
    risk_ratio: float
    rr_ci95: list[float]
    risk_difference_pp: float
    rd_ci95_pp: list[float]
    crude_rd_pp: float
    e_value: EValue
    identification: str
    phase7_verdict: str


def get_risk_factor_effect(ctx: ServerContext, exposure: str, adjustment: str, estimand: str) -> Envelope:
    require(ctx, "l1_effects")
    rid = f"L1:{exposure}:{adjustment}"
    rows = ctx.sandbox.query("risk__effect", [rid, estimand])
    if not rows:
        fail("INVALID_ARGUMENT", f"No {adjustment} adjustment set exists for {exposure}.",
             valid_values=[r["result_id"] for r in ctx.sandbox.query("risk__labels")])
    p = rows[0]
    verdict = l1_card(ctx, rid).verdict
    res = RiskFactorResult(
        result_id=rid, exposure=exposure, adjustment=adjustment, estimand=estimand,
        adjusted_for=[ADJUST_LABELS.get(t, t) for t in p["adjust_for"].split(",")],
        n_admissions=p["n"], n_exposed=p["exposed"], risk_ratio=r4(p["rr"]),
        rr_ci95=[r4(p["rr_ci_low"]), r4(p["rr_ci_high"])], risk_difference_pp=r4(p["rd_pp"]),
        rd_ci95_pp=[r4(p["rd_ci_low_pp"]), r4(p["rd_ci_high_pp"])], crude_rd_pp=r4(p["crude_rd_pp"]),
        e_value=EValue(point=r4(evalue_rr(p["rr"])), ci_limit=r4(evalue_rr_ci(p["rr_ci_low"], p["rr_ci_high"]))),
        identification=IDENTIFICATION[rid], phase7_verdict=verdict)
    word = {"ate_overlap": "in the overlap population", "ate": "in all admissions", "att": "among the exposed"}[estimand]
    summary = (f"{EXPOSURE_LABELS[exposure].capitalize()} is associated with a risk ratio of {res.risk_ratio:.2f} "
               f"(95% CI {res.rr_ci95[0]:.2f}-{res.rr_ci95[1]:.2f}) for acute MI at admission {word}, adjusted for "
               f"{', '.join(res.adjusted_for)}; Phase 7 verdict: {verdict}.")
    caveats = L1_CAVEATS + [c for k, c in EXTRA_CAVEATS.items() if rid.startswith(k)]
    return envelope(ctx, "get_risk_factor_effect", res, summary, caveats=caveats, assumptions=L1_ASSUMPTIONS,
                    views=("layer1_estimates",), phase=5)


# ====================================================================== get_ecg_effect
class EcgRow(BaseModel):
    result_id: str
    measure: str
    label: str
    family: str
    group: str
    analysis_set: str
    att_sd: float
    ci95_sd: list[float]
    att_raw: float | None
    ci95_raw: list[float | None]
    raw_units: str | None
    p_value: float | None
    q_value: float | None = Field(None, description="Benjamini-Hochberg q within the FDR family (primary family: 53 point features)")
    crude_sd: float | None
    classification: str | None = Field(None, description="Phase 6: MI-attributable | partly confounded | explained by "
                                                         "confounding | revealed by adjustment | no association")
    adjusted_share_of_crude: float | None = None
    robust: bool | None = None
    negative_control_ratio: float | None = None
    validated: bool | None = None
    note: str | None = None


class LatentRow(BaseModel):
    latent: str
    r2_baseline: float
    delta_r2_mi: float
    att_sd: float
    q_value: float | None
    top_handcrafted_feature: str
    top_feature_corr: float


class GroupSummary(BaseModel):
    measures: int
    fdr_significant: int | None
    ci_excludes_0: int
    mean_abs_att_sd: float


class LatentSummary(BaseModel):
    latents: int
    fdr_significant: int
    median_r2_baseline: float
    median_delta_r2_mi: float
    max_delta_r2_mi: float
    mi_exceeds_baseline: int
    conclusion: str


class EcgEffectResult(BaseModel):
    analysis_set: str
    analysis_set_description: str
    units: str
    rows: list[EcgRow] = Field(default_factory=list)
    group_summary: GroupSummary | None = None
    latents: list[LatentRow] = Field(default_factory=list)
    latent_summary: LatentSummary | None = None


LATENT_CONCLUSION = ("MI explains a tiny share of each latent's variance (less than baseline factors in 63 of 64), and "
                     "baseline-residualized latents carry no MI information beyond the handcrafted features "
                     "(out-of-fold AUC 0.659 vs 0.653). Do not use the latents as an MI-specific node.")


def _ecg_row(r: dict, var: dict) -> EcgRow:
    m = r["measure"]
    q = r.get("att_q")
    return EcgRow(
        result_id=f"L2:{m}", measure=m, label=var.get("label") or m, family=r["family"], group=r["grp"],
        analysis_set=r["analysis_set"], att_sd=r4(r["att_sd"]), ci95_sd=[r4(r["att_sd_ci_low"]), r4(r["att_sd_ci_high"])],
        att_raw=r4(r["att"]), ci95_raw=[r4(r["att_ci_low"]), r4(r["att_ci_high"])], raw_units=var.get("units") or None,
        p_value=sig(r.get("att_p")), q_value=sig(q), crude_sd=r4(r.get("crude_sd")),
        classification=r.get("class"), adjusted_share_of_crude=r4(r.get("adjusted_share_of_crude")),
        robust=(bool(r["robust_mi_sign"]) and m != "QRS_Dur_Global") if r.get("robust_mi_sign") is not None else None,
        negative_control_ratio=r4(r.get("negative_control_ratio")),
        validated=False if m == "QRS_Dur_Global" else (True if m in ("RR_Mean_Global", "machine_qrs_ms", "machine_rr_ms") else None),
        note=MEASUREMENT_NOTES.get(m))


def _latent_summary(ctx) -> LatentSummary:
    s = ctx.sandbox.query("ecg__latent_summary")[0]
    return LatentSummary(latents=s["latents"], fdr_significant=s["fdr_significant"],
                         median_r2_baseline=sig(s["median_r2_baseline"]), median_delta_r2_mi=sig(s["median_delta_r2_mi"]),
                         max_delta_r2_mi=sig(s["max_delta_r2_mi"]), mi_exceeds_baseline=s["mi_exceeds_baseline"],
                         conclusion=LATENT_CONCLUSION)


def _latent_rows(rows) -> list[LatentRow]:
    return [LatentRow(latent=r["latent"], r2_baseline=sig(r["r2_baseline"]), delta_r2_mi=sig(r["delta_r2_mi_adjusted"]),
                      att_sd=r4(r["att_sd"]), q_value=sig(r["att_q"]), top_handcrafted_feature=r["top_handcrafted_feature"],
                      top_feature_corr=r4(r["top_feature_corr"])) for r in rows]


def get_ecg_effect(ctx: ServerContext, measure: str | None, group: str | None, analysis_set: str, units: str,
                   top_k: int) -> Envelope:
    require(ctx, "l2_effects", "l2_class", "variables")
    if (measure is None) == (group is None):
        fail("INVALID_ARGUMENT", "Give exactly one of 'measure' or 'group'.")
    names = {e["name"]: e for e in ctx.sandbox.query("variables__names")}
    res = EcgEffectResult(analysis_set=analysis_set, analysis_set_description=SETS[analysis_set], units=units)
    caveats = list(L2_CAVEATS)
    views = ["layer2_estimates", "deconfounded_ecg_effects"] + ([] if analysis_set == "primary" else list(L2_VIEWS))

    if measure is not None:
        var = resolve_variable(ctx, measure, ECG_CATEGORIES)
        m = var["name"]
        if analysis_set != "primary" and var["family"] not in ("point", "st_composite"):
            fail("INVALID_ARGUMENT", f"{m} is only estimated in the primary analysis set; sensitivity sets cover the "
                                     "53 point features and 5 ST composites.")
        rows = ctx.sandbox.query("ecg__measure", [analysis_set, m])
        if not rows:
            fail("DATA_NOT_PRESENT", f"No estimate for {m} in analysis set {analysis_set}.")
        row = _ecg_row(rows[0], var)
        res.rows = [row]
        if var["family"] == "latent":
            res.latents = _latent_rows(ctx.sandbox.query("ecg__latent_one", [m]))
            res.latent_summary = _latent_summary(ctx)
            caveats.append("The beta-VAE encoder saw acute-MI ECGs during unsupervised training.")
            views.append("latent_audit")
        val, lo, hi = (row.att_sd, *row.ci95_sd) if units == "sd" else (row.att_raw, *row.ci95_raw)
        unit_word = "SD" if units == "sd" else (row.raw_units or "")
        status = ("robust MI sign" if row.robust else
                  "not robust" if row.robust is False else "not classified")
        summary = (f"{row.label}: acute MI shifts it by {val:+.3f} {unit_word} (95% CI {lo:+.3f} to {hi:+.3f}) among "
                   f"after-onset MI admissions ({SETS[analysis_set]})"
                   + (f"; classified '{row.classification}', {status}." if row.classification else "."))
        if row.note:
            caveats.append(row.note)
    else:
        if group == "latent":
            if analysis_set != "primary":
                fail("INVALID_ARGUMENT", "The latent audit exists only for the primary analysis set.")
            require(ctx, "latent_audit")
            res.latents = _latent_rows(ctx.sandbox.query("ecg__latents", [top_k]))
            res.latent_summary = _latent_summary(ctx)
            summary = (f"Latent audit: {res.latent_summary.fdr_significant}/{res.latent_summary.latents} latents differ "
                       f"significantly by MI, but MI explains a median {res.latent_summary.median_delta_r2_mi:.2g} of a "
                       f"latent's variance (max {res.latent_summary.max_delta_r2_mi:.2g}); not an MI-specific signal.")
            caveats.append("The beta-VAE encoder saw acute-MI ECGs during unsupervised training.")
            views.append("latent_audit")
        else:
            family, grp = ("st_composite", None) if group == "composite" else ("point", group)
            rows = ctx.sandbox.query("ecg__group", [analysis_set, family, grp, top_k])
            res.rows = [_ecg_row(r, names.get(r["measure"], {})) for r in rows]
            s = ctx.sandbox.query("ecg__group_summary", [analysis_set, family, grp])[0]
            res.group_summary = GroupSummary(measures=s["measures"], fdr_significant=s["fdr_significant"],
                                             ci_excludes_0=s["ci_excludes_0"], mean_abs_att_sd=r4(s["mean_abs_att_sd"]))
            top = res.rows[0] if res.rows else None
            sig_txt = (f"{s['fdr_significant']}/{s['measures']} FDR-significant" if analysis_set == "primary" and family == "point"
                       else f"{s['ci_excludes_0']}/{s['measures']} with CI excluding 0")
            summary = (f"{group} group ({SETS[analysis_set]}): {sig_txt}; mean |effect| {s['mean_abs_att_sd']:.3f} SD"
                       + (f"; largest {top.label} {top.att_sd:+.3f} SD." if top else "."))
            for r in res.rows:
                if r.note:
                    caveats.append(f"{r.measure}: {r.note}")
    return envelope(ctx, "get_ecg_effect", res, summary, caveats=caveats, assumptions=L2_ASSUMPTIONS,
                    views=views, phase=5 if analysis_set == "primary" else 6)


# ====================================================================== get_subgroup_effects
class SubgroupRow(BaseModel):
    subgroup: str
    label: str
    n: int | None
    cases: int | None
    estimate: float
    ci95: list[float]


class Interaction(BaseModel):
    contrast: str
    difference: float
    scale: str
    p_value: float
    q_value: float | None
    source: str


class SubgroupResult(BaseModel):
    result_id: str
    modifier: str
    subgroups: list[SubgroupRow]
    interaction: Interaction


PAIRS = {"age": ("age_lt65", "age_ge65"), "sex": ("female", "male")}
LABELS = {"age_lt65": "Age < 65", "age_ge65": "Age >= 65", "female": "Female", "male": "Male"}
MIN_CASES = 100


def _se(lo, hi, log=False):
    return ((math.log(hi) - math.log(lo)) if log else (hi - lo)) / (2 * 1.96)


def get_subgroup_effects(ctx: ServerContext, result_id: str, modifier: str) -> Envelope:
    a, b = PAIRS[modifier]
    rid = result_id.strip()
    if rid.upper().startswith("L1:"):
        c = l1_card(ctx, ":".join([rid[:2].upper(), *[x.lower() for x in rid.split(":")[1:]]]))
        rows = {r["check_name"]: r for r in ctx.sandbox.query("risk__checks", [c.result_id]) if r["check_group"] == "subgroup"}
        if not rows:
            fail("NOT_SUPPORTED", f"Subgroups were estimated for the primary Layer 1 sets, not {c.result_id}.")
        out = []
        for s in (a, b):
            r = rows[s]
            if min(r["outcome_events"], r["exposed"]) < MIN_CASES:
                fail("DISCLOSURE_BLOCKED", f"Subgroup {s} has fewer than {MIN_CASES} cases.")
            out.append(SubgroupRow(subgroup=s, label=LABELS[s], n=r["n"], cases=r["outcome_events"], estimate=r4(r["rr"]),
                                   ci95=[r4(r["rr_ci_low"]), r4(r["rr_ci_high"])]))
        ra, rb = rows[a], rows[b]
        diff = math.log(rb["rr"]) - math.log(ra["rr"])
        se = math.hypot(_se(ra["rr_ci_low"], ra["rr_ci_high"], True), _se(rb["rr_ci_low"], rb["rr_ci_high"], True))
        inter = Interaction(contrast=f"{LABELS[b]} vs {LABELS[a]}", difference=r4(math.exp(diff)),
                            scale="ratio of risk ratios", p_value=sig(2 * norm.sf(abs(diff / se))), q_value=None,
                            source="computed from subgroup confidence intervals (normal approximation)")
        res = SubgroupResult(result_id=c.result_id, modifier=modifier, subgroups=out, interaction=inter)
        summary = (f"{c.label}: RR {out[0].estimate:.2f} ({LABELS[a]}) vs {out[1].estimate:.2f} ({LABELS[b]}); "
                   f"ratio {inter.difference:.2f}, p = {inter.p_value:.2g}.")
        return envelope(ctx, "get_subgroup_effects", res, summary, caveats=L1_CAVEATS + ["Subgroup comparisons are "
                        "descriptive effect modification under the same assumptions."], assumptions=L1_ASSUMPTIONS,
                        views=("subgroups_layer1",), phase=7)
    if rid.upper().startswith("L2:"):
        require(ctx, "l2_subgroups")
        var = resolve_variable(ctx, rid[3:], ("ecg_feature", "ecg_composite"))
        m = var["name"]
        rows = {r["subgroup"]: r for r in ctx.sandbox.query("ecg__subgroups", [m])}
        if not rows:
            fail("NOT_SUPPORTED", f"Subgroups were estimated for point features and ST composites, not {m}.")
        counts = ctx.registry.json("roadmap/phase7_stress_test/p7_05_subgroups.json")["layer2_by_subgroup"]
        out = []
        for s in (a, b):
            if counts[s]["exposed"] < MIN_CASES:
                fail("DISCLOSURE_BLOCKED", f"Subgroup {s} has fewer than {MIN_CASES} MI cases.")
            r = rows[s]
            out.append(SubgroupRow(subgroup=s, label=LABELS[s], n=counts[s]["n"], cases=counts[s]["exposed"],
                                   estimate=r4(r["att_sd"]), ci95=[r4(r["att_sd_ci_low"]), r4(r["att_sd_ci_high"])]))
        stored = ctx.sandbox.query("ecg__interaction", [m, modifier])
        if stored:
            s0 = stored[0]
            inter = Interaction(contrast=f"{LABELS[b]} minus {LABELS[a]}", difference=r4(s0["difference"]),
                                scale="difference in ATT (control SD)", p_value=sig(s0["p"]), q_value=sig(s0["q"]),
                                source="Phase 7 (BH across the 16 robust measures)")
        else:
            ra, rb = rows[a], rows[b]
            diff = rb["att_sd"] - ra["att_sd"]
            se = math.hypot(_se(ra["att_sd_ci_low"], ra["att_sd_ci_high"]), _se(rb["att_sd_ci_low"], rb["att_sd_ci_high"]))
            inter = Interaction(contrast=f"{LABELS[b]} minus {LABELS[a]}", difference=r4(diff),
                                scale="difference in ATT (control SD)", p_value=sig(2 * norm.sf(abs(diff / se))),
                                q_value=None, source="computed from subgroup confidence intervals (not in the Phase 7 FDR family)")
        res = SubgroupResult(result_id=f"L2:{m}", modifier=modifier, subgroups=out, interaction=inter)
        summary = (f"{var['label']}: {out[0].estimate:+.3f} SD ({LABELS[a]}) vs {out[1].estimate:+.3f} SD ({LABELS[b]}); "
                   f"difference {inter.difference:+.3f} SD, p = {inter.p_value:.2g}"
                   + (f", q = {inter.q_value:.2g}." if inter.q_value is not None else "."))
        return envelope(ctx, "get_subgroup_effects", res, summary, caveats=L2_CAVEATS + ["Subgroup comparisons are "
                        "descriptive effect modification under the same assumptions."], assumptions=L2_ASSUMPTIONS,
                        views=("subgroups_layer2", "subgroup_interactions_layer2"),
                        paths=("roadmap/phase7_stress_test/p7_05_subgroups.json",), phase=7)
    fail("INVALID_ARGUMENT", f"result_id must start with 'L1:' or 'L2:', got {result_id!r}.")


# ====================================================================== robustness_card
def robustness_card(ctx: ServerContext, result_id: str) -> Envelope:
    c: Card = card(ctx, result_id)
    n_counted = sum(x.counted_in_verdict for x in c.checks)
    n_pass = sum(x.counted_in_verdict and x.passed for x in c.checks)
    scale = "RR" if c.layer == 1 else "SD"
    summary = (f"{c.label}: primary {c.primary.estimate:+.3f} {scale} " if c.layer == 2 else
               f"{c.label}: primary RR {c.primary.estimate:.2f} ") + \
              (f"(95% CI {c.primary.ci95[0]:.2f} to {c.primary.ci95[1]:.2f}); {n_pass}/{n_counted} stress tests passed; "
               f"E-value {c.e_value.point:.2f} (CI {c.e_value.ci_limit:.2f}); verdict: {c.verdict}.")
    if c.layer == 1:
        caveats = L1_CAVEATS + [v for k, v in EXTRA_CAVEATS.items() if c.result_id.startswith(k)]
        return envelope(ctx, "robustness_card", c, summary, caveats=caveats, assumptions=L1_ASSUMPTIONS, views=L1_VIEWS,
                        paths=("roadmap/phase7_stress_test/p7_03_bmi.json",), phase=7)
    caveats = L2_CAVEATS + ([c.measurement_note] if c.measurement_note else [])
    return envelope(ctx, "robustness_card", c, summary, caveats=caveats, assumptions=L2_ASSUMPTIONS, views=L2_VIEWS,
                    phase=7)


# ====================================================================== registration
def register(app: FastMCP, ctx: ServerContext) -> None:
    @app.tool(name="get_risk_factor_effect", title="Layer 1: chronic condition and acute MI",
              description=("Adjusted risk ratio and risk difference for acute MI at admission by chronic condition "
                           "(Layer 1), with E-value, identification note and Phase 7 verdict. 'primary' is the "
                           "adjustment set valid under the study's revised causal graph; 'alternative' is the other "
                           "set (diabetes bracket, roadmap set for hypertension, +prior MI for CKD)."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "get_risk_factor_effect")
    def _risk(
        exposure: Annotated[Literal["diabetes", "hypertension", "ckd", "prior_mi"], Field(description="Chronic condition")],
        adjustment: Annotated[Literal["primary", "alternative"], Field(description="Adjustment set")] = "primary",
        estimand: Annotated[Literal["ate_overlap", "att", "ate"],
                            Field(description="ate_overlap (primary), att (among the exposed), ate (all admissions)")] = "ate_overlap",
    ) -> Envelope[RiskFactorResult]:
        return get_risk_factor_effect(ctx, exposure, adjustment, estimand)

    @app.tool(name="get_ecg_effect", title="Layer 2: effect of acute MI on the ECG",
              description=("Average effect of acute MI on ECG measures among after-onset MI admissions (ATT, in "
                           "control-SD and raw units), with FDR q-value, Phase 6 classification and robustness flag. "
                           "Give one 'measure' (e.g. 'ST_lateral_mean', 'T wave V5', 'z_ecg_57') or one 'group'. "
                           "group='latent' returns the beta-VAE latent audit."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "get_ecg_effect")
    def _ecg(
        measure: Annotated[str | None, Field(description="One ECG measure; names, labels and common phrases are matched")] = None,
        group: Annotated[Literal["ST", "T", "Q", "R", "S", "QRS", "RR", "composite", "latent"] | None,
                         Field(description="A whole feature group instead of one measure")] = None,
        analysis_set: Annotated[AnalysisSet, Field(description="Which analysis (sensitivity sets cover point features and composites)")] = "primary",
        units: Annotated[Literal["sd", "raw"], Field(description="Units used in the summary sentence")] = "sd",
        top_k: Annotated[int, Field(ge=1, le=20, description="Group mode: number of largest effects to return")] = 10,
    ) -> Envelope[EcgEffectResult]:
        return get_ecg_effect(ctx, measure, group, analysis_set, units, top_k)

    @app.tool(name="get_subgroup_effects", title="Age and sex subgroups",
              description=("Estimates in age (<65 / >=65) or sex subgroups for one result, with an interaction test. "
                           "result_id is 'L1:<exposure>:primary' or 'L2:<measure>'."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "get_subgroup_effects")
    def _sub(
        result_id: Annotated[str, Field(description="e.g. 'L1:hypertension:primary' or 'L2:ST_lateral_mean'")],
        modifier: Annotated[Literal["age", "sex"], Field(description="Effect modifier")],
    ) -> Envelope[SubgroupResult]:
        return get_subgroup_effects(ctx, result_id, modifier)

    @app.tool(name="robustness_card", title="Robustness card",
              description=("One result against every stress test (alternative MI codes, NSTEMI-only, chest-pain "
                           "comparator, selection weighting, BMI, timing, subgroups, negative control, look-back) with "
                           "the pre-declared pass rule for each, the E-value and the overall verdict."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "robustness_card")
    def _card(
        result_id: Annotated[str, Field(description="e.g. 'L1:ckd:primary' or 'L2:ST_lateral_mean'")],
    ) -> Envelope[Card]:
        return robustness_card(ctx, result_id)
