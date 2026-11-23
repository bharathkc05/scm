"""Orientation tools: study_overview (call first) and describe_variable (data dictionary)."""
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from local_mcp.envelope import Envelope, fail
from local_mcp.runtime import ServerContext, audited
from local_mcp.tools.common import EXPOSURE_LABELS, READ_ONLY, _indexed, envelope, match_name, require
from local_mcp.tools.robustness import l1_card, l2_card

COHORT_STEPS = {
    "All admissions (MIMIC-IV v2.2)": "all_admissions",
    "Layer 1: with >=1 diagnosis code": "layer1_admissions",
    "of which acute MI": "layer1_acute_mi",
    "Layer 2 primary set": "layer2_primary_admissions",
    "acute MI, after onset": "layer2_after_onset_mi",
    "controls (no acute MI)": "layer2_controls",
}


# ====================================================================== study_overview
class Headline(BaseModel):
    result_id: str
    description: str
    scale: str
    estimate: float
    ci95: list[float]
    verdict: str


class OverviewResult(BaseModel):
    question: str
    data: str
    design: list[str]
    cohort: dict[str, int]
    estimands: dict[str, str]
    headline_layer1: list[Headline]
    headline_layer2: list[Headline]
    layer2_significance: dict[str, str]
    latent_conclusion: str
    tiers_present: dict[str, str]
    result_id_format: dict[str, str]
    suggested_next_tools: list[str]


def study_overview(ctx: ServerContext) -> Envelope:
    require(ctx, "cohort_flow", "l1_effects", "l2_effects", "l2_class")
    cohort = {}
    for r in ctx.sandbox.query("overview__cohort_flow"):
        key = COHORT_STEPS.get(r["step"].strip())
        if key:
            cohort[key] = int(r["admissions"])
    l1 = []
    for exp in ("diabetes", "hypertension", "ckd", "prior_mi"):
        c = l1_card(ctx, f"L1:{exp}:primary")
        l1.append(Headline(result_id=c.result_id, description=f"{EXPOSURE_LABELS[exp]} -> acute MI ({c.label})",
                           scale="risk ratio", estimate=c.primary.estimate, ci95=c.primary.ci95, verdict=c.verdict))
    l2 = []
    for r in ctx.sandbox.query("overview__l2_robust"):
        if r["measure"] == "QRS_Dur_Global":          # excluded from claims: measurement not validated
            continue
        c = l2_card(ctx, r["measure"])
        l2.append(Headline(result_id=c.result_id, description=c.label, scale="ATT, control SD",
                           estimate=c.primary.estimate, ci95=c.primary.ci95, verdict=c.verdict))
        if len(l2) == 6:
            break
    counts = {r["family"]: f"{r['significant']}/{r['tests']} FDR-significant" for r in ctx.sandbox.query("overview__l2_counts")}
    res = OverviewResult(
        question=("Two-layer structural causal model of acute myocardial infarction: (1) how chronic conditions relate "
                  "to acute MI at admission; (2) what acute MI does to the 12-lead ECG, separating MI-caused changes "
                  "from differences in who has an MI."),
        data="MIMIC-IV v2.2 and MIMIC-IV-ECG v1.0; single centre (Beth Israel Deaconess); hospitalized adults.",
        design=["Unit: hospital admission; patients kept whole in one data split.",
                "Layer 2 uses the earliest ECG with extracted features, only when it followed MI onset "
                "(MI is the principal diagnosis of an unplanned admission).",
                "Revised causal graph (diabetes -> hypertension plus a shared metabolic cause) chosen after a failed "
                "independence test.",
                "Cross-fitted doubly robust estimators (AIPW), patient-level bootstrap, Benjamini-Hochberg FDR.",
                "Stress tests: NSTEMI-only, chest-pain comparator, negative control, ECG-selection weighting, BMI, "
                "alternative MI codes, subgroups, E-values."],
        cohort=cohort,
        estimands={"layer1": "Average effect in the overlap population (risk ratio and risk difference)",
                   "layer2": "Average effect among after-onset MI admissions (ATT), in control-SD units"},
        headline_layer1=l1, headline_layer2=l2, layer2_significance=counts,
        latent_conclusion=("MI explains a median 0.009% of each beta-VAE latent's variance; the latents add no MI "
                           "information beyond the handcrafted features (negative result)."),
        tiers_present=ctx.registry.tiers(),
        result_id_format={"layer1": "L1:<diabetes|hypertension|ckd|prior_mi>:<primary|alternative>",
                          "layer2": "L2:<measure>, e.g. L2:ST_lateral_mean"},
        suggested_next_tools=["get_risk_factor_effect", "get_ecg_effect", "robustness_card", "confounding_what_if",
                              "get_subgroup_effects", "explain_ecg", "cohort_summary", "render_figure",
                              "describe_variable", "trace_claim"])
    survives = sum(h.verdict.startswith("survives") for h in l1)
    summary = (f"{cohort.get('layer1_admissions', 0):,} admissions ({cohort.get('layer1_acute_mi', 0):,} acute MI) for "
               f"Layer 1 and {cohort.get('layer2_primary_admissions', 0):,} ({cohort.get('layer2_after_onset_mi', 0):,} "
               f"after-onset MI) for Layer 2. {survives}/4 chronic-condition results survive the stress tests (CKD is "
               f"fragile); acute MI causes lateral ST depression and T-wave flattening with reciprocal aVR changes.")
    return envelope(ctx, "study_overview", res, summary,
                    caveats=["Results are causal only under the stated assumptions; Layer 1 contrasts are risk-factor "
                             "associations among hospitalized adults."],
                    assumptions=("A1", "A2", "A3", "A4", "A5", "A6"),
                    views=("cohort_flow", "layer1_estimates", "layer2_estimates", "deconfounded_ecg_effects"), phase=8)


# ====================================================================== describe_variable
class VariableInfo(BaseModel):
    name: str
    label: str
    category: str
    layer: str | None
    role: str | None
    definition: str
    icd9: str | None
    icd10: str | None
    units: str | None
    family: str | None
    group: str | None
    source: str | None


class DescribeResult(BaseModel):
    variable: VariableInfo
    matched_from: str


def describe_variable(ctx: ServerContext, name: str) -> Envelope:
    entries, index = _indexed(ctx, None)
    found, suggestions = match_name(name, entries, index)
    if found is None:
        fail("INVALID_ARGUMENT", f"No variable matches {name!r}.", did_you_mean=suggestions)
    row = ctx.sandbox.query("variables__by_name", [found])[0]
    clean = {k: (v if v not in ("", None) else None) for k, v in row.items()}
    var = VariableInfo(name=clean["name"], label=clean["label"], category=clean["category"], layer=clean["layer"],
                       role=clean["role"], definition=clean["definition"], icd9=clean["icd9"], icd10=clean["icd10"],
                       units=clean["units"], family=clean["family"], group=clean["grp"], source=clean["source"])
    codes = "; ".join(x for x in (f"ICD-9 {var.icd9}" if var.icd9 else "", f"ICD-10 {var.icd10}" if var.icd10 else "") if x)
    summary = f"{var.label} ({var.name}): {var.definition}" + (f" Codes: {codes}." if codes else "")
    return envelope(ctx, "describe_variable", DescribeResult(variable=var, matched_from=name), summary,
                    views=("variables",), tier="none")


def register(app: FastMCP, ctx: ServerContext) -> None:
    @app.tool(name="study_overview", title="Study overview (call first)",
              description=("Call this first. The study question, data, design, cohort sizes, estimands, headline "
                           "results with their result IDs and robustness verdicts, and which tools to use next."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "study_overview")
    def _overview() -> Envelope[OverviewResult]:
        return study_overview(ctx)

    @app.tool(name="describe_variable", title="Describe a variable",
              description=("Definition, ICD code lists, units and causal role of a study variable or ECG measure. "
                           "Accepts exact names or plain phrases such as 'heart attack', 'lateral ST' or 'T wave V5'."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "describe_variable")
    def _describe(
        name: Annotated[str, Field(min_length=1, max_length=80, description="Variable name or plain-language phrase")],
    ) -> Envelope[DescribeResult]:
        return describe_variable(ctx, name)
