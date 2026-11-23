"""trace_claim: any manuscript claim -> evidence files with hashes, assumptions, strength, and the tools that
reproduce its numbers. Only manifest files are hashed; other cited files are listed as not verified.
"""
import fnmatch
import re
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from local_mcp.envelope import Assumption, Envelope, fail
from local_mcp.runtime import ServerContext, audited
from local_mcp.tools.common import ASSUMPTIONS, READ_ONLY, envelope, require

VIEW_TOOLS = {
    "layer1_estimates": "get_risk_factor_effect", "layer1_diagnostics": "get_risk_factor_effect",
    "layer2_estimates": "get_ecg_effect", "deconfounded_ecg_effects": "get_ecg_effect",
    "sensitivity_estimates": "get_ecg_effect", "latent_audit": "get_ecg_effect",
    "layer2_test_stability": "get_ecg_effect", "feature_group_summary": "get_ecg_effect",
    "evalues": "robustness_card", "outcome_codes_layer1": "robustness_card", "outcome_codes_layer2_broad": "robustness_card",
    "selection_weighted_layer2": "robustness_card", "bmi_layer2_with": "robustness_card",
    "bmi_layer2_without": "robustness_card", "layer1_history_variants": "robustness_card",
    "subgroups_layer1": "get_subgroup_effects", "subgroups_layer2": "get_subgroup_effects",
    "subgroup_interactions_layer2": "get_subgroup_effects", "cohort_flow": "study_overview",
    "independence_tests": "study_overview", "adjustment_table": "get_risk_factor_effect",
}
PATH_TOOLS = {"roadmap/phase7_stress_test/p7_03_bmi.json": "robustness_card",
              "roadmap/phase7_stress_test/p7_05_subgroups.json": "get_subgroup_effects"}


class Evidence(BaseModel):
    cited: str
    path: str
    in_manifest: bool
    verified: bool
    sha256: str | None = None


class ClaimTrace(BaseModel):
    claim_id: str
    section: str
    claim: str
    strength: str
    assumptions: list[Assumption]
    assumption_notes: str
    evidence: list[Evidence]
    reproduce_with: list[str]


class TraceResult(BaseModel):
    claims: list[ClaimTrace]
    matched_by: str


def _evidence(ctx: ServerContext, cited: str) -> list[Evidence]:
    out = []
    for token in [t.strip() for t in cited.split(";") if t.strip()]:
        path = re.sub(r"\s*\(.*\)$", "", token).split(" ")[0]
        if path.startswith("phase"):
            path = "roadmap/" + path
        matches = [a for a in ctx.registry.artifacts if fnmatch.fnmatch(a.path, path)] if "*" in path else \
                  [a for a in [ctx.registry.find(path)] if a is not None]
        if matches:
            out += [Evidence(cited=token, path=a.path, in_manifest=True, verified=a.status == "ok", sha256=a.sha256)
                    for a in matches]
        else:
            out.append(Evidence(cited=token, path=path, in_manifest=False, verified=False))
    return out


def _trace(ctx: ServerContext, row: dict) -> ClaimTrace:
    ids = list(dict.fromkeys(re.findall(r"\bA[1-6]\b", row["assumptions"] or "")))
    ev = _evidence(ctx, row["evidence_files"] or "")
    tools = []
    for e in ev:
        a = ctx.registry.find(e.path)
        t = VIEW_TOOLS.get(a.view) if a is not None and a.view else PATH_TOOLS.get(e.path)
        if t and t not in tools:
            tools.append(t)
    return ClaimTrace(claim_id=row["claim_id"], section=row["manuscript_section"], claim=row["claim"],
                      strength=row["strength"], assumptions=[Assumption(id=i, text=ASSUMPTIONS[i]) for i in ids],
                      assumption_notes=row["assumptions"] or "", evidence=ev, reproduce_with=tools)


def trace_claim(ctx: ServerContext, claim_id: str | None, query: str | None) -> Envelope:
    require(ctx, "claim_evidence_map")
    if (claim_id is None) == (query is None):
        fail("INVALID_ARGUMENT", "Give exactly one of 'claim_id' or 'query'.")
    rows = ctx.sandbox.query("claims__all")
    if claim_id is not None:
        m = re.fullmatch(r"[Cc]?0*(\d{1,2})", claim_id.strip())
        cid = f"C{int(m.group(1)):02d}" if m else claim_id.strip()
        hit = [r for r in rows if r["claim_id"] == cid]
        if not hit:
            fail("INVALID_ARGUMENT", f"Unknown claim {claim_id!r}.", valid_values=[r["claim_id"] for r in rows])
        matched = "claim_id"
    else:
        words = set(re.findall(r"[a-z0-9]+", query.lower())) - {"the", "of", "and", "a", "in", "is", "to", "on", "mi"}
        scored = sorted(((len(words & set(re.findall(r"[a-z0-9]+", (r["claim"] + " " + r["strength"]).lower()))), r)
                         for r in rows), key=lambda t: -t[0])
        hit = [r for s, r in scored if s > 0][:3]
        if not hit:
            fail("INVALID_ARGUMENT", f"No claim matches {query!r}.", valid_values=[r["claim_id"] for r in rows])
        matched = "keyword query"
    traces = [_trace(ctx, r) for r in hit]
    t0 = traces[0]
    verified = sum(e.verified for e in t0.evidence)
    summary = (f"{t0.claim_id} ({t0.section}): \"{t0.claim}\" rests on {', '.join(a.id for a in t0.assumptions) or 'no listed assumptions'}; "
               f"{verified}/{len(t0.evidence)} evidence files verified by hash; strength: {t0.strength}.")
    return envelope(ctx, "trace_claim", TraceResult(claims=traces, matched_by=matched), summary,
                    views=("claim_evidence_map",), phase=8, tier="none")


def register(app: FastMCP, ctx: ServerContext) -> None:
    @app.tool(name="trace_claim", title="Trace a manuscript claim",
              description=("Trace a manuscript claim (C01-C20) to its evidence files with SHA-256 hashes verified at "
                           "server start, the assumptions it rests on, its strength, and the tools that reproduce it. "
                           "Give 'claim_id' or a keyword 'query'."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "trace_claim")
    def _trace_claim(
        claim_id: Annotated[str | None, Field(description="e.g. 'C06'")] = None,
        query: Annotated[str | None, Field(max_length=200, description="Keywords, e.g. 'NSTEMI coding'")] = None,
    ) -> Envelope[TraceResult]:
        return trace_claim(ctx, claim_id, query)
