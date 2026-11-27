"""Shared helpers for the study tools: envelopes, provenance, name matching, E-values and server-authored text.

The interpretation text here (assumptions, caveats, identification notes) is written by the server so every
client and every LLM receives the same wording with the numbers.
"""
import difflib
import math
import re

from mcp.types import ToolAnnotations

from local_mcp.envelope import ArtifactRef, Assumption, Disclosure, Envelope, Provenance, fail
from local_mcp.runtime import ServerContext

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

ASSUMPTIONS = {
    "A1": "Diagnosis codes at or before the admission reflect conditions present before the MI.",
    "A2": "After-onset ECGs (MI is the principal diagnosis of an unplanned admission) were recorded after MI onset.",
    "A3": "No unmeasured common cause of MI and the ECG beyond age, sex, diabetes, hypertension, CKD and prior MI.",
    "A4": "Whether an ECG was recorded is ignorable given MI and the measured covariates.",
    "A5": "No unmeasured common cause of each chronic condition and MI (smoking and lipids violate this; E-values bound it).",
    "A6": "The shared metabolic cause of diabetes and hypertension acts on MI only through them.",
    "CF-A2": "The patient's own deviation from the expected ECG does not change with MI; the share that persisted "
             "between real visits (rho, fitted on validation pairs) is carried into the counterfactual.",
    "CF-A3": "The MI effect depends only on age, sex, comorbidities and MI type, not on unmeasured traits "
             "(not identifiable; bounded by the pair-validation error).",
    "CF-A4": "Another visit's ECG is an acceptable stand-in for the patient without MI.",
}
L1_ASSUMPTIONS = ("A1", "A5", "A6")
L2_ASSUMPTIONS = ("A2", "A3", "A4")

L1_CAVEATS = [
    "Risk-factor contrast among hospitalized adults, not the effect of an intervention.",
    "Everyone studied was hospitalized; selection into hospital can distort risk-factor associations.",
]
L2_CAVEATS = [
    "Average effect of MI among after-onset MI admissions (ATT); causal only under assumptions A2-A4.",
    "ECGs recorded before a later MI (negative control) show about 38% of the ST/T magnitude: up to that share "
    "may be residual bias or pre-existing coronary disease.",
]
EXPOSURE_LABELS = {"diabetes": "diabetes", "hypertension": "hypertension", "ckd": "chronic kidney disease",
                   "prior_mi": "prior MI"}
IDENTIFICATION = {
    "L1:diabetes:primary": "Not identified under the revised graph (diabetes -> hypertension plus a shared "
                           "metabolic cause); this set is valid only if there is no shared cause. Read it with the "
                           "alternative set as a bracket.",
    "L1:diabetes:alternative": "Not identified under the revised graph; this set (adding hypertension) is valid only "
                               "if hypertension causes diabetes or they share a cause without diabetes -> hypertension. "
                               "Read it with the primary set as a bracket.",
    "L1:hypertension:primary": "Valid under the revised graph and every variant except hypertension -> diabetes.",
    "L1:hypertension:alternative": "Roadmap set (age, sex): valid only if hypertension causes diabetes.",
    "L1:ckd:primary": "Valid under every graph variant except prior MI -> CKD.",
    "L1:ckd:alternative": "Valid only if prior MI causes CKD.",
    "L1:prior_mi:primary": "Descriptive only: unmeasured coronary disease causes both prior and new MI.",
}
EXTRA_CAVEATS = {
    "L1:diabetes": "The total effect of diabetes is not identified; the two adjustment sets bracket it.",
    "L1:ckd": "Fragile: the association disappears for MI coded as the principal diagnosis.",
    "L1:prior_mi": "Descriptive association, not a causal effect.",
}
MEASUREMENT_NOTES = {
    "QRS_Dur_Global": "Not validated: the ECG cart's own QRS duration shows no MI effect (+0.4 ms, 95% CI -0.9 to 1.6).",
    "RR_Mean_Global": "Validated against the machine RR interval, but equally present before MI (negative control), "
                      "so it reflects the patient rather than the event.",
}
ADJUST_LABELS = {"age(spline)": "age (spline)", "sex_male": "sex", "diabetes_primary": "diabetes",
                 "hypertension_primary": "hypertension", "ckd_primary": "CKD", "prior_mi_primary": "prior MI"}
SETS = {
    "primary": "all after-onset MI vs controls",
    "nstemi": "NSTEMI only (diagnosis not based on ST elevation)",
    "chest_pain_comparator": "vs chest-pain admissions without MI",
    "after_strict": "admission context and troponin timing agree",
    "no_type2": "type 2 MI excluded",
    "first_ecg_has_features": "first ECG of the admission analysed",
    "selection_weighted": "weighted for which admissions had an ECG",
    "bmi_adjusted": "BMI-available admissions, adjusted for BMI",
    "broad_codes": "broad MI codes including 410.x2 follow-ups",
    "negative_control": "negative control: ECG recorded before a later first MI",
}


# ------------------------------------------------------------------ envelopes and data checks
def require(ctx: ServerContext, *relations: str) -> None:
    if ctx.sandbox is None or not ctx.sandbox.has(*relations):
        fail("DATA_NOT_PRESENT", "The result files this tool needs are not available on this machine.",
             needs=list(relations), hint="Call server_status to see which tiers are present.")


def envelope(ctx: ServerContext, tool: str, result, summary: str, *, caveats=(), assumptions=(), views=(),
             paths=(), phase: int | None = None, tier: str = "aggregate") -> Envelope:
    refs = []
    for v in views:
        r = ctx.registry.ref(v)
        if r and ArtifactRef(path=r[0], sha256=r[1]) not in refs:
            refs.append(ArtifactRef(path=r[0], sha256=r[1]))
    for p in paths:
        a = ctx.registry.find(p)
        if a is not None:
            refs.append(ArtifactRef(path=a.path, sha256=a.sha256))
    return Envelope[type(result)](
        tool=tool, result=result, summary=summary, caveats=list(dict.fromkeys(caveats)),
        assumptions=[Assumption(id=a, text=ASSUMPTIONS[a]) for a in dict.fromkeys(assumptions)],
        provenance=Provenance(artifacts=refs, manifest_version=ctx.registry.version, phase=phase),
        disclosure=Disclosure(tier=tier))


# ------------------------------------------------------------------ numbers
def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def sig(x, digits: int = 3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return float(f"{float(x):.{digits}g}")


def evalue_rr(rr: float) -> float:
    r = rr if rr >= 1 else 1 / rr
    return r + math.sqrt(r * (r - 1))


def evalue_rr_ci(lo: float, hi: float) -> float:
    if lo <= 1 <= hi:
        return 1.0
    return evalue_rr(lo if lo > 1 else hi)


def evalue_sd(d: float, lo: float, hi: float) -> tuple[float, float]:
    """E-value for a standardized difference via RR ~ exp(0.91 |d|) (VanderWeele & Ding 2017)."""
    point = evalue_rr(math.exp(0.91 * abs(d)))
    near = lo if d > 0 else hi
    ci = evalue_rr(math.exp(0.91 * abs(near))) if (near > 0) == (d > 0) and near != 0 else 1.0
    return point, ci


# ------------------------------------------------------------------ names
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def build_index(entries: list[dict]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for e in entries:
        for s in [e["name"], e.get("label") or "", *((e.get("synonyms") or "").split(";"))]:
            n = _norm(s)
            if n:
                index.setdefault(n, set()).add(e["name"])
                index.setdefault(n.replace(" ", ""), set()).add(e["name"])
    return index


def match_name(query: str, entries: list[dict], index: dict[str, set[str]] | None = None) -> tuple[str | None, list[str]]:
    """Match free text to a dictionary name via names, labels and synonyms; otherwise suggest candidates."""
    q = _norm(re.sub(r"^w_", "", str(query).strip()))
    index = index if index is not None else build_index(entries)
    hit = index.get(q) or index.get(q.replace(" ", ""))
    if hit and len(hit) == 1:
        return next(iter(hit)), []
    if hit:
        return None, sorted(hit)[:5]
    out: list[str] = []
    for c in difflib.get_close_matches(q, list(index), n=10, cutoff=0.6):
        for n in sorted(index[c]):
            if n not in out:
                out.append(n)
    return None, out[:5]


def dictionary(ctx: ServerContext) -> list[dict]:
    require(ctx, "variables")
    if "dictionary" not in ctx.cache:              # the variables table is a verified in-memory snapshot
        ctx.cache["dictionary"] = ctx.sandbox.query("variables__names")
    return ctx.cache["dictionary"]


def _indexed(ctx: ServerContext, categories: tuple[str, ...] | None) -> tuple[list[dict], dict]:
    key = ("index", categories)
    if key not in ctx.cache:
        entries = [e for e in dictionary(ctx) if not categories or e["category"] in categories]
        ctx.cache[key] = (entries, build_index(entries))
    return ctx.cache[key]


def resolve_variable(ctx: ServerContext, name: str, categories: tuple[str, ...] | None = None) -> dict:
    entries, index = _indexed(ctx, categories)
    found, suggestions = match_name(name, entries, index)
    if found is None:
        fail("INVALID_ARGUMENT", f"Unknown {'measure' if categories else 'variable'} {name!r}.",
             did_you_mean=suggestions, valid_values_tool="describe_variable")
    return next(e for e in entries if e["name"] == found)
