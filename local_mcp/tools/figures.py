"""render_figure: a study figure as a PNG image plus its caption and the data table behind it."""
import base64
import hashlib
import json
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel, Field

from local_mcp.envelope import Envelope
from local_mcp.runtime import ServerContext, audited
from local_mcp.tools.common import READ_ONLY, envelope, r4, require

FIGURES = {
    "st_t_effects": {
        "title": "ST and T effects of acute MI by lead",
        "caption": ("Effect of acute MI on ST level and T amplitude per lead (control-SD units, 95% CI), beside the "
                    "NSTEMI-only estimate and the negative control (ECG recorded before a later MI)."),
        "axes": {"x": "Effect of MI (control SD)", "y": "ECG measure (lead)"},
        "view": "deconfounded_ecg_effects", "phase": 6},
    "crude_vs_adjusted": {
        "title": "Crude vs adjusted ECG differences",
        "caption": ("Crude MI-vs-control difference against the adjusted effect for the 53 point features, coloured "
                    "by Phase 6 class; ST/T signs stay near the diagonal, R/Q/S differences shrink to zero."),
        "axes": {"x": "Crude difference (control SD)", "y": "Adjusted effect, ATT (control SD)"},
        "view": "deconfounded_ecg_effects", "phase": 6},
    "latent_audit": {
        "title": "Beta-VAE latent audit",
        "caption": ("Per latent dimension: variance explained by age, sex and comorbidities vs the extra variance "
                    "explained by MI (cross-fitted R2); all far below the identity line."),
        "axes": {"x": "R2 from baseline factors (log)", "y": "Extra R2 from MI (log)"},
        "view": "latent_audit", "phase": 6},
    "counterfactual_profiles": {
        "title": "Model-implied lateral ST shift by profile",
        "caption": ("Expected lateral ST level with and without MI for four illustrative profiles (Phase 6, "
                    "exploratory; model-implied, not observed counterfactuals)."),
        "axes": {"x": "Mean lateral ST level (mV)", "y": "Profile"},
        "view": "counterfactual_profiles", "phase": 6},
    "cf_pair_validation": {
        "title": "Individual counterfactuals vs real within-patient pairs",
        "caption": ("Absolute error (control SD, averaged over the 14 ST/T measures) of Track A counterfactuals on "
                    "held-out patients with an ECG before and during their MI, beside two baselines: copying the "
                    "other ECG and the population expectation. The shrunk model A2 is what counterfactual_ecg uses."),
        "axes": {"x": "Absolute error (control SD), lower is better", "y": "Method, per direction"},
        "view": "cf_pair_validation", "phase": 10},
    "cohort_flow": {
        "title": "Cohort flow",
        "caption": "Admissions at each step from MIMIC-IV v2.2 to the Layer 2 primary set.",
        "axes": {"x": "Admissions (log scale)", "y": "Step"},
        "view": "cohort_flow", "phase": 3},
}


class FigureResult(BaseModel):
    figure_id: str
    title: str
    caption: str
    axes: dict[str, str]
    image_included: bool
    image_sha256: str
    table_columns: list[str]
    table: list[dict]


def render_figure(ctx: ServerContext, figure_id: str, include_image: bool) -> CallToolResult:
    spec = FIGURES[figure_id]
    require(ctx, spec["view"])
    png_path = f"local_mcp/assets/figures/{figure_id}.png"
    png = ctx.registry.bytes(png_path)
    rows = ctx.sandbox.query(f"figure__{figure_id}")
    table = [{k: (r4(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]
    res = FigureResult(figure_id=figure_id, title=spec["title"], caption=spec["caption"], axes=spec["axes"],
                       image_included=include_image, image_sha256=hashlib.sha256(png).hexdigest(),
                       table_columns=list(rows[0]) if rows else [], table=table)
    env: Envelope = envelope(ctx, "render_figure", res, f"{spec['title']}: {spec['caption']}",
                             caveats=["The image and the table show the same verified results."],
                             views=(spec["view"],), paths=(png_path,), phase=spec["phase"])
    body = env.model_dump(mode="json")
    content = [ImageContent(type="image", data=base64.b64encode(png).decode(), mimeType="image/png")] if include_image else []
    content.append(TextContent(type="text", text=json.dumps(body)))
    return CallToolResult(content=content, structuredContent=body)


def register(app: FastMCP, ctx: ServerContext) -> None:
    @app.tool(name="render_figure", title="Render a study figure",
              description=("A study figure as a PNG image, with its caption, axis definitions and the data table "
                           "behind it (clients that cannot show images still get the table)."),
              annotations=READ_ONLY, structured_output=True)
    @audited(ctx, "render_figure")
    def _figure(
        figure_id: Annotated[Literal["st_t_effects", "crude_vs_adjusted", "latent_audit", "counterfactual_profiles",
                                     "cohort_flow", "cf_pair_validation"], Field(description="Which figure")],
        include_image: Annotated[bool, Field(description="Include the PNG image")] = True,
    ) -> Annotated[CallToolResult, Envelope[FigureResult]]:
        return render_figure(ctx, figure_id, include_image)
