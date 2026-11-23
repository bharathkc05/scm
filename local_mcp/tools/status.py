"""server_status: integrity, tiers present, DuckDB sandbox state, network guard, audit chain, versions."""
import importlib.metadata
import platform
import time
from typing import Annotated

import duckdb
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from local_mcp import SERVER_NAME, __version__, audit, guards
from local_mcp.envelope import Disclosure, Envelope, Provenance
from local_mcp.paths import rel
from local_mcp.registry import IntegrityError
from local_mcp.runtime import ServerContext, audited


class Tiers(BaseModel):
    T1: str = Field(description="patient-level analysis data: present | partial | absent")
    T2: str = Field(description="aggregate result tables")
    T3: str = Field(description="model parameters and assets")
    code: str = Field(description="reviewed SQL templates")


class Integrity(BaseModel):
    verified: int
    missing: list[str]
    mismatched: list[str]
    changed_since_start: list[str]
    t1_check: str = Field(description="stat (size + modified time) | full_hash")


class SandboxState(BaseModel):
    duckdb_version: str
    available: bool
    error: str | None = None
    external_access: bool | None = None
    config_locked: bool | None = None
    allowlisted_files: int = 0
    views: list[str] = Field(default_factory=list)
    self_test: dict[str, bool] = Field(default_factory=dict, description="True = blocked as expected")


class GuardState(BaseModel):
    active: bool
    blocked_attempts: int


class AuditState(BaseModel):
    path: str
    entries: int
    chain_ok: bool


class StatusResult(BaseModel):
    server: str
    mcp_sdk: str
    python: str
    transport: str = "stdio"
    manifest_version: str | None
    manifest_built_at: str | None
    manifest_error: str | None = None
    tiers: Tiers
    integrity: Integrity
    duckdb_sandbox: SandboxState
    network_guard: GuardState
    raw_mimic_registered: bool
    view_row_counts: dict[str, int]
    calls_this_session: int
    uptime_s: float
    audit: AuditState


def _status(ctx: ServerContext, verify_full: bool) -> Envelope[StatusResult]:
    reg = ctx.registry
    if verify_full:
        reg.verify(full=True)
    sb = ctx.sandbox
    if sb is not None:
        try:
            views = [r["view_name"] for r in sb.query("status__views")]
        except IntegrityError:          # status must still report when artifacts changed under it
            views = list(sb.views)
        sandbox = SandboxState(duckdb_version=duckdb.__version__, available=True, **sb.settings(),
                               allowlisted_files=len(sb.allowlist), views=views, self_test=sb.self_test())
    else:
        sandbox = SandboxState(duckdb_version=duckdb.__version__, available=False, error=ctx.sandbox_error)
    chain = audit.verify(ctx.audit.path)
    t1 = [a for a in reg.artifacts if a.tier == "T1"]
    result = StatusResult(
        server=f"{SERVER_NAME} {__version__}",
        mcp_sdk=importlib.metadata.version("mcp"),
        python=platform.python_version(),
        manifest_version=reg.version,
        manifest_built_at=reg.built_at,
        manifest_error=reg.error,
        tiers=Tiers(**reg.tiers()),
        integrity=Integrity(verified=len(reg.by_status("ok")), missing=reg.by_status("missing"),
                            mismatched=reg.by_status("mismatch"), changed_since_start=reg.changed_since_start(),
                            t1_check="full_hash" if any(a.check == "full_hash" for a in t1) else "stat"),
        duckdb_sandbox=sandbox,
        network_guard=GuardState(**guards.state()),
        raw_mimic_registered=reg.raw_mimic_registered(),
        view_row_counts=sb.row_counts if sb is not None else {},
        calls_this_session=ctx.calls,
        uptime_s=round(time.time() - ctx.started, 1),
        audit=AuditState(path=rel(ctx.audit.path), entries=chain["entries"], chain_ok=chain["ok"]),
    )
    healthy = (reg.error is None and not result.integrity.mismatched and not result.integrity.changed_since_start
               and sandbox.available and sandbox.config_locked and not sandbox.external_access
               and all(sandbox.self_test.values()) and result.network_guard.active and chain["ok"])
    if healthy:
        summary = (f"Local server healthy: {result.integrity.verified} artifacts verified, DuckDB sandbox locked "
                   f"with {sandbox.allowlisted_files} allowlisted files, network guard active, audit chain intact.")
    else:
        summary = "Local server has problems; see integrity, duckdb_sandbox, network_guard and audit fields."
    caveats = []
    if result.tiers.T1 != "present":
        caveats.append("Patient-level files are not all present on this machine; cohort_summary will be unavailable.")
    return Envelope[StatusResult](
        tool="server_status", result=result, summary=summary, caveats=caveats,
        provenance=Provenance(manifest_version=reg.version), disclosure=Disclosure(tier="none"))


def register(app: FastMCP, ctx: ServerContext) -> None:
    @app.tool(
        name="server_status",
        title="Server status and integrity",
        description=("Health of the local study server: which data tiers are on this machine, artifact integrity, "
                     "the DuckDB sandbox (allowlisted files only, no network, locked) with live self-test probes, "
                     "the outbound-network guard, and the audit-log chain. Returns no study data."),
        annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    @audited(ctx, "server_status")
    def server_status(
        verify_full: Annotated[bool, Field(description="Also re-hash the large patient-level files (slower).")] = False,
    ) -> Envelope[StatusResult]:
        return _status(ctx, verify_full)
