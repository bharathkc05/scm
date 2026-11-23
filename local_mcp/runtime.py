"""Server context and the audited-call wrapper applied to every tool.

Every tool call goes through `audited`: it counts and times the call, passes the result (or the error text)
through the egress guard, writes one hash-chained audit line, and turns unexpected failures into structured
errors. Nothing leaves the process without passing the egress guard.
"""
import functools
import json
import logging
import time
from dataclasses import dataclass, field

from mcp.server.fastmcp.exceptions import ToolError

from local_mcp import egress
from local_mcp.audit import AuditLog
from local_mcp.db import Sandbox
from local_mcp.registry import IntegrityError, Registry

log = logging.getLogger("local_mcp")


@dataclass
class ServerContext:
    registry: Registry
    sandbox: Sandbox | None
    audit: AuditLog
    sandbox_error: str | None = None
    started: float = field(default_factory=time.time)
    calls: int = 0
    egress_blocked: int = 0
    cache: dict = field(default_factory=dict)   # derived from verified snapshots only (e.g. name-matching index)


def _blocked_body(codes: list[str]) -> str:
    return json.dumps({"ok": False, "error": {
        "code": "DISCLOSURE_BLOCKED",
        "message": "The local egress guard withheld this response because it looked like patient-level data.",
        "checks_failed": codes}})


def audited(ctx: ServerContext, tool: str, redact: tuple[str, ...] = ()):
    """Count, time, guard and audit each call.

    functools.wraps keeps the original signature, which FastMCP uses to build the input schema.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx.calls += 1
            t0 = time.perf_counter()

            def record(outcome: str, output: str) -> None:
                ctx.audit.record(tool, kwargs, outcome, (time.perf_counter() - t0) * 1000, output, redact)

            def block(violations: list[dict]):
                ctx.egress_blocked += 1
                codes = sorted({v["code"] for v in violations})
                log.warning("egress guard blocked %s: %s", tool, ", ".join(f"{v['code']} at {v['at']}" for v in violations))
                body = _blocked_body(codes)
                record("blocked", body)
                raise ToolError(body) from None

            try:
                out = fn(*args, **kwargs)
            except ToolError as e:
                text = str(e)
                hits = egress.scan_text(text)                     # error text can echo inputs; scan it too
                if hits:
                    block(hits)
                record("error", text)
                raise
            except IntegrityError as e:
                body = json.dumps({"ok": False, "error": {"code": "INTEGRITY_MISMATCH", "message": str(e),
                                                          "paths": e.paths}})
                record("error", body)
                raise ToolError(body) from None
            except Exception:
                log.exception("tool %s failed", tool)            # stderr only; never stdout
                body = json.dumps({"ok": False, "error": {"code": "INTERNAL",
                                                          "message": "Internal error; details are in the local server log."}})
                record("internal_error", body)
                raise ToolError(body) from None
            violations = egress.check(egress.payload_of(out))
            if violations:
                block(violations)
            record("ok", out.model_dump_json() if hasattr(out, "model_dump_json") else str(out))
            return out
        return wrapper
    return deco
