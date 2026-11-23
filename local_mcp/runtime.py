"""Server context and the audited-call wrapper applied to every tool."""
import functools
import json
import logging
import time
from dataclasses import dataclass, field

from mcp.server.fastmcp.exceptions import ToolError

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
    cache: dict = field(default_factory=dict)   # derived from verified snapshots only (e.g. name-matching index)


def audited(ctx: ServerContext, tool: str, redact: tuple[str, ...] = ()):
    """Count, time and audit each call; turn unexpected failures into structured errors.

    functools.wraps keeps the original signature, which FastMCP uses to build the input schema.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx.calls += 1
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
            except ToolError as e:
                ctx.audit.record(tool, kwargs, "error", (time.perf_counter() - t0) * 1000, str(e), redact)
                raise
            except IntegrityError as e:
                body = json.dumps({"ok": False, "error": {"code": "INTEGRITY_MISMATCH", "message": str(e),
                                                          "paths": e.paths}})
                ctx.audit.record(tool, kwargs, "error", (time.perf_counter() - t0) * 1000, body, redact)
                raise ToolError(body) from None
            except Exception:
                log.exception("tool %s failed", tool)            # stderr only; never stdout
                body = json.dumps({"ok": False, "error": {"code": "INTERNAL",
                                                          "message": "Internal error; details are in the local server log."}})
                ctx.audit.record(tool, kwargs, "internal_error", (time.perf_counter() - t0) * 1000, body, redact)
                raise ToolError(body) from None
            ctx.audit.record(tool, kwargs, "ok", (time.perf_counter() - t0) * 1000,
                             out.model_dump_json() if hasattr(out, "model_dump_json") else str(out), redact)
            return out
        return wrapper
    return deco
