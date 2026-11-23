"""Entry point for the local MCP server (stdio). Clients launch:
    <project python> -X utf8 <repo>/local_mcp/server.py

Nothing in this process may write to stdout except the MCP transport; logs go to stderr.
"""
import logging
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
# Run as a script, sys.path[0] is local_mcp/ itself; replace it with the repo root so that module names in
# this folder (db, paths, ...) cannot shadow other packages and `import local_mcp` resolves.
if sys.path and Path(sys.path[0]).resolve() == _PKG:
    sys.path[0] = str(_PKG.parent)
elif str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from local_mcp import guards  # noqa: E402

guards.install()  # before anything else can open a socket

from mcp.server.fastmcp import FastMCP  # noqa: E402

from local_mcp import SERVER_NAME  # noqa: E402
from local_mcp.audit import AuditLog  # noqa: E402
from local_mcp.db import Sandbox  # noqa: E402
from local_mcp.paths import AUDIT_LOG  # noqa: E402
from local_mcp.registry import Registry  # noqa: E402
from local_mcp.runtime import ServerContext  # noqa: E402
from local_mcp.tools import status  # noqa: E402

INSTRUCTIONS = (
    "Local server for a causal study of acute myocardial infarction (MIMIC-IV, hospitalized adults). "
    "All computation runs on the user's machine; tools return only aggregate results, never patient rows. "
    "Call server_status to check that the data and the sandbox are intact. Quote each tool's 'summary' and "
    "'caveats' with any number you report."
)


def build_app() -> FastMCP:
    registry = Registry()
    sandbox, sandbox_error = None, None
    try:
        sandbox = Sandbox(registry)
    except Exception as e:  # keep serving server_status so the problem is visible to the client
        sandbox_error = f"{type(e).__name__}: {e}"
        logging.getLogger("local_mcp").error("DuckDB sandbox unavailable: %s", sandbox_error)
    ctx = ServerContext(registry=registry, sandbox=sandbox, audit=AuditLog(AUDIT_LOG), sandbox_error=sandbox_error)
    app = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS, log_level="WARNING")
    status.register(app, ctx)
    return app


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    build_app().run("stdio")


if __name__ == "__main__":
    main()
