"""Render the MCP client configs for this machine, and optionally install one.

The committed templates in `templates/` hold two placeholders: {{PYTHON}} (the interpreter that has `mcp` and
`duckdb`) and {{SCM_ROOT}} (this repository). Rendered files contain machine-specific absolute paths, so they go
to the git-ignored `rendered/` folder, and installed client files are git-ignored too.

    python -X utf8 local_mcp/configs/render.py                       # render every client into rendered/
    python -X utf8 local_mcp/configs/render.py --print claude_desktop # print one config to paste by hand
    python -X utf8 local_mcp/configs/render.py --install cursor       # merge into <repo>/.cursor/mcp.json
    python -X utf8 local_mcp/configs/render.py --install claude_desktop --target "$APPDATA/Claude/claude_desktop_config.json"

`--install` merges the `scm-causal-mi` entry into the client's existing file and keeps every other server;
the previous file is saved next to it as `.bak`. The interpreter defaults to the one running this script.
Standard library only, so it runs before the server's dependencies are checked.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
RENDERED = HERE / "rendered"
REPO = HERE.parents[1]
SERVER_NAME = "scm-causal-mi"
PLACEHOLDERS = ("{{PYTHON}}", "{{SCM_ROOT}}")


@dataclass(frozen=True)
class Client:
    label: str
    key: str                 # top-level key that holds the servers
    target: str | None       # repo-relative file --install writes; None = outside the repo, paste by hand
    note: str                # where the file lives / how the client picks it up


CLIENTS: dict[str, Client] = {
    "claude_code": Client("Claude Code", "mcpServers", ".mcp.json",
                          "project scope; Claude Code asks once to approve project servers. "
                          "Or: claude mcp add-json / --mcp-config <file>"),
    "cursor": Client("Cursor", "mcpServers", ".cursor/mcp.json",
                     "project scope; global alternative ~/.cursor/mcp.json"),
    "vscode": Client("GitHub Copilot (VS Code)", "servers", ".vscode/mcp.json",
                     "workspace scope; start the server from the file's CodeLens or 'MCP: List Servers'"),
    "antigravity": Client("Antigravity", "mcpServers", ".agents/mcp_config.json",
                          "workspace scope; global alternative ~/.gemini/config/mcp_config.json. "
                          "Optional enabledTools / disabledTools limit what the agent sees"),
    "claude_desktop": Client("Claude Desktop", "mcpServers", None,
                             "paste into %APPDATA%/Claude/claude_desktop_config.json, then restart the app"),
}


def _fill(node, values: dict[str, str]):
    if isinstance(node, str):
        for k, v in values.items():
            node = node.replace(k, v)
        return node
    if isinstance(node, list):
        return [_fill(x, values) for x in node]
    if isinstance(node, dict):
        return {k: _fill(v, values) for k, v in node.items()}
    return node


def render(client: str, python: str | Path | None = None, root: str | Path | None = None) -> dict:
    """The client's config with absolute forward-slash paths (valid on Windows and safe inside JSON)."""
    if client not in CLIENTS:
        raise KeyError(f"unknown client {client!r}; choose from {', '.join(CLIENTS)}")
    values = {"{{PYTHON}}": Path(python or sys.executable).resolve().as_posix(),
              "{{SCM_ROOT}}": Path(root or REPO).resolve().as_posix()}
    config = _fill(json.loads((TEMPLATES / f"{client}.json").read_text(encoding="utf-8")), values)
    left = [p for p in PLACEHOLDERS if p in json.dumps(config)]
    if left:
        raise ValueError(f"{client}: unfilled placeholders {left}")
    return config


def server_entry(config: dict) -> dict:
    """The scm-causal-mi entry, whichever top-level key the client uses."""
    for key in ("mcpServers", "servers"):
        if SERVER_NAME in config.get(key, {}):
            return config[key][SERVER_NAME]
    raise KeyError(f"no {SERVER_NAME!r} entry under mcpServers or servers")


def check_interpreter(python: str | Path) -> str | None:
    """None if the interpreter can import the server's dependencies, else the error text."""
    marker = "scm-mcp-deps-ok"
    try:
        r = subprocess.run([str(python), "-X", "utf8", "-c", f"import mcp, duckdb, pydantic; print('{marker}')"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"{type(e).__name__}: {e}"
    if r.returncode == 0 and marker in r.stdout:
        return None
    return (r.stderr.strip().splitlines() or ["not a Python interpreter with the dependencies"])[-1]


def write_rendered(python=None, root=None, out_dir: Path = RENDERED) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in CLIENTS:
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(render(name, python, root), indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def install(client: str, python=None, root=None, target: Path | None = None) -> Path:
    """Merge the entry into the client's config file. Other servers and keys are kept; the old file -> .bak."""
    spec = CLIENTS[client]
    if target is None:
        if spec.target is None:
            raise ValueError(f"{spec.label} keeps its config outside the repo: use --print {client} and "
                             f"{spec.note}")
        target = Path(root or REPO) / spec.target
    target = Path(target)
    existing: dict = {}
    if target.exists():
        text = target.read_text(encoding="utf-8")
        try:
            existing = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as e:   # e.g. VS Code JSONC with comments: do not rewrite what we cannot parse
            raise ValueError(f"{target} is not plain JSON ({e}); paste the entry by hand "
                             f"(--print {client})") from e
        if not isinstance(existing, dict):
            raise ValueError(f"{target} does not hold a JSON object")
        target.with_name(target.name + ".bak").write_text(text, encoding="utf-8")
    servers = existing.setdefault(spec.key, {})
    if not isinstance(servers, dict):
        raise ValueError(f"{target}: '{spec.key}' is not an object")
    servers[SERVER_NAME] = server_entry(render(client, python, root))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", help="interpreter with mcp + duckdb (default: this one)")
    ap.add_argument("--root", help="SCM repository root (default: this checkout)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--print", dest="show", choices=CLIENTS, help="print one rendered config")
    group.add_argument("--install", choices=CLIENTS, help="merge the entry into the client's config file")
    ap.add_argument("--target", help="with --install: write this file instead of the repo default "
                                     "(e.g. a global ~/.cursor/mcp.json or claude_desktop_config.json)")
    args = ap.parse_args(argv)
    if args.target and not args.install:
        ap.error("--target needs --install")
    python = Path(args.python or sys.executable)

    if args.show:
        print(json.dumps(render(args.show, python, args.root), indent=2))
        return 0
    problem = check_interpreter(python)
    if problem:
        print(f"warning: {python.as_posix()} cannot import the server's dependencies: {problem}\n"
              f"         pass --python <interpreter with mcp and duckdb>", file=sys.stderr)
    if args.install:
        try:
            path = install(args.install, python, args.root,
                           Path(args.target).expanduser() if args.target else None)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        spec = CLIENTS[args.install]
        hint = f"restart or reload {spec.label} to pick it up" if args.target else spec.note
        print(f"{spec.label}: wrote {path.as_posix()} ({hint})")
        return 0
    for path in write_rendered(python, args.root):
        spec = CLIENTS[path.stem]
        where = spec.target or "(outside the repo)"
        print(f"{spec.label:<26} {path.relative_to(REPO).as_posix():<45} -> {where}")
    print("\nInstall one with --install <client>; check them all with: python -X utf8 local_mcp/smoke.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
