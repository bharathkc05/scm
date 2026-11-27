"""M5 smoke test: launch the server exactly as each MCP client config says, then call all 15 tools over stdio.

    python -X utf8 local_mcp/smoke.py                            # every client template, rendered for this machine
    python -X utf8 local_mcp/smoke.py --client cursor vscode     # some clients
    python -X utf8 local_mcp/smoke.py --config .cursor/mcp.json  # an installed config file
    python -X utf8 local_mcp/smoke.py --quick                    # initialize, list tools and server_status only

Each launch uses the config's command, args and env merged over the MCP SDK's minimal inherited environment
(what clients do, so nothing leaks in from this shell), and the config's `cwd` or else a neutral temporary
folder (so nothing depends on where the client was started). Exit code 0 only if every launch passes.
`cohort_summary` counts as passed with "gated" when the patient-level files are absent (public clone).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _PKG:      # run as a script: import the package, not its modules
    sys.path[0] = str(_PKG.parent)
elif str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from local_mcp import egress  # noqa: E402
from local_mcp.configs import render  # noqa: E402

EXPECTED_TOOLS = ("study_overview", "describe_variable", "get_risk_factor_effect", "get_ecg_effect",
                  "get_subgroup_effects", "robustness_card", "confounding_what_if", "cohort_summary", "explain_ecg",
                  "render_figure", "trace_claim", "server_status", "counterfactual_ecg",
                  "clinical_what_if", "counterfactual_validation")

# One representative call per tool (server_status first, so a broken start is reported before anything else).
SMOKE_CALLS: tuple[tuple[str, dict], ...] = (
    ("server_status", {}),
    ("study_overview", {}),
    ("describe_variable", {"name": "lateral ST"}),
    ("get_risk_factor_effect", {"exposure": "hypertension"}),
    ("get_ecg_effect", {"group": "ST"}),
    ("get_subgroup_effects", {"result_id": "L1:ckd:primary", "modifier": "age"}),
    ("robustness_card", {"result_id": "L2:ST_lateral_mean"}),
    ("confounding_what_if", {"result_id": "L1:hypertension:primary", "rr_confounder_exposure": 2,
                             "rr_confounder_outcome": 2}),
    ("cohort_summary", {"layer": "layer2_candidates", "stratify_by": ["timing_class"],
                        "metrics": ["age_mean", "pct_male"]}),
    ("explain_ecg", {"patient": {"age": 65, "sex": "male", "diabetes": True},
                     "example_id": "synthetic_typical_mi_shift", "timing": "after_onset"}),
    ("render_figure", {"figure_id": "st_t_effects"}),
    ("trace_claim", {"claim_id": "C06"}),
    ("counterfactual_ecg", {"patient": {"age": 65, "sex": "male", "hypertension": True}, "direction": "remove_mi",
                            "mi_type": "nstemi", "example_id": "synthetic_expected"}),
    ("clinical_what_if", {"patient": {"age": 65, "sex": "male", "hypertension": True}, "exposure": "hypertension",
                          "had_mi": True}),
    ("counterfactual_validation", {}),
)
QUICK_CALLS = SMOKE_CALLS[:1]
GATED = {"cohort_summary": "DATA_NOT_PRESENT"}           # possession-gated: allowed to be absent on a public clone
NON_DETERMINISTIC = {"server_status"}                     # uptime, call counters: excluded from the fingerprint


@dataclass
class Call:
    tool: str
    status: str               # ok | gated | FAIL
    ms: float
    bytes: int = 0
    detail: str = ""
    fingerprint: str = ""


@dataclass
class Launch:
    name: str
    command: str
    cwd: str
    ok: bool = False
    start_ms: float = 0.0     # spawn -> initialize response (cold start as the client sees it)
    total_ms: float = 0.0
    tools: list[str] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    stderr_tail: str = ""

    @property
    def fingerprint(self) -> str:
        """Hash of every deterministic result: identical across clients means the same server answered."""
        h = hashlib.sha256()
        for c in self.calls:
            h.update(f"{c.tool}:{c.status}:{c.fingerprint}".encode())
        return h.hexdigest()[:12]


def params_from_entry(entry: dict, neutral_cwd: Path) -> StdioServerParameters:
    """What a client does with a config entry. Clients that need `type` accept only stdio here."""
    if entry.get("type", "stdio") != "stdio":
        raise ValueError(f"unsupported transport type {entry.get('type')!r}")
    return StdioServerParameters(command=entry["command"], args=list(entry.get("args", [])),
                                 env=dict(entry.get("env", {})), cwd=entry.get("cwd") or str(neutral_cwd))


def _error_body(result) -> dict:
    text = result.content[0].text if result.content else ""
    try:
        return json.loads(text[text.index("{"):]).get("error", {})
    except ValueError:
        return {"code": "UNPARSEABLE", "message": text[:200]}


def _check_call(tool: str, result) -> Call:
    if result.isError:
        err = _error_body(result)
        if GATED.get(tool) == err.get("code"):
            return Call(tool, "gated", 0, detail=err.get("message", "")[:80])
        return Call(tool, "FAIL", 0, detail=f"{err.get('code')}: {err.get('message', '')[:120]}")
    body = result.structuredContent
    if not isinstance(body, dict) or not body.get("ok") or body.get("tool") != tool:
        return Call(tool, "FAIL", 0, detail="missing or malformed structuredContent envelope")
    leaks = egress.check(body)                            # client-side re-check of the server's own guard
    if leaks:
        return Call(tool, "FAIL", 0, detail=f"egress: {sorted({v['code'] for v in leaks})}")
    if tool == "render_figure" and [c.type for c in result.content] != ["image", "text"]:
        return Call(tool, "FAIL", 0, detail="expected image + text content")
    if tool != "render_figure" and not result.content:
        return Call(tool, "FAIL", 0, detail="no text fallback for clients without structuredContent")
    size = len(json.dumps(body, separators=(",", ":")).encode())
    fp = "" if tool in NON_DETERMINISTIC else hashlib.sha256(
        json.dumps(body["result"], sort_keys=True).encode()).hexdigest()[:12]
    return Call(tool, "ok", 0, bytes=size, detail=body.get("summary", "")[:90], fingerprint=fp)


def _check_status(body: dict) -> list[str]:
    r = body["result"]
    checks = {"transport is stdio": r.get("transport") == "stdio",
              "network guard active": r.get("network_guard", {}).get("active") is True,
              "DuckDB config locked": r.get("duckdb_sandbox", {}).get("config_locked") is True,
              "DuckDB external access off": r.get("duckdb_sandbox", {}).get("external_access") is False,
              "sandbox self-test passes": all(r.get("duckdb_sandbox", {}).get("self_test", {"x": False}).values()),
              "raw MIMIC not registered": r.get("raw_mimic_registered") is False,
              "no integrity mismatches": r.get("integrity", {}).get("mismatched") == [],
              "egress guard active": r.get("egress_guard", {}).get("active") is True}
    return [f"server_status: {name} failed" for name, ok in checks.items() if not ok]


async def _session(params: StdioServerParameters, calls, launch: Launch, errlog) -> None:
    t0 = time.perf_counter()
    async with stdio_client(params, errlog=errlog) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            launch.start_ms = (time.perf_counter() - t0) * 1000
            if init.serverInfo.name != render.SERVER_NAME:
                launch.problems.append(f"server name {init.serverInfo.name!r}")
            launch.tools = sorted(t.name for t in (await s.list_tools()).tools)
            if launch.tools != sorted(EXPECTED_TOOLS):
                launch.problems.append(f"tools listed {launch.tools}")
            for tool, args in calls:
                t1 = time.perf_counter()
                result = await s.call_tool(tool, args)
                call = _check_call(tool, result)
                call.ms = (time.perf_counter() - t1) * 1000
                if tool == "server_status" and call.status == "ok":
                    launch.problems += _check_status(result.structuredContent)
                launch.calls.append(call)
    launch.total_ms = (time.perf_counter() - t0) * 1000


def run_entry(name: str, entry: dict, calls=SMOKE_CALLS, timeout: float = 180) -> Launch:
    """Launch one config entry and run the calls. Never raises: problems are collected on the Launch."""
    with tempfile.TemporaryDirectory(prefix="scm-mcp-smoke-") as tmp:
        params = params_from_entry(entry, Path(tmp))
        launch = Launch(name, Path(params.command).name, Path(params.cwd).as_posix())
        errpath = Path(tmp) / "stderr.log"
        with open(errpath, "w", encoding="utf-8") as errlog:
            try:
                asyncio.run(asyncio.wait_for(_session(params, calls, launch, errlog), timeout))
            except Exception as e:  # noqa: BLE001 - a smoke test reports, it does not crash
                launch.problems.append(f"{type(e).__name__}: {e}"[:300])
        launch.stderr_tail = "\n".join(errpath.read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
    failed = [c for c in launch.calls if c.status == "FAIL"]
    launch.problems += [f"{c.tool}: {c.detail}" for c in failed]
    launch.ok = not launch.problems and len(launch.calls) == len(calls)
    return launch


def load_entries(clients=None, config_files=None) -> list[tuple[str, dict]]:
    if config_files:
        return [(Path(f).as_posix(), render.server_entry(json.loads(Path(f).read_text(encoding="utf-8"))))
                for f in config_files]
    return [(c, render.server_entry(render.render(c))) for c in (clients or render.CLIENTS)]


def report(launches: list[Launch], verbose: bool) -> str:
    lines = []
    for L in launches:
        lines.append(f"[{'PASS' if L.ok else 'FAIL'}] {L.name:<16} cold start {L.start_ms:6.0f} ms   "
                     f"{len(L.tools)} tools   {len(L.calls)} calls   total {L.total_ms / 1000:5.1f} s   "
                     f"cwd={L.cwd}")
        if verbose or not L.ok:
            for c in L.calls:
                lines.append(f"    {c.status:<5} {c.tool:<22} {c.ms:7.0f} ms {c.bytes:7d} B  {c.detail}")
        for p in L.problems:
            lines.append(f"    problem: {p}")
        if not L.ok and L.stderr_tail:
            lines.append("    server stderr (tail):\n      " + L.stderr_tail.replace("\n", "\n      "))
    prints = {L.fingerprint for L in launches if L.ok}
    if len(launches) > 1 and all(L.ok for L in launches):
        lines.append("Same answers from every config: " + ("yes" if len(prints) == 1 else f"NO ({len(prints)})"))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--client", nargs="+", choices=render.CLIENTS, help="clients to render and launch")
    src.add_argument("--config", nargs="+", help="installed config file(s) to launch as written")
    ap.add_argument("--quick", action="store_true", help="only initialize, list tools and server_status")
    ap.add_argument("-v", "--verbose", action="store_true", help="show every call")
    args = ap.parse_args(argv)
    calls = QUICK_CALLS if args.quick else SMOKE_CALLS
    launches = [run_entry(name, entry, calls) for name, entry in load_entries(args.client, args.config)]
    print(report(launches, args.verbose))
    same = len({L.fingerprint for L in launches}) == 1
    return 0 if all(L.ok for L in launches) and same else 1


if __name__ == "__main__":
    sys.exit(main())
