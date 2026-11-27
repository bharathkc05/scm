"""M6, real-client layer: ask Claude Code (headless) the golden questions and red-team prompts with only this MCP
server available, then grade the tools it called, the values in its answers, and leaks.

    python -X utf8 local_mcp/eval/run_claude.py                      # core: 15 golden + 8 red-team -> RESULTS.md
    python -X utf8 local_mcp/eval/run_claude.py --set extended       # 90 questions, 7 categories -> RESULTS_EXTENDED.md
    python -X utf8 local_mcp/eval/run_claude.py --only G03 R03 P06   # some items from any set (no results file)
    python -X utf8 local_mcp/eval/run_claude.py --model sonnet --jobs 3
    python -X utf8 local_mcp/eval/run_claude.py --dry-run            # print the command, run nothing

Each item runs `claude -p` in an empty temporary folder (no CLAUDE.md, no project .mcp.json) with
`--strict-mcp-config` (only this server), `--tools ""` (no built-in file, shell or web tools, so the MCP tools are
the only way to the data) and `--permission-mode dontAsk` with only `mcp__scm-causal-mi` allowed. Questions are
sent as-is: no extra system prompt, so the server's own instructions and tool descriptions are what is tested.

This uses the Claude account that `claude` is logged in to; tool outputs (aggregates) go to the model provider,
as in normal use. Transcripts go to data/local_mcp/eval/<run>/ (git-ignored); the summary to eval/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)                                     # do not let eval/ shadow anything

from local_mcp import audit  # noqa: E402
from local_mcp.configs import render  # noqa: E402
from local_mcp.eval import grading  # noqa: E402
from local_mcp.paths import AUDIT_LOG, ROOT, STATE_DIR  # noqa: E402

RESULTS_MD = Path(__file__).resolve().parent / "RESULTS.md"
RESULTS_EXTENDED_MD = Path(__file__).resolve().parent / "RESULTS_EXTENDED.md"
SETS = {"core": ("golden", "redteam"), "extended": ("extended",), "all": ("golden", "redteam", "extended")}


def load_items(kinds) -> list[dict]:
    """Items tagged with the set they come from (golden and extended are graded alike; red-team for leaks)."""
    source = {"golden": grading.golden, "redteam": grading.redteam, "extended": grading.extended}
    return [dict(item, _kind=kind) for kind in kinds for item in source[kind]()]


def claude_command(prompt: str, mcp_config: Path, model: str | None, max_turns: int, budget: float) -> list[str]:
    exe = shutil.which("claude") or "claude"
    cmd = [exe, "-p", prompt, "--output-format", "stream-json", "--verbose",
           "--mcp-config", str(mcp_config), "--strict-mcp-config",
           "--tools", "", "--allowedTools", f"mcp__{grading.SERVER}", "--permission-mode", "dontAsk",
           "--no-session-persistence", "--max-turns", str(max_turns), "--max-budget-usd", f"{budget:g}"]
    return cmd + (["--model", model] if model else [])


def run_item(item: dict, out_dir: Path, args) -> dict:
    kind = item["_kind"]
    prompt = item.get("question") or item["prompt"]
    with tempfile.TemporaryDirectory(prefix="scm-mcp-eval-") as tmp:
        cfg = Path(tmp) / "mcp.json"
        cfg.write_text(json.dumps(render.render("claude_code")), encoding="utf-8")
        cmd = claude_command(prompt, cfg, args.model, args.max_turns, args.max_budget_usd)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=args.timeout, stdin=subprocess.DEVNULL)
            stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, rc = f"timeout after {args.timeout}s", -1
    wall = time.perf_counter() - t0
    (out_dir / f"{item['id']}.jsonl").write_text(stdout, encoding="utf-8")
    if stderr.strip():
        (out_dir / f"{item['id']}.stderr.txt").write_text(stderr, encoding="utf-8")
    t = grading.parse_stream(stdout.splitlines())
    grade = grading.grade_redteam(item, t) if kind == "redteam" else grading.grade_golden(item, t)
    if rc != 0 and not t.answer:
        grade["passed"] = False
        grade.setdefault("problems", []).append(f"claude exited {rc}: {stderr.strip()[-300:]}")
    grade.update(kind=kind, prompt=prompt, model=t.model, mcp_status=t.mcp_status, turns=t.turns,
                 cost_usd=round(t.cost_usd, 4), seconds=round(wall, 1), subtype=t.subtype, answer=t.answer,
                 category=item.get("category", grade.get("category")), rubric=item.get("rubric", ""))
    mark = "PASS" if grade["passed"] else "FAIL"
    print(f"[{mark}] {item['id']} {wall:5.0f}s  tools={grade['tools_called']}", flush=True)
    return grade


def _cell(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= n else text[: n - 1] + "…"


def write_markdown(results: list[dict], meta: dict, path: Path) -> None:
    gold = [r for r in results if r["kind"] == "golden"]
    red = [r for r in results if r["kind"] == "redteam"]
    lines = [
        "# M6 evaluation in Claude Code",
        "",
        f"Run {meta['run']} with {meta['claude_version']}, model `{meta['model']}`. Each item was one headless "
        "`claude -p` session in an empty folder, with only the `scm-causal-mi` server, no built-in tools and no extra "
        "system prompt. Graded by `local_mcp/eval/grading.py`; transcripts are in "
        f"`data/local_mcp/eval/{meta['run']}/` (not committed).",
        "",
        f"- **Golden questions:** {sum(r['passed'] for r in gold)}/{len(gold)} passed (right tools called, expected "
        "values in the answer, nothing unsafe in any tool output or the answer).",
        f"- **Red-team prompts:** {sum(r['passed'] for r in red)}/{len(red)} passed (no identifier, date, patient row, "
        "small cell or probability reached the client or the answer).",
        f"- **Audit log:** {meta['audit_added']} tool calls recorded during the run; hash chain "
        f"{'verifies' if meta['audit_ok'] else 'BROKEN'} ({meta['audit_entries']} entries in total).",
        f"- **Cost and time:** ${meta['cost']:.2f} as reported by Claude Code; {meta['minutes']:.1f} min wall clock "
        f"with {meta['jobs']} parallel sessions.",
        "",
        *([f"**Note.** {meta['note']}", ""] if meta.get("note") else []),
    ]
    if gold:
        lines += ["## Golden questions", "",
                  "| ID | Question | Tools called | Values found in the answer | Result |", "|---|---|---|---|---|"]
        for r in gold:
            found = ", ".join(v["found"] or f"~~{v['expected']}~~" for v in r["values"]) or "(figure shown)"
            why = _why(r)
            lines.append(f"| {r['id']} | {_cell(r['prompt'], 90)} | {', '.join(r['tools_called']) or '-'} | {found} | "
                         f"{'pass' if r['passed'] else 'FAIL: ' + _cell(why, 120)} |")
        lines.append("")
    if red:
        lines += ["## Red-team prompts", "",
                  "| ID | Asks for | Tools called (refusals) | Answer (start) | Result |", "|---|---|---|---|---|"]
        for r in red:
            calls = ", ".join(r["tools_called"]) or "none"
            if r["refusals"]:
                calls += f" ({', '.join(r['refusals'])})"
            res = "pass" if r["passed"] else "FAIL: " + _cell("; ".join(r["safety"] + r.get("problems", [])), 120)
            if r["flags"]:
                res += " (" + "; ".join(r["flags"]) + ")"
            lines.append(f"| {r['id']} | {r['category'].replace('_', ' ')} | {calls} | {_cell(r['answer'], 160)} | {res} |")
        no_tools = sum(not r["tools_called"] for r in red)
        lines += ["", f"In {no_tools} of {len(red)} red-team runs the model declined before calling any tool, from the "
                  "server's instructions and tool schemas alone. That shows the client behaving, not the server "
                  "refusing. The server side is tested separately, without an LLM, by `tests/test_eval.py`. It "
                  "sends the same requests as direct tool calls from a worst-case client: identifier dimensions, "
                  "SQL and path injection, out-of-range inputs, and small-cell cross-tabs. Each call must come back "
                  "as a structured refusal or as aggregates that pass every safety rule.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _why(r: dict) -> str:
    return "; ".join(r.get("missing_tools", []) + r.get("missing_values", []) + r.get("missing_mentions", [])
                     + r["safety"] + r.get("problems", []))


def write_extended_markdown(results: list[dict], meta: dict, path: Path) -> None:
    cats = grading.categories()
    rows = [r for r in results if r["kind"] == "extended"]
    lines = [
        "# Extended evaluation in Claude Code",
        "",
        f"Run {meta['run']} with {meta['claude_version']}, model `{meta['model']}`. The questions and rubrics are in "
        "[QUESTIONS.md](QUESTIONS.md). Each question ran in its own headless `claude -p` session in an empty folder. "
        "Only the `scm-causal-mi` server was attached, with no built-in tools and no extra system prompt. "
        f"Transcripts are in `data/local_mcp/eval/{meta['run']}/` (not committed).",
        "",
        f"**{sum(r['passed'] for r in rows)}/{len(rows)} passed** automatic grading: the expected tools were called, "
        "the expected values and required phrases appear in the answer, and nothing unsafe leaked. Phrase checks are "
        "deliberately lenient, so read each failure against its rubric before counting it against the client.",
        "",
        f"Cost ${meta['cost']:.2f} as reported by Claude Code; {meta['minutes']:.1f} min with {meta['jobs']} parallel "
        f"sessions. The audit hash chain {'verifies' if meta['audit_ok'] else 'is BROKEN'}.",
        "",
        *([f"**Note.** {meta['note']}", ""] if meta.get("note") else []),
        "| Category | Passed |",
        "|---|---|",
    ]
    for key, cat in cats.items():
        part = [r for r in rows if r["category"] == key]
        if part:
            lines.append(f"| {cat['title']} | {sum(r['passed'] for r in part)}/{len(part)} |")
    for key, cat in cats.items():
        part = [r for r in rows if r["category"] == key]
        if not part:
            continue
        lines += ["", f"## {cat['title']}", "",
                  "| ID | Question | Tools called | Answer (start) | Result |", "|---|---|---|---|---|"]
        for r in part:
            res = "pass" if r["passed"] else "FAIL: " + _cell(_why(r), 140)
            if r.get("flags"):
                res += " (" + "; ".join(r["flags"]) + ")"
            lines.append(f"| {r['id']} | {_cell(r['prompt'], 80)} | {', '.join(r['tools_called']) or '-'} | "
                         f"{_cell(r['answer'], 140)} | {res} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(results: list[dict], meta: dict) -> list[Path]:
    written = []
    if any(r["kind"] in ("golden", "redteam") for r in results):
        write_markdown([r for r in results if r["kind"] in ("golden", "redteam")], meta, RESULTS_MD)
        written.append(RESULTS_MD)
    if any(r["kind"] == "extended" for r in results):
        write_extended_markdown(results, meta, RESULTS_EXTENDED_MD)
        written.append(RESULTS_EXTENDED_MD)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--set", choices=SETS, default="core",
                    help="core = 15 golden + 8 red-team (default); extended = 90 questions; all = both")
    ap.add_argument("--only", nargs="+", help="item IDs from any set (G01, R03, F05, ...); writes no results file")
    ap.add_argument("--model", help="Claude Code model alias or name (default: the CLI default)")
    ap.add_argument("--jobs", type=int, default=3, help="parallel sessions (each starts its own server)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-budget-usd", type=float, default=1.0, help="per item")
    ap.add_argument("--timeout", type=int, default=420, help="seconds per item")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-markdown", action="store_true", help="do not rewrite the results file(s)")
    ap.add_argument("--report-only", metavar="RESULTS_JSON", help="rebuild the results file(s) from a saved run")
    args = ap.parse_args(argv)
    if args.report_only:
        saved = json.loads(Path(args.report_only).read_text(encoding="utf-8"))
        for r in saved["results"]:
            r.setdefault("kind", "golden")
        for path in write_reports(saved["results"], saved["meta"]):
            print(f"wrote {path.relative_to(ROOT).as_posix()}")
        return 0

    if args.only:
        wanted = {x.upper() for x in args.only}
        items = [i for i in load_items(SETS["all"]) if i["id"] in wanted]
        unknown = wanted - {i["id"] for i in items}
        if unknown:
            ap.error(f"unknown item IDs: {sorted(unknown)}")
    else:
        items = load_items(SETS[args.set])
    if not items:
        ap.error("no items selected")
    if args.dry_run:
        print(subprocess.list2cmdline(claude_command(items[0].get("question") or items[0]["prompt"],
                                                     Path("<tmp>/mcp.json"), args.model, args.max_turns,
                                                     args.max_budget_usd)))
        return 0
    run = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = STATE_DIR / "eval" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    before = audit.verify(AUDIT_LOG)["entries"]
    version = subprocess.run([shutil.which("claude") or "claude", "--version"], capture_output=True, text=True,
                             timeout=60).stdout.strip()
    print(f"{len(items)} items, {args.jobs} parallel; transcripts -> {out_dir.relative_to(ROOT).as_posix()}", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(lambda it: run_item(it, out_dir, args), items))
    chain = audit.verify(AUDIT_LOG)
    models = sorted({r["model"] for r in results if r["model"]})
    meta = {"run": run, "claude_version": version, "model": ", ".join(models) or args.model or "default",
            "audit_added": chain["entries"] - before, "audit_ok": chain["ok"], "audit_entries": chain["entries"],
            "cost": sum(r["cost_usd"] for r in results), "minutes": (time.perf_counter() - t0) / 60,
            "jobs": args.jobs}
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    if args.only:                                      # a partial run must not overwrite the full results
        print(f"(--only: results file not rewritten; rebuild from {out_dir.relative_to(ROOT).as_posix()}/results.json"
              " with --report-only if wanted)")
    elif not args.no_markdown:
        for path in write_reports(results, meta):
            print(f"wrote {path.relative_to(ROOT).as_posix()}")
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed; audit chain {'ok' if chain['ok'] else 'BROKEN'}; "
          f"${meta['cost']:.2f}; {meta['minutes']:.1f} min")
    for r in results:
        if not r["passed"]:
            print(f"  {r['id']}: {_why(r)}")
        if r.get("flags"):
            print(f"  {r['id']} flags: {r['flags']}")
    return 0 if passed == len(results) and chain["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
