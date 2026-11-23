# Local MCP server (`scm-causal-mi`)

Local stdio MCP server over the causal-MI study results. Design and roadmap: [DESIGN.md](DESIGN.md).
**Status: M6 complete (all milestones).** The server has 12 tools, and every response passes the egress guard:

- `study_overview`, `describe_variable`
- `get_risk_factor_effect`, `get_ecg_effect`, `get_subgroup_effects`, `robustness_card`
- `confounding_what_if`, `cohort_summary`, `explain_ecg`
- `render_figure`, `trace_claim`, `server_status`

The same server runs from config templates for Claude Code, Cursor, GitHub Copilot (VS Code), Antigravity and
Claude Desktop. In real Claude Code it answered all 15 golden questions correctly and refused all 8 red-team
prompts ([eval/RESULTS.md](eval/RESULTS.md)).

All commands run from the repo root with the project interpreter
(`C:/Users/Acer/Desktop/Capstone/Capstone/cuda_env/Scripts/python.exe`, shown as `python`).

| Task | Command |
|---|---|
| Regenerate assets (dictionary, profile model ~1.5 min, figure PNGs) | `python -X utf8 local_mcp/export_assets.py` (or `--only dictionary\|profile\|figures`) |
| Rebuild the manifest after re-running any roadmap phase or editing SQL | `python -X utf8 local_mcp/build_manifest.py` |
| Run the tests (98) | `python -X utf8 -m unittest discover -s local_mcp/tests -t . -v` |
| Start the server by hand (it waits for a client on stdin) | `python -X utf8 local_mcp/server.py` |
| Render every client config for this machine (into git-ignored `configs/rendered/`) | `python -X utf8 local_mcp/configs/render.py` |
| Smoke-test every config: launch as the client would, call all 12 tools | `python -X utf8 local_mcp/smoke.py` (`-v` per call, `--quick`, `--config <installed file>`) |
| Evaluate in real Claude Code: 15 golden questions + 8 red-team prompts, graded, summary in `eval/RESULTS.md` (uses your Claude login; about 7 cents per item) | `python -X utf8 local_mcp/eval/run_claude.py` (`--only G03 R03`, `--model sonnet`, `--dry-run`) |
| Run the extended set: 90 questions in 7 categories ([eval/QUESTIONS.md](eval/QUESTIONS.md)), summary in `eval/RESULTS_EXTENDED.md` (about $11) | `python -X utf8 local_mcp/eval/run_claude.py --set extended` |
| Regenerate `eval/QUESTIONS.md` after editing `eval/extended.json` | `python -X utf8 local_mcp/eval/questions_md.py` |
| Register with Claude Code (local scope, not committed) | `claude mcp add scm-causal-mi -- "<python>" -X utf8 "<repo>/local_mcp/server.py"` |

### Connect a client

Run `render.py --install <client>` with the project interpreter. It merges the `scm-causal-mi` entry into the
client's file, keeps any other servers, and saves the previous file as `.bak`. Then reload the client.

| Client | Command | Writes | Then |
|---|---|---|---|
| Claude Code | `--install claude_code` | `.mcp.json` | Start `claude` in the repo, approve the project server, check with `/mcp` |
| Cursor | `--install cursor` | `.cursor/mcp.json` | Settings → MCP: enable `scm-causal-mi` |
| GitHub Copilot (VS Code) | `--install vscode` | `.vscode/mcp.json` | Start it from the file's CodeLens or *MCP: List Servers*; use Agent mode |
| Antigravity | `--install antigravity` | `.agents/mcp_config.json` | Reload the agent's MCP servers |
| Claude Desktop | `--install claude_desktop --target "%APPDATA%/Claude/claude_desktop_config.json"` | that file | Restart the app |

Use `--print <client>` to paste the entry by hand instead (for example into a global file, or a VS Code file
that has comments), and `--python <path>` to pin a different interpreter. Every written file holds absolute
paths for this machine and is git-ignored.

The server only controls what its own tools return. Your client's built-in file and shell tools can still read
`data/` if you allow them. To keep patient-level files out of the chat, deny those reads. In Claude Code, add
`"permissions": {"deny": ["Read(./data/**)"]}` to `.claude/settings.local.json`.

Try asking the client: *"Give me the study overview"*, *"Is the hypertension result robust?"*,
*"Could smoking explain it?"*, *"What does MI do to lateral ST?"*, *"Show the ST/T figure"*,
*"Explain a synthetic ECG for a 65-year-old diabetic man"*, *"How many admissions by age band and MI?"*,
*"Where does claim C06 come from?"*

## Files

| File | Role |
|---|---|
| `server.py` | Entry point: network guard, registry, sandbox, tool registration, stdio |
| `manifest.json` / `build_manifest.py` | Every readable file with tier, kind, DuckDB relation and SHA-256 (CRLF-normalised for text); raw MIMIC refused |
| `export_assets.py` → `assets/` | `variables.csv` (193-variable dictionary), `profile_model.json` (with-MI model + 200 bootstrap draws, training split), `figures/*.png` |
| `registry.py` | Tiers, integrity, verified text cache, freshness of lazy Parquet views |
| `db.py` | DuckDB sandbox: allowlist, no network/extensions, locked config, CSV snapshots, derived views, self-test |
| `sql/` | Reviewed templates: `views__*` (macros and derived views) and one file per query |
| `tools/common.py` | Envelope builder, server-authored assumptions/caveats, name matching, E-values |
| `tools/robustness.py` | Robustness cards and verdict rules shared by all results tools |
| `tools/orientation.py`, `results.py`, `computation.py`, `figures.py`, `provenance.py`, `status.py` | The 12 tools |
| `tools/ecg_explainer.py` | Python port of the Phase 9 browser explainer (parity-tested through Node) |
| `egress.py` | Egress guard: blocks identifiers, MIMIC shifted dates, denylisted keys, long lists and payloads > 32 KB |
| `guards.py`, `audit.py`, `envelope.py`, `runtime.py` | Network guard, hash-chained audit log, response envelope, audited wrapper (applies the egress guard) |
| `configs/templates/`, `configs/render.py` | Client config templates (placeholders only) and the renderer/installer |
| `smoke.py` | Launches the server from each client config (config env over a minimal environment, neutral cwd) and calls all 12 tools |
| `eval/golden.json`, `eval/redteam.json` | 15 golden questions (expected tools, values checked against the live server) and 8 red-team prompts |
| `eval/extended.json` → `eval/QUESTIONS.md` | 90 end-user questions in 7 categories (factual, analytical, diagnostic, predictive, interventional, counterfactual, edge cases), with rubrics and checks; the markdown is generated by `eval/questions_md.py` |
| `eval/grading.py`, `eval/run_claude.py`, `eval/RESULTS.md` | Grader (tools called, values and required phrases in the answer, leaks), headless Claude Code runner, latest results |
| `tests/` | Sandbox, registry, guard, audit, results-vs-source-CSV checks, latency, stdio end-to-end, client configs |

## Data access

The server reads only files listed in `manifest.json`. Patient-level files (`data/roadmap/phase3/*.parquet`)
work only on a machine that holds them; without them the server still starts and reports T1 as absent.
Raw MIMIC files can never be registered.
