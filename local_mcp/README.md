# Local MCP server (`scm-causal-mi`)

Local stdio MCP server over the causal-MI study results. Design and roadmap: [DESIGN.md](DESIGN.md).
**Status: M2 complete** — 8 tools: `study_overview`, `describe_variable`, `get_risk_factor_effect`,
`get_ecg_effect`, `get_subgroup_effects`, `robustness_card`, `trace_claim`, `server_status`.
Computation tools (`confounding_what_if`, `cohort_summary`, `explain_ecg`, `render_figure`) arrive in M3.

All commands run from the repo root with the project interpreter
(`C:/Users/Acer/Desktop/Capstone/Capstone/cuda_env/Scripts/python.exe`, shown as `python`).

| Task | Command |
|---|---|
| Regenerate assets (variable dictionary) | `python -X utf8 local_mcp/export_assets.py` |
| Rebuild the manifest after re-running any roadmap phase or editing SQL | `python -X utf8 local_mcp/build_manifest.py` |
| Run the tests (46) | `python -X utf8 -m unittest discover -s local_mcp/tests -t . -v` |
| Start the server by hand (it waits for a client on stdin) | `python -X utf8 local_mcp/server.py` |
| Register with Claude Code (local scope, not committed) | `claude mcp add scm-causal-mi -- "<python>" -X utf8 "<repo>/local_mcp/server.py"` |

Try asking the client: *"Give me the study overview"*, *"Is the hypertension result robust?"*,
*"What does MI do to lateral ST?"*, *"Where does claim C06 come from?"*

## Files

| File | Role |
|---|---|
| `server.py` | Entry point: network guard, registry, sandbox, tool registration, stdio |
| `manifest.json` / `build_manifest.py` | Every readable file with tier, kind, DuckDB relation and SHA-256 (CRLF-normalised for text); raw MIMIC refused |
| `export_assets.py` → `assets/variables.csv` | Data dictionary (193 variables) built from `roadmap/definitions.py` and the feature list |
| `registry.py` | Tiers, integrity, verified text cache, freshness of lazy Parquet views |
| `db.py` | DuckDB sandbox: allowlist, no network/extensions, locked config, CSV snapshots, derived views, self-test |
| `sql/` | Reviewed templates: `views__*` (macros and derived views) and one file per query |
| `tools/common.py` | Envelope builder, server-authored assumptions/caveats, name matching, E-values |
| `tools/robustness.py` | Robustness cards and verdict rules shared by all results tools |
| `tools/orientation.py`, `results.py`, `provenance.py`, `status.py` | The 8 tools |
| `guards.py`, `audit.py`, `envelope.py`, `runtime.py` | Network guard, hash-chained audit log, response envelope, audited wrapper |
| `tests/` | Sandbox, registry, guard, audit, results-vs-source-CSV checks, latency, stdio end-to-end |

## Data access

The server reads only files listed in `manifest.json`. Patient-level files (`data/roadmap/phase3/*.parquet`)
work only on a machine that holds them; without them the server still starts and reports T1 as absent.
Raw MIMIC files can never be registered.
