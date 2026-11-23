# Local MCP server (`scm-causal-mi`)

Local stdio MCP server over the causal-MI study results. Design and roadmap: [DESIGN.md](DESIGN.md).
**Status: M4 complete** — all 12 tools: `study_overview`, `describe_variable`, `get_risk_factor_effect`,
`get_ecg_effect`, `get_subgroup_effects`, `robustness_card`, `confounding_what_if`, `cohort_summary`,
`explain_ecg`, `render_figure`, `trace_claim`, `server_status`; every response passes the egress guard.
Next: M5 client configs, M6 demo evaluation.

All commands run from the repo root with the project interpreter
(`C:/Users/Acer/Desktop/Capstone/Capstone/cuda_env/Scripts/python.exe`, shown as `python`).

| Task | Command |
|---|---|
| Regenerate assets (dictionary, profile model ~1.5 min, figure PNGs) | `python -X utf8 local_mcp/export_assets.py` (or `--only dictionary\|profile\|figures`) |
| Rebuild the manifest after re-running any roadmap phase or editing SQL | `python -X utf8 local_mcp/build_manifest.py` |
| Run the tests (66) | `python -X utf8 -m unittest discover -s local_mcp/tests -t . -v` |
| Start the server by hand (it waits for a client on stdin) | `python -X utf8 local_mcp/server.py` |
| Register with Claude Code (local scope, not committed) | `claude mcp add scm-causal-mi -- "<python>" -X utf8 "<repo>/local_mcp/server.py"` |

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
| `tests/` | Sandbox, registry, guard, audit, results-vs-source-CSV checks, latency, stdio end-to-end |

## Data access

The server reads only files listed in `manifest.json`. Patient-level files (`data/roadmap/phase3/*.parquet`)
work only on a machine that holds them; without them the server still starts and reports T1 as absent.
Raw MIMIC files can never be registered.
