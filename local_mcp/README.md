# Local MCP server (`scm-causal-mi`)

Local stdio MCP server over the causal-MI study results. Design and roadmap: [DESIGN.md](DESIGN.md).
**Status: M1 complete** — scaffold, manifest/registry, DuckDB sandbox, network guard, audit log,
response envelope, and the `server_status` tool. Study tools arrive in M2–M3.

All commands run from the repo root with the project interpreter
(`C:/Users/Acer/Desktop/Capstone/Capstone/cuda_env/Scripts/python.exe`, shown as `python`).

| Task | Command |
|---|---|
| Rebuild the manifest after re-running any roadmap phase | `python -X utf8 local_mcp/build_manifest.py` |
| Run the tests (29) | `python -X utf8 -m unittest discover -s local_mcp/tests -t . -v` |
| Start the server by hand (it waits for a client on stdin) | `python -X utf8 local_mcp/server.py` |
| Register with Claude Code (local scope, not committed) | `claude mcp add scm-causal-mi -- "<python>" -X utf8 "<repo>/local_mcp/server.py"` |

Then ask the client: *"Call server_status."*

## Files

| File | Role |
|---|---|
| `server.py` | Entry point: installs the network guard, builds the registry and sandbox, registers tools, runs stdio |
| `manifest.json` | Committed list of every readable file: tier, kind, DuckDB view, SHA-256 (CRLF-normalised for text), size |
| `build_manifest.py` | Regenerates the manifest; refuses raw MIMIC paths and anything outside the repo |
| `registry.py` | Tier detection, integrity (hash; stat fast path for large T1 files), freshness, templates |
| `db.py` | DuckDB sandbox: in-memory, allowlist, no network or extensions, locked config, live self-test |
| `guards.py` | In-process outbound-network guard (loopback allowed) |
| `audit.py` | Hash-chained JSONL audit log in `data/local_mcp/` (git-ignored); safe with several server processes |
| `envelope.py` | Response envelope and structured `ToolError`s |
| `runtime.py` | Server context and the audited-call wrapper |
| `sql/` | Reviewed query templates (hashed in the manifest) |
| `tools/status.py` | `server_status` |
| `tests/` | Sandbox (design table 3.4), registry, guard, audit, stdio end-to-end and stdout purity |

## Data access

The server reads only the files listed in `manifest.json`. Patient-level files (`data/roadmap/phase3/*.parquet`)
work only on a machine that holds them; without them the server still starts and reports T1 as absent.
Raw MIMIC files can never be registered.
