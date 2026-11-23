"""DuckDB sandbox: in-memory, reads only manifest-allowlisted files, no network, no extensions, locked config.

Order matters (DuckDB 1.4): allowed_paths can only be set after the database starts, and must be set before
enable_external_access is switched off; lock_configuration then makes all of it immutable.
"""
import duckdb

from local_mcp.paths import MANIFEST, STATE_DIR
from local_mcp.registry import Registry


def sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class Sandbox:
    def __init__(self, registry: Registry, threads: int = 4, memory_limit: str = "2GB"):
        self.registry = registry
        con = duckdb.connect(":memory:", config={
            "autoinstall_known_extensions": False,   # never download extensions
            "autoload_known_extensions": False,
            "threads": threads,
            "memory_limit": memory_limit,
        })
        self.allowlist = registry.allowlist()
        if self.allowlist:
            con.execute(f"SET allowed_paths = [{', '.join(sql_str(p) for p in self.allowlist)}]")
        con.execute("SET enable_external_access = false")
        self.views: list[str] = []
        for view, path, reader in registry.views():      # names validated by the registry; paths from manifest
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM {reader}({sql_str(path)})")
            self.views.append(view)
        con.execute("SET lock_configuration = true")
        self.con = con
        # Warm-up: plans every view once and records sizes for server_status.
        self.row_counts = {v: int(con.execute(f"SELECT count(*) FROM {v}").fetchone()[0]) for v in self.views}

    def query(self, template: str, params: tuple | list = ()) -> list[dict]:
        """Run a reviewed template from local_mcp/sql with typed, bound parameters."""
        self.registry.assert_fresh()
        cur = self.con.execute(self.registry.template(template), list(params))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def settings(self) -> dict[str, bool]:
        ext, locked = self.con.execute(
            "SELECT current_setting('enable_external_access'), current_setting('lock_configuration')").fetchone()
        return {"external_access": bool(ext), "config_locked": bool(locked)}

    def self_test(self) -> dict[str, bool]:
        """Live probes: each must be refused by DuckDB itself. True = blocked as expected."""
        probe_file = (STATE_DIR / "sandbox_probe.csv").as_posix()
        probes = {
            "blocks_non_allowlisted_file": (f"SELECT * FROM read_csv_auto({sql_str(MANIFEST.resolve().as_posix())})",
                                            duckdb.PermissionException),
            "blocks_network": ("SELECT * FROM read_csv_auto('https://localhost/probe.csv')", duckdb.PermissionException),
            "blocks_extension_install": ("INSTALL httpfs", duckdb.PermissionException),
            "blocks_file_write": (f"COPY (SELECT 1 AS x) TO {sql_str(probe_file)}", duckdb.PermissionException),
            "blocks_config_change": ("SET enable_external_access = true", duckdb.InvalidInputException),
        }
        out = {}
        for name, (sql, expected) in probes.items():
            try:
                self.con.execute(sql)
                out[name] = False
            except expected:
                out[name] = True
            except duckdb.Error:
                out[name] = False
        if (STATE_DIR / "sandbox_probe.csv").exists():   # only possible if the write probe failed open
            (STATE_DIR / "sandbox_probe.csv").unlink()
        return out
