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
        # Names are validated by the registry and paths come from the manifest, never from tool input.
        # Small CSV result tables are snapshotted into memory right after their hashes were verified (fast, and
        # exactly the verified content); large Parquet files stay as lazy views so only queried columns are read.
        self.views: list[str] = []
        for view, path, reader in registry.views():
            kind = "TABLE" if reader == "read_csv_auto" else "VIEW"
            con.execute(f"CREATE {kind} {view} AS SELECT * FROM {reader}({sql_str(path)})")
            self.views.append(view)
        con.execute("SET lock_configuration = true")
        self.con = con
        # Warm-up: plans every view once and records sizes for server_status.
        self.row_counts = {v: int(con.execute(f"SELECT count(*) FROM {v}").fetchone()[0]) for v in self.views}
        # Derived views and macros from reviewed templates (views__*.sql), in name order. A template whose base
        # views are absent (e.g. a tier missing on this machine) is skipped and reported, not fatal.
        self.derived: list[str] = []
        self.derived_errors: dict[str, str] = {}
        for name in registry.template_names("views__"):
            try:
                con.execute(registry.template(name))
                self.derived.append(name)
            except duckdb.Error as e:
                self.derived_errors[name] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        # Relations are fixed after startup; cache the set for has().
        self.relations = {r[0] for r in con.execute(
            "SELECT view_name FROM duckdb_views() WHERE NOT internal "
            "UNION ALL SELECT table_name FROM duckdb_tables()").fetchall()}

    def has(self, *relations: str) -> bool:
        return all(r in self.relations for r in relations)

    def query(self, template: str, params: tuple | list = (), fragments: dict[str, str] | None = None) -> list[dict]:
        """Run a reviewed template from local_mcp/sql with typed, bound parameters.

        `fragments` fill {placeholders} in templates that need variable SQL structure (e.g. GROUP BY columns);
        callers must build them only from fixed whitelists, never from tool input.
        """
        self.registry.assert_fresh()
        sql = self.registry.template(template)
        if fragments:
            sql = sql.format(**fragments)
        cur = self.con.execute(sql, list(params))
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
