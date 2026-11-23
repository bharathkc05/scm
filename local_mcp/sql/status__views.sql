-- Views exposed by the DuckDB sandbox (used by server_status). No parameters.
SELECT view_name
FROM duckdb_views()
WHERE NOT internal
ORDER BY view_name
