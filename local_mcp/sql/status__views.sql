-- Relations exposed by the DuckDB sandbox (used by server_status). No parameters.
-- Lazy views (Parquet, derived) and in-memory snapshot tables (verified CSV results).
SELECT view_name
FROM (
    SELECT view_name FROM duckdb_views() WHERE NOT internal
    UNION ALL
    SELECT table_name FROM duckdb_tables()
)
ORDER BY view_name
