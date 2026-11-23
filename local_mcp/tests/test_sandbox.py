"""Reproduces every row of DESIGN.md section 3.4 against the real manifest and data on this machine."""
import unittest

import duckdb

from local_mcp.db import Sandbox, sql_str
from local_mcp.paths import PKG, ROOT, STATE_DIR
from local_mcp.registry import Registry

RAW_MIMIC_DB = (ROOT.parent / "Capstone" / "data" / "mimic_database.duckdb").as_posix()


class SandboxBehaviour(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = Registry()
        if cls.reg.error:
            raise unittest.SkipTest(cls.reg.error)
        cls.sb = Sandbox(cls.reg)

    def blocked(self, sql, exc=duckdb.PermissionException):
        with self.assertRaises(exc):
            self.sb.con.execute(sql)

    # ---- allowed
    def test_grouped_aggregate_over_allowlisted_parquet(self):
        if "layer1" not in self.sb.views:
            self.skipTest("T1 files not on this machine")
        rows = self.sb.con.execute("SELECT acute_mi, count(*) FROM layer1 GROUP BY 1 ORDER BY 1").fetchall()
        self.assertEqual(rows, [(0, 419630), (1, 11222)])

    def test_filter_on_allowlisted_csv(self):
        n = self.sb.con.execute("SELECT count(*) FROM layer1_estimates WHERE estimand = 'ATE_overlap'").fetchone()[0]
        self.assertEqual(n, 7)

    def test_template_runner_with_typed_parameter(self):
        views = [r["view_name"] for r in self.sb.query("status__views")]
        self.assertIn("layer1_estimates", views)
        if "layer1" in self.sb.views:
            n = self.sb.con.execute("SELECT count(*) FROM layer1 WHERE sex_male = ?::TINYINT", [1]).fetchone()[0]
            self.assertGreater(n, 0)

    # ---- blocked
    def test_non_allowlisted_local_file(self):
        self.blocked(f"SELECT * FROM read_csv_auto({sql_str((PKG / 'DESIGN.md').as_posix())})")

    def test_attach_raw_mimic_duckdb(self):
        self.blocked(f"ATTACH {sql_str(RAW_MIMIC_DB)} AS m (READ_ONLY)")

    def test_https_url(self):
        self.blocked("SELECT * FROM read_csv_auto('https://example.com/x.csv')")

    def test_install_and_load_extension(self):
        self.blocked("INSTALL httpfs")
        self.blocked("LOAD httpfs")

    def test_copy_to_file(self):
        target = STATE_DIR / "copy_probe.csv"
        self.blocked(f"COPY (SELECT 1) TO {sql_str(target.as_posix())}")
        self.assertFalse(target.exists())

    def test_config_change_after_lock(self):
        self.blocked("SET enable_external_access = true", duckdb.InvalidInputException)
        self.blocked("SET lock_configuration = false", duckdb.InvalidInputException)

    # ---- state
    def test_settings_and_self_test(self):
        self.assertEqual(self.sb.settings(), {"external_access": False, "config_locked": True})
        self.assertTrue(all(self.sb.self_test().values()), self.sb.self_test())

    def test_raw_mimic_not_in_manifest(self):
        self.assertFalse(self.reg.raw_mimic_registered())
        self.assertFalse(any(RAW_MIMIC_DB.lower() == p.lower() for p in self.sb.allowlist))


if __name__ == "__main__":
    unittest.main()
