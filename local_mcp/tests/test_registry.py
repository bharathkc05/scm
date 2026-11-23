"""Registry and manifest behaviour on a temporary fake repository (no real data needed)."""
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import duckdb

from local_mcp.build_manifest import build, check_allowed, write
from local_mcp.db import Sandbox
from local_mcp.registry import IntegrityError, Registry

ARTS = [("data/roadmap/phase3/t1.parquet", "T1", "parquet", "t1"),
        ("res/est.csv", "T2", "csv", "est"),
        ("res/params.json", "T3", "json", None)]


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "data/roadmap/phase3").mkdir(parents=True)
        (self.tmp / "res").mkdir()
        (self.tmp / "local_mcp/sql").mkdir(parents=True)
        duckdb.sql(f"COPY (SELECT range AS id, range % 2 AS flag FROM range(100)) "
                   f"TO '{(self.tmp / 'data/roadmap/phase3/t1.parquet').as_posix()}' (FORMAT parquet)")
        (self.tmp / "res/est.csv").write_bytes(b"name,value\na,1\nb,2\n")
        (self.tmp / "res/params.json").write_text('{"k": 1}\n')
        (self.tmp / "local_mcp/sql/t__count.sql").write_text("SELECT count(*) AS n FROM est\n")
        (self.tmp / "local_mcp/sql/t__t1count.sql").write_text("SELECT count(*) AS n FROM t1\n")
        self.manifest = self.tmp / "manifest.json"
        write(build(self.tmp, ARTS, self.tmp / "local_mcp/sql"), self.manifest)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def reg(self):
        return Registry(self.manifest, self.tmp)

    def test_all_present_and_queryable(self):
        r = self.reg()
        self.assertEqual(r.tiers(), {"T1": "present", "T2": "present", "T3": "present", "code": "present"})
        self.assertEqual(sorted(v for v, _, _ in r.views()), ["est", "t1"])
        sb = Sandbox(r)
        self.assertEqual(sb.query("t__count"), [{"n": 2}])
        self.assertEqual(sb.row_counts, {"t1": 100, "est": 2})

    def test_crlf_only_change_is_not_a_mismatch(self):
        (self.tmp / "res/est.csv").write_bytes(b"name,value\r\na,1\r\nb,2\r\n")
        self.assertEqual(self.reg().by_status("mismatch"), [])

    def test_content_change_is_a_mismatch_and_not_allowlisted(self):
        (self.tmp / "res/est.csv").write_bytes(b"name,value\na,1\nb,999\n")
        r = self.reg()
        self.assertEqual(r.by_status("mismatch"), ["res/est.csv"])
        self.assertEqual(r.tiers()["T2"], "absent")
        self.assertNotIn("est", [v for v, _, _ in r.views()])

    def test_missing_t1_is_possession_gated(self):
        os.remove(self.tmp / "data/roadmap/phase3/t1.parquet")
        r = self.reg()
        self.assertEqual(r.tiers()["T1"], "absent")
        sb = Sandbox(r)
        self.assertNotIn("t1", sb.views)
        with self.assertRaises(duckdb.CatalogException):
            sb.con.execute("SELECT * FROM t1")

    def test_touched_t1_with_same_content_falls_back_to_hash(self):
        p = self.tmp / "data/roadmap/phase3/t1.parquet"
        later = time.time() + 100
        os.utime(p, (later, later))
        r = self.reg()
        t1 = [a for a in r.artifacts if a.tier == "T1"][0]
        self.assertEqual((t1.status, t1.check), ("ok", "stat_changed_then_hash"))

    def test_lazy_parquet_change_after_start_blocks_queries(self):
        r = self.reg()
        sb = Sandbox(r)
        self.assertEqual(sb.query("t__t1count"), [{"n": 100}])
        duckdb.sql(f"COPY (SELECT range AS id, 0 AS flag FROM range(5)) "
                   f"TO '{(self.tmp / 'data/roadmap/phase3/t1.parquet').as_posix()}' (FORMAT parquet)")
        with self.assertRaises(IntegrityError):
            sb.query("t__t1count")

    def test_csv_and_json_changes_after_start_keep_serving_the_verified_snapshot(self):
        r = self.reg()
        sb = Sandbox(r)
        (self.tmp / "res/est.csv").write_bytes(b"name,value\na,1\nb,2\nc,3\n")
        (self.tmp / "res/params.json").write_text('{"k": 999}\n')
        self.assertEqual(sb.query("t__count"), [{"n": 2}])              # in-memory snapshot of verified content
        self.assertEqual(r.json("res/params.json"), {"k": 1})            # cached verified text
        self.assertEqual(sorted(r.changed_since_start()), ["res/est.csv", "res/params.json"])  # still reported

    def test_template_names_are_validated(self):
        r = self.reg()
        for bad in ("../x", "t__count; drop", "T__x", "count"):
            with self.assertRaises(ValueError):
                r.template(bad)

    def test_manifest_refuses_raw_and_outside_paths(self):
        for bad in ("Capstone/data/raw/MIMIC-IV-2.2/hosp/patients.csv", "x/mimic_database.duckdb",
                    "../secret.csv", "C:/abs/file.csv"):
            with self.assertRaises(ValueError):
                check_allowed(bad, "T2")
        with self.assertRaises(ValueError):
            check_allowed("roadmap/layer.parquet", "T1")

    def test_missing_manifest_reports_error(self):
        r = Registry(self.tmp / "nope.json", self.tmp)
        self.assertIn("manifest missing", r.error)
        self.assertEqual(r.tiers()["T2"], "absent")


if __name__ == "__main__":
    unittest.main()
