"""Phase 9 acceptance tests (stdlib unittest + Node).

Run from the repo root with the project env:
  C:/Users/Acer/Desktop/Capstone/Capstone/cuda_env/Scripts/python.exe -m unittest roadmap.phase9_demo.tests.test_demo -v
Needs the git-ignored fixtures from p9_01_export.py (data/roadmap/phase9/).
"""
import json
import subprocess
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS
from roadmap.estimators import Design
from roadmap.phase5_estimation.queries import L2_COVS
from roadmap.phase7_stress_test.common import ROBUST, spec

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((PHASE_DIRS[9] / "demo_params.json").read_text())
CASES = json.loads((DATA_OUT / "phase9/test_cases.json").read_text())
FRESH = DATA_OUT / "phase9/fresh_file_raw_format.csv"


def node(cmd: dict):
    r = subprocess.run(["node", str(HERE / "node_harness.js")], input=json.dumps(cmd), capture_output=True,
                       text=True, check=True)
    return json.loads(r.stdout)


def patient(c):
    return {"age": c["age"], **{k: c[k] for k in L2_COVS}}


class Preprocessing(unittest.TestCase):
    def test_design_matches_sklearn(self):
        """JS spline + design + scaling equals the fitted Python Design on a grid (incl. extrapolation)."""
        l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet", columns=["age", "split", "set_primary"] + L2_COVS)
        train = l2[(l2["set_primary"] == 1) & (l2["split"] == "train")]
        des = Design(spec("layer2"), L2_COVS).fit(train)
        rng = np.random.default_rng(0)
        grid = pd.DataFrame({"age": np.r_[np.linspace(10, 110, 60), [18, 100, float(train["age"].min()), float(train["age"].max())]]})
        for c in L2_COVS:
            grid[c] = rng.integers(0, 2, len(grid))
        js = np.array(node({"op": "design", "patients": grid.to_dict("records")}))
        np.testing.assert_allclose(js, des.transform(grid), atol=1e-9)

    def test_winsorization_and_lead_swap(self):
        pre = PARAMS["preprocessing"]
        c = CASES[0]
        f = dict(c["features"])
        f["ST_Elev_I"] = 99.0                                              # far outside -> rejected
        r = node({"op": "explain", "cases": [{"patient": patient(c), "features": f}]})[0]
        self.assertTrue(r["rejected"])
        f = dict(c["features"])
        f["ST_Elev_I"] = pre["training_range_p001_p999"]["ST_Elev_I"][1] + 1e-3   # just outside -> flagged + clipped
        r = node({"op": "explain", "cases": [{"patient": patient(c), "features": f, "opts": {"timing": "after_onset"}}]})[0]
        self.assertFalse(r["rejected"])
        self.assertTrue(any("outside the range seen in training" in x for x in r["flags"]))
        row = next(x for x in r["rows"] if x["measure"] == "ST_Elev_I")
        self.assertAlmostEqual(row["observed"], pre["winsor_bounds"]["ST_Elev_I"][1], places=12)


class HeldOutAndFreshFile(unittest.TestCase):
    def test_reproduces_stored_values_for_heldout_patients(self):
        out = node({"op": "explain", "cases": [{"patient": patient(c), "features": c["features"],
                                                "opts": {"timing": "after_onset"}} for c in CASES]})
        for c, r in zip(CASES, out):
            self.assertFalse(r["rejected"], r["errors"])
            got = {x["measure"]: x["expected_without_mi"] for x in r["rows"]}
            for m in ROBUST:
                self.assertAlmostEqual(got[m], c["expected_without_mi"][m], delta=1e-9)

    def test_fresh_raw_format_file(self):
        """A file straight from the ECGdeli pipeline (aVL/aVF swapped) gives the same observed values."""
        raw = pd.read_csv(FRESH)
        by_id = {c["study_id"]: c for c in CASES}
        pre = PARAMS["preprocessing"]
        for _, row in raw.iterrows():
            c = by_id[int(row["study_id"])]
            r = node({"op": "explain", "cases": [{"patient": patient(c), "features": row.to_dict(),
                                                  "opts": {"rawEcgdeli": True, "timing": "after_onset"}}]})[0]
            self.assertFalse(r["rejected"], r["errors"])
            for x in r["rows"]:
                if x["measure"] in pre["point_inputs"]:
                    want = min(max(c["features"][x["measure"]], pre["winsor_bounds"][x["measure"]][0]),
                               pre["winsor_bounds"][x["measure"]][1])
                    self.assertAlmostEqual(x["observed"], want, delta=1e-9, msg=x["measure"])

    def test_csv_parser_path(self):
        c = by = CASES[0]
        out = node({"op": "csv", "file": str(FRESH), "patient": patient(by)})
        self.assertEqual(len(out), len(pd.read_csv(FRESH)))
        self.assertTrue(all(not r["rejected"] for r in out))


class Rejections(unittest.TestCase):
    def run_one(self, p, f, opts=None):
        return node({"op": "explain", "cases": [{"patient": p, "features": f, "opts": opts or {}}]})[0]

    def test_missing_field_rejected_with_message(self):
        c = CASES[0]
        f = {k: v for k, v in c["features"].items() if k != "T_Amp_V5"}
        r = self.run_one(patient(c), f)
        self.assertTrue(r["rejected"])
        self.assertTrue(any("T_Amp_V5" in e for e in r["errors"]))

    def test_non_numeric_and_bad_patient_fields(self):
        c = CASES[0]
        f = dict(c["features"], ST_Elev_V5="abc")
        p = dict(patient(c), age=150, diabetes_primary=2)
        r = self.run_one(p, f)
        self.assertTrue(r["rejected"])
        joined = " ".join(r["errors"])
        for s in ("ST_Elev_V5", "Age 150", "diabetes_primary"):
            self.assertIn(s, joined)


class Safety(unittest.TestCase):
    def test_banner_caveat_flags_always_present(self):
        c = CASES[0]
        f = dict(c["features"], QRS_Dur_Global_count=3)
        r = node({"op": "explain", "cases": [{"patient": patient(c), "features": f}]})[0]   # timing unknown
        self.assertIn("not a diagnosis", r["caveat"])
        self.assertTrue(any("Unknown ECG timing" in x for x in r["flags"]))
        self.assertTrue(any("beat count" in x for x in r["flags"]))
        rej = node({"op": "explain", "cases": [{"patient": patient(c), "features": {}}]})[0]
        self.assertIn("not a diagnosis", rej["caveat"])

    def test_only_gate2_measures_and_no_forbidden_outputs(self):
        self.assertEqual(PARAMS["structural"]["measures"], ROBUST)
        txt = json.dumps(PARAMS)
        self.assertNotIn("z_ecg", txt)
        self.assertNotIn("QRS_Dur_Global\"", json.dumps(PARAMS["effects"]))
        r = node({"op": "explain", "cases": [{"patient": patient(CASES[0]), "features": CASES[0]["features"]}]})[0]
        keys = json.dumps(r).lower()
        for word in ("probability", "risk_score", "diagnosis_label", "recommend", "treatment"):
            self.assertNotIn(word, keys.replace("not a diagnosis", ""))

    def test_page_has_banner_and_no_network_calls(self):
        page = (PHASE_DIRS[9] / "ecg_explainer.html").read_text(encoding="utf-8")
        self.assertIn("Research prototype: not a diagnosis, not medical advice", page)
        for s in ("fetch(", "XMLHttpRequest", "navigator.sendBeacon", "<script src=\"http"):
            self.assertNotIn(s, page)


if __name__ == "__main__":
    unittest.main()
