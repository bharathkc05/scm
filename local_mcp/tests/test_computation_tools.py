"""M3 acceptance: confounding_what_if arithmetic, cohort_summary disclosure control, explain_ecg parity with the
Phase 9 browser explainer, render_figure content, and audit redaction of user-supplied inputs."""
import asyncio
import base64
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp.exceptions import ToolError

from local_mcp.envelope import Envelope
from local_mcp.paths import ROOT
from local_mcp.server import build_app, build_context
from local_mcp.tools import computation as C
from local_mcp.tools import ecg_explainer as X
from local_mcp.tools import figures as F
from local_mcp.tools.common import evalue_rr, evalue_rr_ci, evalue_sd

P = ROOT / "roadmap"
FIXTURES = ROOT / "data/roadmap/phase9"
NODE_HARNESS = P / "phase9_demo/tests/node_harness.js"
ROBUST = ["Q_Amp_I", "ST_Elev_I", "T_Amp_II", "ST_Elev_III", "T_Amp_aVR", "ST_Elev_aVR", "Q_Amp_aVL", "ST_Elev_aVL",
          "T_Amp_V4", "ST_Elev_V4", "T_Amp_V5", "ST_Elev_V5", "T_Amp_V6", "ST_Elev_V6", "ST_lateral_mean",
          "ST_max_depression"]
COVS = ["sex_male", "diabetes_primary", "hypertension_primary", "ckd_primary", "prior_mi_primary"]


def code_of(err) -> str:
    s = str(err)
    return json.loads(s[s.index("{"):])["error"]["code"]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.ctx = build_context(cls.tmp / "audit.jsonl")
        if cls.ctx.sandbox is None:
            raise unittest.SkipTest("sandbox unavailable")
        cls.params = cls.ctx.registry.json(C.DEMO)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class WhatIf(Base):
    def test_e_value_on_both_sides_brings_every_estimate_exactly_to_the_null(self):
        l1 = pd.read_csv(P / "phase5_estimation/LAYER1_ESTIMATES.csv").query("estimand == 'ATE_overlap'")
        ids = {"diabetes | age, sex": "L1:diabetes:primary", "diabetes | age, sex, HTN": "L1:diabetes:alternative",
               "hypertension | age, sex, diabetes": "L1:hypertension:primary",
               "hypertension | age, sex (roadmap)": "L1:hypertension:alternative",
               "CKD | age, sex, D, H": "L1:ckd:primary", "CKD | age, sex, D, H, prior MI": "L1:ckd:alternative",
               "prior MI | age, sex, D, H, CKD": "L1:prior_mi:primary"}
        for _, r in l1.iterrows():
            e = evalue_rr(r["rr"])
            if e > 10:
                continue
            out = C.confounding_what_if(self.ctx, ids[r["label"]], e, e).result
            self.assertAlmostEqual(out.worst_case, 1.0, places=4)
            e_ci = evalue_rr_ci(r["rr_ci_low"], r["rr_ci_high"])
            out = C.confounding_what_if(self.ctx, ids[r["label"]], e_ci, e_ci).result
            self.assertAlmostEqual(out.worst_case_ci_limit, 1.0, places=4)
        l2 = pd.read_csv(P / "phase5_estimation/LAYER2_ESTIMATES.csv")
        l2 = l2[l2["family"].isin(["point", "st_composite"])].assign(m=lambda d: d["outcome"].str.removeprefix("w_")).set_index("m")
        for m in ROBUST:
            e, _ = evalue_sd(l2.loc[m, "att_sd"], l2.loc[m, "att_sd_ci_low"], l2.loc[m, "att_sd_ci_high"])
            out = C.confounding_what_if(self.ctx, f"L2:{m}", e, e).result
            self.assertAlmostEqual(out.worst_case, 0.0, places=4)

    def test_verdict_flags_and_frontier(self):
        weak = C.confounding_what_if(self.ctx, "L1:hypertension:primary", 2.0, 2.0).result
        strong = C.confounding_what_if(self.ctx, "L1:hypertension:primary", 2.5, 2.5).result
        self.assertEqual((weak.explained_away_point, weak.explained_away_ci), (False, False))
        self.assertAlmostEqual(weak.worst_case, 1.0815, places=3)
        self.assertTrue(strong.explained_away_point)
        for f in weak.frontier:
            if f.min_rr_confounder_outcome_point is None:
                self.assertLessEqual(f.rr_confounder_exposure, weak.observed)        # impossible below the RR
                continue
            a, b = f.rr_confounder_exposure, f.min_rr_confounder_outcome_point
            self.assertAlmostEqual(a * b / (a + b - 1), weak.observed, places=3)

    def test_errors(self):
        for rid in ("hypertension", "L2:z_ecg_3", "L1:prior_mi:alternative"):
            with self.assertRaises(ToolError):
                C.confounding_what_if(self.ctx, rid, 2, 2)


class CohortSummary(Base):
    def setUp(self):
        if not self.ctx.sandbox.has("layer1", "layer2"):
            self.skipTest("T1 files not on this machine")

    def test_values_match_an_independent_read(self):
        l1 = pd.read_parquet(ROOT / "data/roadmap/phase3/layer1.parquet", columns=["acute_mi", "age", "sex_male"])
        out = {c.strata["acute_mi"]: c for c in C.cohort_summary(self.ctx, "layer1", ["acute_mi"], ["age_mean", "pct_male"]).result.rows}
        for v in (0, 1):
            sub = l1[l1["acute_mi"] == v]
            self.assertEqual(out[v].n_admissions, C._round(len(sub)))
            self.assertAlmostEqual(out[v].metrics["age_mean"], round(sub["age"].mean(), 1))
            self.assertAlmostEqual(out[v].metrics["pct_male"], round(100 * sub["sex_male"].mean(), 1))

    def test_disclosure_control_on_a_sparse_table(self):
        env = C.cohort_summary(self.ctx, "layer2_primary", ["mi_subtype", "age_band"], ["age_mean"])
        rows = env.result.rows
        self.assertGreater(env.result.suppressed_cells, 0)
        self.assertEqual(env.disclosure.suppressed_cells, env.result.suppressed_cells)
        # true counts, read independently
        l2 = pd.read_parquet(ROOT / "data/roadmap/phase3/layer2.parquet",
                             columns=["set_primary", "acute_mi", "mi_stemi", "mi_nstemi", "mi_type2", "age"])
        l2 = l2[(l2["set_primary"] == 1) & (l2["acute_mi"] == 1)]
        sub = l2.apply(lambda r: "STEMI" if r.mi_stemi else "NSTEMI" if r.mi_nstemi else "type 2" if r.mi_type2 else "other", axis=1)
        band = pd.cut(l2["age"], [0, 45, 65, 80, 200], right=False, labels=["18-44", "45-64", "65-79", "80+"]).astype(str)
        true = pd.crosstab(sub, band)
        for c in rows:
            n_true = int(true.loc[c.strata["mi_subtype"], c.strata["age_band"]])
            if c.suppressed:
                self.assertIsNone(c.n_admissions)
                self.assertTrue(all(v is None for v in c.metrics.values()))
            else:
                self.assertGreaterEqual(n_true, C.MIN_CELL)                  # nothing small ever leaves
                self.assertEqual(c.n_admissions % 5, 0)
                self.assertEqual(c.n_admissions, C._round(n_true))
        for dim, other in (("mi_subtype", "age_band"), ("age_band", "mi_subtype")):
            lines: dict = {}
            for c in rows:
                lines.setdefault(c.strata[other], []).append(c.suppressed)
            for flags in lines.values():          # no line isolates a single suppressed cell (margin differencing)
                self.assertNotEqual(sum(flags), 1)

    def test_suppress_unit_cases(self):
        one_dim = [{"d": "a", "n_admissions": 5}, {"d": "b", "n_admissions": 100}, {"d": "c", "n_admissions": 200}]
        self.assertEqual(C.suppress(one_dim, ["d"]), {0, 1})
        self.assertEqual(C.suppress([{"n_admissions": 500}], []), set())
        self.assertEqual(C.suppress([{"d": "a", "n_admissions": 50}, {"d": "b", "n_admissions": 60}], ["d"]), set())

    def test_possession_gating_and_argument_errors(self):
        sb = copy.copy(self.ctx.sandbox)
        sb.relations = self.ctx.sandbox.relations - {"layer1", "layer2"}
        gated = copy.copy(self.ctx)
        gated.sandbox = sb
        with self.assertRaises(ToolError) as e:
            C.cohort_summary(gated, "layer1", [], ["age_mean"])
        self.assertEqual(code_of(e.exception), "DATA_NOT_PRESENT")
        for args in (("layer1", ["timing_class"], []), ("layer1", ["sex", "sex"], []),
                     ("layer1", ["sex", "ckd", "age_band"], [])):
            with self.assertRaises(ToolError) as e:
                C.cohort_summary(self.ctx, *args)
            self.assertEqual(code_of(e.exception), "INVALID_ARGUMENT")


class ExplainEcg(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        f = FIXTURES / "test_cases.json"
        cls.cases = json.loads(f.read_text()) if f.exists() else []

    def cov(self, c):
        return {"age": c["age"], **{k: c[k] for k in COVS}}

    def test_expected_values_match_phase9_held_out_cases(self):
        if not self.cases:
            self.skipTest("Phase 9 fixtures not on this machine")
        for c in self.cases:
            got = X.expected_without_mi(self.cov(c), self.params)
            for m, v in c["expected_without_mi"].items():
                self.assertAlmostEqual(got[m], v, delta=1e-9)

    def test_bitwise_parity_with_the_browser_explainer(self):
        if not self.cases or shutil.which("node") is None:
            self.skipTest("fixtures or node not available")
        cmd = {"op": "explain", "cases": [{"patient": self.cov(c), "features": c["features"],
                                           "opts": {"timing": "after_onset"}} for c in self.cases]}
        js = json.loads(subprocess.run(["node", str(NODE_HARNESS)], input=json.dumps(cmd), capture_output=True,
                                       text=True, check=True).stdout)
        for c, j in zip(self.cases, js):
            py = X.explain(self.cov(c), c["features"], self.params, timing="after_onset")
            self.assertEqual(py["flags"], j["flags"])
            for a, b in zip(py["rows"], j["rows"]):
                for k in ("observed", "expected_without_mi", "difference_sd"):
                    self.assertAlmostEqual(a[k], b[k], delta=1e-12)
                self.assertEqual(a["in_mi_effect_direction"], b["in_mi_effect_direction"])
            for a, b in zip(py["groups"], j["groups"]):
                self.assertAlmostEqual(a["mean_aligned_difference_sd"], b["mean_aligned_difference_sd"], delta=1e-12)

    def test_raw_ecgdeli_file_lead_swap(self):
        f = FIXTURES / "fresh_file_raw_format.csv"
        if not f.exists():
            self.skipTest("Phase 9 fixtures not on this machine")
        by_id = {c["study_id"]: c for c in self.cases}
        pre = self.params["preprocessing"]
        for _, row in pd.read_csv(f).iterrows():
            c = by_id[int(row["study_id"])]
            out = X.explain(self.cov(c), row.to_dict(), self.params, timing="after_onset", raw_ecgdeli=True)
            for r in out["rows"]:
                if r["measure"] in pre["point_inputs"]:
                    lo, hi = pre["winsor_bounds"][r["measure"]]
                    self.assertAlmostEqual(r["observed"], min(max(c["features"][r["measure"]], lo), hi), delta=1e-12)

    def test_profile_mode(self):
        pat = C.Patient(age=65, sex="male", diabetes=True, hypertension=True)
        prof = C.explain_ecg(self.ctx, pat, None, None, "unknown", False).result
        self.assertEqual(prof.mode, "profile")
        base = X.expected_without_mi(pat.covariates(), self.params)
        for r in prof.rows:
            self.assertAlmostEqual(r.expected_without_mi, round(base[r.measure], 4), places=4)  # same no-MI model
            lo, hi = r.model_effect_ci95
            self.assertLessEqual(lo, r.model_implied_mi_effect + 1e-4)
            self.assertGreaterEqual(hi, r.model_implied_mi_effect - 1e-4)
        ph6 = pd.read_csv(P / "phase6_novelty/COUNTERFACTUAL_PROFILES.csv")
        profiles = {"55y man, no comorbidity": C.Patient(age=55, sex="male"),
                    "65y man, diabetes + HTN": C.Patient(age=65, sex="male", diabetes=True, hypertension=True),
                    "75y woman, HTN + CKD": C.Patient(age=75, sex="female", hypertension=True, ckd=True),
                    "70y man, D + H + CKD + prior MI": C.Patient(age=70, sex="male", diabetes=True, hypertension=True,
                                                                ckd=True, prior_mi=True)}
        for name, pat in profiles.items():
            rows = {r.measure: r for r in C.explain_ecg(self.ctx, pat, None, None, "unknown", False).result.rows}
            for m in ("ST_lateral_mean", "T_Amp_V5"):
                p6 = ph6.query("profile == @name and outcome == @m").iloc[0]["model_implied_mi_effect"]
                self.assertEqual(rows[m].model_implied_mi_effect < 0, p6 < 0, (name, m))   # same direction as Phase 6

    def test_synthetic_examples_and_safety(self):
        pat = C.Patient(age=65, sex="male", diabetes=True, hypertension=True)
        no_mi = C.explain_ecg(self.ctx, pat, None, "synthetic_no_mi", "after_onset", False)
        for r in no_mi.result.rows:
            if r.measure in self.params["preprocessing"]["point_inputs"]:
                self.assertAlmostEqual(r.difference_sd, 0.0, places=6)
        shift = C.explain_ecg(self.ctx, pat, None, "synthetic_typical_mi_shift", "after_onset", False).result
        self.assertTrue(shift.synthetic)
        self.assertTrue(all(r.in_mi_effect_direction for r in shift.rows if r.measure in self.params["preprocessing"]["point_inputs"]))
        dumped = no_mi.model_dump_json().lower()
        for word in ("probability", "diagnosis_label", "risk_score", "recommend"):
            self.assertNotIn(word, dumped.replace("no probability of mi", ""))
        with self.assertRaises(ToolError):
            C.explain_ecg(self.ctx, pat, {"ST_Elev_I": 0.0}, "synthetic_no_mi", "unknown", False)
        with self.assertRaises(ToolError) as e:
            C.explain_ecg(self.ctx, pat, {"ST_Elev_I": 0.0}, None, "unknown", False)
        body = json.loads(str(e.exception)[str(e.exception).index("{"):])["error"]
        self.assertTrue(any("Missing ECG field" in m for m in body["errors"]))


class RenderFigure(Base):
    def test_image_table_and_schema(self):
        for fid, n in (("st_t_effects", 24), ("crude_vs_adjusted", 53), ("cohort_flow", 11)):
            r = F.render_figure(self.ctx, fid, True)
            png = base64.b64decode(r.content[0].data)
            self.assertTrue(png.startswith(b"\x89PNG"))
            a = self.ctx.registry.find(f"local_mcp/assets/figures/{fid}.png")
            self.assertEqual(hashlib.sha256(png).hexdigest(), a.sha256)
            Envelope[F.FigureResult].model_validate(r.structuredContent)
            self.assertEqual(len(r.structuredContent["result"]["table"]), n)
        no_img = F.render_figure(self.ctx, "latent_audit", False)
        self.assertEqual([c.type for c in no_img.content], ["text"])
        self.assertLessEqual(len(no_img.structuredContent["result"]["table"]), 60)


class AuditRedaction(Base):
    def test_user_supplied_inputs_are_hashed_in_the_audit_log(self):
        app = build_app(self.ctx)
        asyncio.run(app.call_tool("explain_ecg", {"patient": {"age": 61, "sex": "female", "ckd": True},
                                                  "example_id": "synthetic_no_mi"}))
        last = json.loads(self.ctx.audit.path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(last["tool"], "explain_ecg")
        self.assertEqual(set(last["args"]["patient"]), {"sha256"})
        self.assertEqual(last["args"]["example_id"], "synthetic_no_mi")
        self.assertNotIn("61", json.dumps(last["args"]))


if __name__ == "__main__":
    unittest.main()
