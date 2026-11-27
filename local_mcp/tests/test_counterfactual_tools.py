"""Phase 10 tools: counterfactual_ecg, clinical_what_if, counterfactual_validation, and the cf_pair_validation figure.

Parity tests against the fitted models need the local T1 model files (data/roadmap/phase10) and are skipped on a
public clone; everything else uses only committed aggregates and assets.
"""
import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError

from local_mcp import egress
from local_mcp.paths import ROOT
from local_mcp.server import build_context
from local_mcp.tools import cf_math as M
from local_mcp.tools import counterfactual as C
from local_mcp.tools.computation import Patient
from local_mcp.tools.figures import render_figure

T1 = ROOT / "data" / "roadmap" / "phase10"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def features(seed=0, **over):
    rng = random.Random(seed)
    f = {f"ST_Elev_{l}": rng.uniform(-0.08, 0.08) for l in LEADS}
    f.update({k: rng.uniform(-0.3, 0.4) for k in ("T_Amp_II", "T_Amp_aVR", "T_Amp_V4", "T_Amp_V5", "T_Amp_V6")})
    f.update({"Q_Amp_I": rng.uniform(-0.15, 0), "Q_Amp_aVL": rng.uniform(-0.15, 0)})
    f.update(over)
    return f


class Tools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = build_context(Path(tempfile.mkdtemp()) / "audit.jsonl")
        cls.p = cls.ctx.registry.json(C.TRACK_A)
        cls.pt = Patient(age=62, sex="female", diabetes=True)

    def payload_ok(self, env):
        self.assertEqual(egress.check(egress.payload_of(env)), [])

    # ---------------------------------------------------------------- counterfactual_ecg
    def test_counterfactual_matches_formula_and_is_deterministic(self):
        f = features(1)
        a = C.counterfactual_ecg(self.ctx, self.pt, "add_mi", "stemi_inferior", f, None, "before_mi", False)
        b = C.counterfactual_ecg(self.ctx, self.pt, "add_mi", "stemi_inferior", f, None, "before_mi", False)
        self.assertEqual(a.model_dump(), b.model_dump())
        obs = [M.composites(f, self.p)[m] for m in self.p["measures"]]
        m0, m1 = M.track_a_means(self.pt.covariates(), "stemi_inferior", self.p)
        t = self.p["tuned"]["before"]["add"]
        for j, r in enumerate(a.result.rows):
            want = m1[j] + t["rho"][j] * (obs[j] - m0[j])
            self.assertAlmostEqual(r.counterfactual, round(want, 4), places=4)
            self.assertAlmostEqual(r.interval90[1] - r.interval90[0], 2 * t["q90"][j], places=3)
            self.assertIsNotNone(r.validation_error_sd)
        self.assertTrue(any("STEMI" in fl for fl in a.result.flags))
        self.payload_ok(a)

    def test_synthetic_counterfactual_equals_target_expectation(self):
        e = C.counterfactual_ecg(self.ctx, self.pt, "remove_mi", "nstemi", None, "synthetic_expected", "after_mi", False)
        for r in e.result.rows:
            self.assertAlmostEqual(r.counterfactual, r.expected_in_target_state, places=4)
        self.assertTrue(e.result.synthetic)
        self.assertEqual(e.result.validation_st_t["pairs"],
                         self.ctx.registry.json(C.SUMMARY_JSON)["composite_st_t"]["all_after"]["remove"]["n_pairs"])

    def test_lead_swap_option(self):
        f = features(2, ST_Elev_aVL=0.05, ST_Elev_aVF=-0.05)
        a = C.counterfactual_ecg(self.ctx, self.pt, "add_mi", "nstemi", f, None, "before_mi", False)
        swapped = {(k.replace("aVL", "#").replace("aVF", "aVL").replace("#", "aVF")): v for k, v in f.items()}
        b = C.counterfactual_ecg(self.ctx, self.pt, "add_mi", "nstemi", swapped, None, "before_mi", True)
        self.assertEqual(a.result.rows, b.result.rows)

    def test_refusals(self):
        cases = [
            (features(3, subject_id=10000032), "Unknown ECG feature"),
            ({k: v for k, v in features(3).items() if k != "ST_Elev_V4"}, "Missing ECG feature"),
            (features(3, ST_Elev_V2=12.0), "ECG input rejected"),
            (features(3, ST_Elev_V2=float("nan")), "ECG input rejected"),
        ]
        for f, msg in cases:
            with self.assertRaises(ToolError) as cm:
                C.counterfactual_ecg(self.ctx, self.pt, "add_mi", "nstemi", f, None, "before_mi", False)
            self.assertIn(msg, str(cm.exception))
        for f, ex in ((None, None), (features(4), "synthetic_expected")):
            with self.assertRaises(ToolError):
                C.counterfactual_ecg(self.ctx, self.pt, "add_mi", "nstemi", f, ex, "before_mi", False)

    def test_extra_known_ecgdeli_features_are_ignored(self):
        f = features(5)
        a = C.counterfactual_ecg(self.ctx, self.pt, "add_mi", "nstemi", f, None, "before_mi", False)
        b = C.counterfactual_ecg(self.ctx, self.pt, "add_mi", "nstemi", {**f, "R_Amp_V1": 0.4, "QRS_Dur_Global": 96.0},
                                 None, "before_mi", False)
        self.assertEqual(a.result.rows, b.result.rows)

    # ---------------------------------------------------------------- clinical_what_if
    def test_clinical_never_returns_absolute_risk(self):
        e = C.clinical_what_if(self.ctx, Patient(age=70, sex="male", diabetes=True, hypertension=True), "diabetes", True)
        text = json.dumps(e.model_dump(mode="json"))
        self.assertNotIn("_r0", text)
        self.assertNotIn("_r1", text)
        self.assertNotIn("risk_do", text)
        self.assertEqual([x.role for x in e.result.estimates], ["primary", "alternative"])
        for x in e.result.estimates:
            self.assertLess(x.risk_ratio_ci95[0], x.risk_ratio)
            self.assertLess(x.risk_ratio, x.risk_ratio_ci95[1])
            self.assertAlmostEqual(x.probability_of_necessity, round(max(0, 1 - 1 / x.risk_ratio), 4), places=3)
        self.payload_ok(e)

    def test_pn_only_when_applicable(self):
        no_exp = C.clinical_what_if(self.ctx, Patient(age=70, sex="male"), "hypertension", True)
        no_mi = C.clinical_what_if(self.ctx, Patient(age=70, sex="male", ckd=True), "ckd", False)
        self.assertIsNone(no_exp.result.estimates[0].probability_of_necessity)
        self.assertIsNone(no_mi.result.estimates[0].probability_of_necessity)
        self.assertIn("does not apply", no_exp.summary)

    # ---------------------------------------------------------------- counterfactual_validation
    def test_validation_matches_committed_results(self):
        s = self.ctx.registry.json(C.SUMMARY_JSON)
        e = C.counterfactual_validation(self.ctx, "summary", None, "primary")
        self.assertEqual(e.result.composite_st_t["add"], s["composite_st_t"]["primary"]["add"])
        m = C.counterfactual_validation(self.ctx, "by_measure", "remove_mi", "all_after")
        self.assertEqual(len(m.result.rows), 16)
        self.assertEqual(set(m.result.rows[0].values), {"A1_additive", "A2_shrunk", "no_change", "population_mean"})
        both = C.counterfactual_validation(self.ctx, "by_measure", None, "primary")
        self.assertEqual(len(both.result.rows), 32)
        self.payload_ok(both)
        g = C.counterfactual_validation(self.ctx, "subgroups", "add_mi", "primary")
        self.assertTrue(all(r.n >= C.MIN_CELL and not r.suppressed for r in g.result.rows))
        with self.assertRaises(ToolError):
            C.counterfactual_validation(self.ctx, "subgroups", None, "all_after")
        for env in (e, m, g):
            self.payload_ok(env)

    def test_figure(self):
        out = render_figure(self.ctx, "cf_pair_validation", True)
        body = out.structuredContent
        self.assertEqual(len(body["result"]["table"]), 8)        # 2 directions x 4 methods
        a2 = next(r for r in body["result"]["table"] if r["direction"] == "remove" and r["method"] == "A2_shrunk")
        self.assertLess(a2["mean_abs_error_sd"], 0.75)


@unittest.skipUnless((T1 / "cf3_track_a_model.pkl").is_file(), "local T1 model files not present")
class Parity(unittest.TestCase):
    """The portable formulas reproduce the fitted (sklearn) models to 1e-9."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        cls.pd = pd
        ctx = build_context(Path(tempfile.mkdtemp()) / "audit.jsonl")
        cls.p, cls.c = ctx.registry.json(C.TRACK_A), ctx.registry.json(C.CLINICAL)
        with open(T1 / "cf3_track_a_model.pkl", "rb") as fh:
            cls.ta = pickle.load(fh)["model"]
        rng = random.Random(7)
        cls.profiles = [{"age": rng.uniform(18, 100), "sex_male": rng.randint(0, 1), "diabetes_primary": rng.randint(0, 1),
                         "hypertension_primary": rng.randint(0, 1), "ckd_primary": rng.randint(0, 1),
                         "prior_mi_primary": rng.randint(0, 1)} for _ in range(25)]

    def test_track_a(self):
        for t in self.p["types"]:
            df = self.pd.DataFrame(self.profiles).assign(cf_type=t)
            m0, m1 = self.ta.m0(df), self.ta.m1(df)
            for i, prof in enumerate(self.profiles):
                a0, a1 = M.track_a_means(prof, t, self.p)
                for j in range(16):
                    self.assertAlmostEqual(a0[j], m0[i, j], places=9)
                    self.assertAlmostEqual(a1[j], m1[i, j], places=9)

    @unittest.skipUnless((T1 / "explorer_layer1_models.pkl").is_file(), "explorer Layer 1 models not cached")
    def test_clinical_matches_patient_explorer(self):
        with open(T1 / "explorer_layer1_models.pkl", "rb") as fh:
            ex = pickle.load(fh)
        for key, q in self.c["queries"].items():
            if not key.endswith("primary"):
                continue
            mod = ex[q["exposure"]]
            xt = mod["design"].transform(self.pd.DataFrame(self.profiles))
            for i, prof in enumerate(self.profiles):
                r = M.risk_ratio(prof, q)
                self.assertAlmostEqual(r["_r0"], mod["arm"][0].predict_proba(xt[[i]])[0, 1], places=9)
                self.assertAlmostEqual(r["_r1"], mod["arm"][1].predict_proba(xt[[i]])[0, 1], places=9)


if __name__ == "__main__":
    unittest.main()
