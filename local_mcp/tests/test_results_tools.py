"""M2 acceptance: every number the results tools return matches the source CSVs (read independently with
pandas), verdicts follow the pre-declared rules, names resolve, errors are structured, and calls are fast."""
import dataclasses
import json
import shutil
import statistics
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp.exceptions import ToolError

from local_mcp.paths import ROOT
from local_mcp.server import build_context
from local_mcp.tools import orientation, provenance, results
from local_mcp.tools.common import evalue_rr, evalue_rr_ci, evalue_sd

P = ROOT / "roadmap"
ROBUST = ["Q_Amp_I", "ST_Elev_I", "T_Amp_II", "ST_Elev_III", "T_Amp_aVR", "ST_Elev_aVR", "Q_Amp_aVL", "ST_Elev_aVL",
          "T_Amp_V4", "ST_Elev_V4", "T_Amp_V5", "ST_Elev_V5", "T_Amp_V6", "ST_Elev_V6", "ST_lateral_mean",
          "ST_max_depression"]                         # Phase 6 decision S2
L1_IDS = {"diabetes | age, sex": "L1:diabetes:primary", "diabetes | age, sex, HTN": "L1:diabetes:alternative",
          "hypertension | age, sex, diabetes": "L1:hypertension:primary",
          "hypertension | age, sex (roadmap)": "L1:hypertension:alternative", "CKD | age, sex, D, H": "L1:ckd:primary",
          "CKD | age, sex, D, H, prior MI": "L1:ckd:alternative", "prior MI | age, sex, D, H, CKD": "L1:prior_mi:primary"}


def code_of(err: ToolError) -> str:
    return json.loads(str(err))["error"]["code"]


class ResultsTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.ctx = build_context(cls.tmp / "audit.jsonl")
        if cls.ctx.sandbox is None or not cls.ctx.sandbox.has("l1_effects", "l2_effects"):
            raise unittest.SkipTest("results tier not available")
        cls.l1 = pd.read_csv(P / "phase5_estimation/LAYER1_ESTIMATES.csv")
        cls.l2 = pd.read_csv(P / "phase5_estimation/LAYER2_ESTIMATES.csv")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------ Layer 1 numbers
    def test_risk_factor_effects_match_phase5(self):
        est = {"ATE_overlap": "ate_overlap", "ATT": "att", "ATE": "ate"}
        for _, r in self.l1.iterrows():
            _, exp, adj = L1_IDS[r["label"]].split(":")
            out = results.get_risk_factor_effect(self.ctx, exp, adj, est[r["estimand"]]).result
            self.assertAlmostEqual(out.risk_ratio, r["rr"], places=4)
            self.assertAlmostEqual(out.rr_ci95[0], r["rr_ci_low"], places=4)
            self.assertAlmostEqual(out.rr_ci95[1], r["rr_ci_high"], places=4)
            self.assertAlmostEqual(out.risk_difference_pp, r["rd_pp"], places=4)
            self.assertEqual(out.n_admissions, r["n"])

    def test_layer1_evalues_match_phase7(self):
        ev = pd.read_csv(P / "phase7_stress_test/EVALUES.csv").query("layer == 1")
        l1 = self.l1.query("estimand == 'ATE_overlap'").set_index("label")
        for _, r in ev.iterrows():
            row = l1.loc[r["label"]]
            self.assertAlmostEqual(evalue_rr(row["rr"]), r["evalue_point"], places=6)
            self.assertAlmostEqual(evalue_rr_ci(row["rr_ci_low"], row["rr_ci_high"]), r["evalue_ci"], places=6)

    def test_layer1_verdicts(self):
        want = {"L1:diabetes:primary": "survives", "L1:diabetes:alternative": "survives",
                "L1:hypertension:primary": "survives", "L1:ckd:primary": "fragile",
                "L1:prior_mi:primary": "survives (descriptive)", "L1:hypertension:alternative": "not stress-tested"}
        for rid, verdict in want.items():
            self.assertEqual(results.robustness_card(self.ctx, rid).result.verdict, verdict, rid)
        ckd = results.robustness_card(self.ctx, "L1:ckd:primary").result
        self.assertEqual(ckd.failed_checks, ["MI coded as the principal diagnosis"])
        htn = results.robustness_card(self.ctx, "L1:hypertension:primary").result
        strict = [c for c in htn.checks if c.check.startswith("Strict")][0]
        self.assertFalse(strict.counted_in_verdict)          # the 0.72 artifact is shown, not counted
        self.assertLess(strict.estimate, 1)
        self.assertTrue(any(c.group == "bmi" for c in htn.checks))

    # ------------------------------------------------------------ Layer 2 numbers
    def test_ecg_effects_match_phase5_for_all_point_features_and_composites(self):
        rows = self.l2[self.l2["family"].isin(["point", "st_composite"])]
        for _, r in rows.iterrows():
            m = r["outcome"].removeprefix("w_")
            out = results.get_ecg_effect(self.ctx, m, None, "primary", "sd", 10).result.rows[0]
            self.assertEqual(out.measure, m)
            self.assertAlmostEqual(out.att_sd, r["att_sd"], places=4)
            self.assertAlmostEqual(out.ci95_sd[0], r["att_sd_ci_low"], places=4)
            self.assertAlmostEqual(out.att_raw, r["att"], places=4)
            if r["family"] == "point":
                self.assertAlmostEqual(out.q_value, float(f"{r['att_q']:.3g}"))

    def test_sensitivity_sets_match_their_sources(self):
        sens = pd.read_csv(P / "phase6_novelty/SENSITIVITY_ESTIMATES.csv")
        ns = sens.query("analysis_set == 'nstemi_only' and outcome == 'w_ST_lateral_mean'").iloc[0]
        out = results.get_ecg_effect(self.ctx, "ST_lateral_mean", None, "nstemi", "sd", 10).result.rows[0]
        self.assertAlmostEqual(out.att_sd, ns["att_sd"], places=4)
        sel = pd.read_csv(P / "phase7_stress_test/SELECTION_WEIGHTED_LAYER2.csv").set_index("outcome")
        out = results.get_ecg_effect(self.ctx, "T_Amp_V5", None, "selection_weighted", "sd", 10).result.rows[0]
        self.assertAlmostEqual(out.att_sd, sel.loc["T_Amp_V5", "att_sd"], places=4)
        broad = pd.read_csv(P / "phase7_stress_test/OUTCOME_CODES_LAYER2_BROAD.csv").set_index("outcome")
        out = results.get_ecg_effect(self.ctx, "ST_Elev_I", None, "broad_codes", "sd", 10).result.rows[0]
        self.assertAlmostEqual(out.att_sd, broad.loc["ST_Elev_I", "att_sd"], places=4)

    def test_group_mode_and_latents(self):
        st = results.get_ecg_effect(self.ctx, None, "ST", "primary", "sd", 20).result
        self.assertEqual(st.group_summary.measures, 12)
        self.assertEqual(st.group_summary.fdr_significant, 12)
        self.assertEqual(st.rows[0].measure, "ST_Elev_I")                  # largest |effect| first
        lat = results.get_ecg_effect(self.ctx, None, "latent", "primary", "sd", 3).result
        self.assertEqual((lat.latent_summary.latents, lat.latent_summary.fdr_significant), (64, 43))
        self.assertEqual(lat.latents[0].latent, "z_ecg_57")

    def test_layer2_verdicts(self):
        for m in ROBUST:
            self.assertEqual(results.robustness_card(self.ctx, f"L2:{m}").result.verdict, "survives", m)
        lat = results.robustness_card(self.ctx, "L2:ST_lateral_mean").result
        self.assertEqual(len(lat.checks), 13)
        nc = [c for c in lat.checks if c.group == "negative_control"][0]
        src = pd.read_csv(P / "phase6_novelty/DECONFOUNDED_ECG_EFFECTS.csv").set_index("outcome")
        self.assertAlmostEqual(nc.ratio_to_primary, src.loc["ST_lateral_mean", "negative_control_ratio"], delta=1e-3)
        self.assertTrue(nc.passed)
        self.assertEqual(results.robustness_card(self.ctx, "L2:ST_max_elevation").result.failed_checks,
                         ["NSTEMI only (no ST-elevation coding)"])
        rr = results.robustness_card(self.ctx, "L2:RR_Mean_Global").result
        self.assertEqual(rr.verdict, "not robust")
        self.assertIn("Negative control (ECG before a later MI)", rr.failed_checks)
        self.assertEqual(results.robustness_card(self.ctx, "L2:QRS_Dur_Global").result.verdict,
                         "survives (measurement not validated)")

    def test_layer2_evalues_formula_matches_phase7(self):
        ev = pd.read_csv(P / "phase7_stress_test/EVALUES.csv").query("layer == 2").set_index("outcome")
        l2 = self.l2[self.l2["family"].isin(["point", "st_composite"])].assign(m=lambda d: d["outcome"].str.removeprefix("w_")).set_index("m")
        for m in ROBUST:
            pt, ci = evalue_sd(l2.loc[m, "att_sd"], l2.loc[m, "att_sd_ci_low"], l2.loc[m, "att_sd_ci_high"])
            self.assertAlmostEqual(pt, ev.loc[m, "evalue_point"], places=6)
            self.assertAlmostEqual(ci, ev.loc[m, "evalue_ci"], places=6)

    def test_subgroups(self):
        inter = pd.read_csv(P / "phase7_stress_test/SUBGROUP_INTERACTIONS_LAYER2.csv")
        row = inter.query("modifier == 'age' and outcome == 'ST_lateral_mean'").iloc[0]
        out = results.get_subgroup_effects(self.ctx, "L2:ST_lateral_mean", "age").result
        self.assertAlmostEqual(out.interaction.difference, row["difference"], places=4)
        self.assertAlmostEqual(out.interaction.q_value, float(f"{row['q']:.3g}"))
        self.assertEqual([s.subgroup for s in out.subgroups], ["age_lt65", "age_ge65"])
        nonrobust = results.get_subgroup_effects(self.ctx, "L2:R_Amp_V1", "sex").result
        self.assertIsNone(nonrobust.interaction.q_value)               # computed, outside the Phase 7 FDR family
        l1 = results.get_subgroup_effects(self.ctx, "L1:hypertension:primary", "sex").result
        src = pd.read_csv(P / "phase7_stress_test/SUBGROUPS_LAYER1.csv").query(
            "label == 'hypertension | age, sex, diabetes'").set_index("subgroup")
        self.assertAlmostEqual(l1.subgroups[0].estimate, src.loc["female", "rr"], places=4)

    # ------------------------------------------------------------ orientation and provenance
    def test_overview(self):
        o = orientation.study_overview(self.ctx).result
        flow = pd.read_csv(P / "phase3_dataset/cohort_flow.csv")
        self.assertEqual(o.cohort["layer1_admissions"], int(flow.iloc[1]["admissions"]))
        self.assertEqual(o.cohort["layer2_after_onset_mi"], 5643)
        self.assertEqual([h.verdict for h in o.headline_layer1],
                         ["survives", "survives", "fragile", "survives (descriptive)"])
        self.assertNotIn("L2:QRS_Dur_Global", [h.result_id for h in o.headline_layer2])
        self.assertEqual(o.layer2_significance["point"], "34/53 FDR-significant")

    def test_name_matching(self):
        cases = {"heart attack": "acute_mi", "lateral ST": "ST_lateral_mean", "T wave V5": "T_Amp_V5",
                 "z57": "z_ecg_57", "st aVL": "ST_Elev_aVL", "w_ST_Elev_I": "ST_Elev_I", "HTN": "hypertension",
                 "NSTEMI": "mi_nstemi", "chest pain": "chest_pain"}
        for q, want in cases.items():
            self.assertEqual(orientation.describe_variable(self.ctx, q).result.variable.name, want, q)
        with self.assertRaises(ToolError) as e:
            orientation.describe_variable(self.ctx, "lateral ST levle")
        body = json.loads(str(e.exception))["error"]
        self.assertEqual(body["code"], "INVALID_ARGUMENT")
        self.assertIn("ST_lateral_mean", body["did_you_mean"])

    def test_trace_claim(self):
        t = provenance.trace_claim(self.ctx, "6", None).result.claims[0]
        self.assertEqual(t.claim_id, "C06")
        self.assertEqual([a.id for a in t.assumptions], ["A2", "A3", "A4"])
        for e in t.evidence:
            self.assertTrue(e.verified)
            self.assertEqual(e.sha256, self.ctx.registry.find(e.path).sha256)
        self.assertIn("get_ecg_effect", t.reproduce_with)
        q = provenance.trace_claim(self.ctx, None, "NSTEMI coding artefact").result.claims[0]
        self.assertEqual(q.claim_id, "C09")
        self.assertTrue(all(e.verified for e in q.evidence))

    # ------------------------------------------------------------ errors
    def test_structured_errors(self):
        bad = [
            (lambda: results.get_ecg_effect(self.ctx, "ST_Elev_I", "ST", "primary", "sd", 5), "INVALID_ARGUMENT"),
            (lambda: results.get_ecg_effect(self.ctx, None, "latent", "nstemi", "sd", 5), "INVALID_ARGUMENT"),
            (lambda: results.get_ecg_effect(self.ctx, "ST_Elev_I_iqr", None, "nstemi", "sd", 5), "INVALID_ARGUMENT"),
            (lambda: results.get_ecg_effect(self.ctx, "diabetes", None, "primary", "sd", 5), "INVALID_ARGUMENT"),
            (lambda: results.get_risk_factor_effect(self.ctx, "prior_mi", "alternative", "ate_overlap"), "INVALID_ARGUMENT"),
            (lambda: results.robustness_card(self.ctx, "hypertension"), "INVALID_ARGUMENT"),
            (lambda: results.robustness_card(self.ctx, "L2:z_ecg_3"), "NOT_SUPPORTED"),
            (lambda: provenance.trace_claim(self.ctx, "C99", None), "INVALID_ARGUMENT"),
        ]
        for fn, code in bad:
            with self.assertRaises(ToolError) as e:
                fn()
            self.assertEqual(code_of(e.exception), code)

    def test_data_not_present_without_sandbox(self):
        empty = dataclasses.replace(self.ctx, sandbox=None, cache={})
        with self.assertRaises(ToolError) as e:
            results.get_ecg_effect(empty, "ST_Elev_I", None, "primary", "sd", 5)
        self.assertEqual(code_of(e.exception), "DATA_NOT_PRESENT")

    # ------------------------------------------------------------ envelope and speed
    def test_envelopes_carry_provenance_and_caveats(self):
        env = results.get_ecg_effect(self.ctx, "ST_lateral_mean", None, "primary", "sd", 10)
        self.assertTrue(env.summary)
        self.assertTrue(any("negative control" in c for c in env.caveats))
        self.assertEqual([a.id for a in env.assumptions], ["A2", "A3", "A4"])
        for ref in env.provenance.artifacts:
            self.assertEqual(ref.sha256, self.ctx.registry.find(ref.path).sha256)
        self.assertEqual(env.provenance.manifest_version, self.ctx.registry.version)
        dumped = env.model_dump_json()
        for forbidden in ("subject_id", "hadm_id", "study_id"):
            self.assertNotIn(forbidden, dumped)

    def test_latency(self):
        def median_ms(fn, n=5):
            fn()
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                fn()
                ts.append((time.perf_counter() - t) * 1000)
            return statistics.median(ts)
        for name, params in [("ecg__measure", ["primary", "ST_Elev_I"]), ("ecg__all_sets", ["ST_lateral_mean"]),
                             ("risk__checks", ["L1:ckd:primary"]), ("ecg__group", ["primary", "point", None, 10])]:
            self.assertLess(median_ms(lambda: self.ctx.sandbox.query(name, params)), 100, name)
        self.assertLess(median_ms(lambda: results.robustness_card(self.ctx, "L2:ST_lateral_mean")), 100)
        self.assertLess(median_ms(lambda: orientation.study_overview(self.ctx), 3), 1000)


if __name__ == "__main__":
    unittest.main()
