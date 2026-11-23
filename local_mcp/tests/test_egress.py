"""M4 acceptance: the egress guard blocks planted identifiers, shifted dates, denylisted keys and oversized
payloads, through the real audited wrapper and MCP protocol; and it never blocks a real tool output."""
import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import ToolAnnotations

from local_mcp import egress
from local_mcp.envelope import Envelope, Provenance, fail
from local_mcp.paths import ROOT
from local_mcp.runtime import audited
from local_mcp.server import build_app, build_context

REAL_CALLS = [
    ("study_overview", {}), ("describe_variable", {"name": "lateral ST"}),
    ("get_risk_factor_effect", {"exposure": "diabetes"}), ("get_ecg_effect", {"group": "T", "top_k": 20}),
    ("get_ecg_effect", {"group": "latent", "top_k": 20}),
    ("get_subgroup_effects", {"result_id": "L1:ckd:primary", "modifier": "age"}),
    ("robustness_card", {"result_id": "L2:ST_lateral_mean"}), ("robustness_card", {"result_id": "L1:hypertension:primary"}),
    ("confounding_what_if", {"result_id": "L2:T_Amp_V5", "rr_confounder_exposure": 2, "rr_confounder_outcome": 2}),
    ("cohort_summary", {"layer": "layer2_candidates", "stratify_by": ["timing_class", "age_band"],
                        "metrics": ["age_mean", "age_sd", "pct_male", "pct_diabetes", "pct_hypertension", "pct_ckd",
                                    "pct_prior_mi", "pct_acute_mi"]}),
    ("explain_ecg", {"patient": {"age": 70, "sex": "female"}, "example_id": "synthetic_typical_mi_shift",
                     "timing": "after_onset"}),
    ("explain_ecg", {"patient": {"age": 70, "sex": "female"}}),
    ("render_figure", {"figure_id": "crude_vs_adjusted"}), ("render_figure", {"figure_id": "latent_audit"}),
    ("trace_claim", {"query": "ECG"}), ("server_status", {}),
]


def error_body(result) -> dict:
    text = result.content[0].text
    return json.loads(text[text.index("{"):])["error"]


class EgressUnit(unittest.TestCase):
    def codes(self, payload):
        return {v["code"] for v in egress.check(payload)}

    def test_planted_values_are_caught(self):
        self.assertEqual(self.codes({"result": {"subject_id": 1}}), {"denylisted_key"})
        self.assertEqual(self.codes({"rows": [{"HADM_ID": "x"}]}), {"denylisted_key"})
        self.assertEqual(self.codes({"x": 12345678}), {"large_integer"})
        self.assertEqual(self.codes({"x": 2.5e7}), {"large_integer"})
        self.assertEqual(self.codes({"s": "patient 10000032 was seen"}), {"identifier_in_text"})
        self.assertEqual(self.codes({"s": "admission 29079034"}), {"identifier_in_text"})
        self.assertEqual(self.codes({"s": "ECG 40689238"}), {"identifier_in_text"})
        self.assertEqual(self.codes({"s": "seen on 2180-04-27"}), {"shifted_date"})
        self.assertEqual(self.codes({"s": "2155/12/01 08:00"}), {"shifted_date"})
        self.assertEqual(self.codes({"rows": list(range(61))}), {"list_too_long"})
        self.assertEqual(self.codes({"blob": "a" * 40000}), {"payload_too_large"})
        self.assertEqual(self.codes({"10000032": 1}), {"identifier_in_text"})

    def test_legitimate_values_pass(self):
        ok = {"n": 431231, "patients": 180733, "p": 1.74e-85, "q": "8.7e-05",
              "sha256": "4e76c9575b40687e2c7014844482075e943931bf4aa19e257349e6ec3695149d",
              "built_at": "2026-11-23T05:49:50Z", "text": "430,852 admissions; RR 1.44 (95% CI 1.35-1.53)",
              "decimal": "0.12345678", "rows": list(range(60)), "manifest": "548bfc21a0d8", "flag": True}
        self.assertEqual(egress.check(ok), [])


class EgressThroughProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.ctx = build_context(cls.tmp / "audit.jsonl")
        cls.app = build_app(cls.ctx)
        ann = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

        @cls.app.tool(name="leaky_test_tool", annotations=ann, structured_output=True)
        @audited(cls.ctx, "leaky_test_tool")
        def leaky(kind: str) -> Envelope[dict]:
            payload = {"ids": {"subject_id": 10000032}, "date": {"note": "admitted 2180-04-27"},
                       "big": {"rows": list(range(500))}, "clean": {"n": 5}}[kind]
            return Envelope[dict](tool="leaky_test_tool", result=payload, summary="test", provenance=Provenance())

        @cls.app.tool(name="echo_error_tool", annotations=ann)
        @audited(cls.ctx, "echo_error_tool")
        def echo(value: str) -> str:
            fail("INVALID_ARGUMENT", f"Unknown value {value!r}.")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, name, args):
        async def run():
            async with create_connected_server_and_client_session(self.app._mcp_server) as s:
                return await s.call_tool(name, args)
        return asyncio.run(run())

    def last_audit(self):
        return json.loads(self.ctx.audit.path.read_text(encoding="utf-8").splitlines()[-1])

    def test_planted_payloads_are_withheld_and_audited(self):
        for kind, code in (("ids", "denylisted_key"), ("date", "shifted_date"), ("big", "list_too_long")):
            before = self.ctx.egress_blocked
            r = self.call("leaky_test_tool", {"kind": kind})
            self.assertTrue(r.isError, kind)
            err = error_body(r)
            self.assertEqual(err["code"], "DISCLOSURE_BLOCKED")
            self.assertIn(code, err["checks_failed"])
            self.assertNotIn("10000032", r.content[0].text)
            self.assertNotIn("2180", r.content[0].text)
            self.assertEqual(self.ctx.egress_blocked, before + 1)
            self.assertEqual(self.last_audit()["outcome"], "blocked")
        self.assertFalse(self.call("leaky_test_tool", {"kind": "clean"}).isError)

    def test_error_text_echoing_an_identifier_is_withheld(self):
        r = self.call("echo_error_tool", {"value": "patient 10000032"})
        self.assertTrue(r.isError)
        self.assertEqual(error_body(r)["code"], "DISCLOSURE_BLOCKED")
        self.assertNotIn("10000032", r.content[0].text)
        r = self.call("echo_error_tool", {"value": "lateral"})
        self.assertEqual(error_body(r)["code"], "INVALID_ARGUMENT")

    def test_no_real_tool_output_is_blocked_and_all_fit_the_size_limit(self):
        async def run():
            out = []
            async with create_connected_server_and_client_session(self.app._mcp_server) as s:
                for name, args in REAL_CALLS:
                    out.append((name, await s.call_tool(name, args)))
            return out
        for name, r in asyncio.run(run()):
            self.assertFalse(r.isError, f"{name}: {r.content[0].text[:200] if r.isError else ''}")
            self.assertEqual(egress.check(r.structuredContent), [], name)
            size = len(json.dumps(r.structuredContent, separators=(",", ":")).encode())
            self.assertLess(size, egress.MAX_PAYLOAD_BYTES, name)

    def test_status_reports_the_guard_without_absolute_paths(self):
        r = self.call("server_status", {})
        self.assertFalse(r.isError)
        res = r.structuredContent["result"]
        self.assertTrue(res["egress_guard"]["active"])
        self.assertEqual(res["egress_guard"]["max_payload_bytes"], egress.MAX_PAYLOAD_BYTES)
        text = json.dumps(r.structuredContent)
        self.assertNotIn(str(ROOT), text)
        self.assertNotIn(str(self.tmp), text)
        self.assertTrue(res["audit"]["path"].startswith("<outside repo>/"))


if __name__ == "__main__":
    unittest.main()
