"""End-to-end over stdio, exactly as an MCP client launches the server.

1. The official Python MCP client initializes, lists tools and calls server_status.
2. A raw JSON-RPC exchange checks that every byte the server writes to stdout is protocol (no stray prints).
"""
import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import LATEST_PROTOCOL_VERSION

SERVER = Path(__file__).resolve().parents[1] / "server.py"
EXPECTED_TOOLS = ["study_overview", "describe_variable", "get_risk_factor_effect", "get_ecg_effect",
                  "get_subgroup_effects", "robustness_card", "confounding_what_if", "cohort_summary", "explain_ecg",
                  "render_figure", "trace_claim", "server_status", "counterfactual_ecg", "clinical_what_if",
                  "counterfactual_validation"]
CMD = [sys.executable, "-X", "utf8", str(SERVER)]
ENV = {**os.environ, "PYTHONUTF8": "1"}


class StdioClient(unittest.TestCase):
    def test_initialize_list_and_call_server_status(self):
        async def run():
            with open(os.devnull, "w") as err:
                async with stdio_client(StdioServerParameters(command=CMD[0], args=CMD[1:], env=ENV), errlog=err) as (r, w):
                    async with ClientSession(r, w) as s:
                        init = await s.initialize()
                        tools = (await s.list_tools()).tools
                        res = await s.call_tool("server_status", {})
                        bad = await s.call_tool("server_status", {"verify_full": "not-a-bool"})
                        ecg = await s.call_tool("get_ecg_effect", {"measure": "lateral ST"})
                        card_err = await s.call_tool("robustness_card", {"result_id": "hypertension"})
                        fig = await s.call_tool("render_figure", {"figure_id": "st_t_effects"})
                        return init, tools, res, bad, ecg, card_err, fig

        init, tools, res, bad, ecg, card_err, fig = asyncio.run(run())
        self.assertFalse(fig.isError)
        self.assertEqual([c.type for c in fig.content], ["image", "text"])
        self.assertEqual(fig.structuredContent["result"]["figure_id"], "st_t_effects")
        self.assertEqual(init.serverInfo.name, "scm-causal-mi")
        self.assertIn("aggregate", init.instructions)
        self.assertEqual(sorted(t.name for t in tools), sorted(EXPECTED_TOOLS))
        for tool in tools:
            self.assertTrue(tool.annotations.readOnlyHint, tool.name)
            self.assertFalse(tool.annotations.openWorldHint, tool.name)
            self.assertIsNotNone(tool.outputSchema, tool.name)
        self.assertFalse(ecg.isError)
        self.assertEqual(ecg.structuredContent["result"]["rows"][0]["measure"], "ST_lateral_mean")
        self.assertTrue(card_err.isError)                 # structured error the LLM can act on
        text = card_err.content[0].text                   # FastMCP prefixes "Error executing tool <name>: "
        self.assertEqual(json.loads(text[text.index("{"):])["error"]["code"], "INVALID_ARGUMENT")

        self.assertFalse(res.isError)
        body = res.structuredContent
        self.assertTrue(body["ok"])
        r = body["result"]
        self.assertEqual(r["transport"], "stdio")
        self.assertTrue(r["network_guard"]["active"])
        self.assertTrue(r["duckdb_sandbox"]["config_locked"])
        self.assertFalse(r["duckdb_sandbox"]["external_access"])
        self.assertTrue(all(r["duckdb_sandbox"]["self_test"].values()), r["duckdb_sandbox"]["self_test"])
        self.assertFalse(r["raw_mimic_registered"])
        self.assertEqual(r["integrity"]["mismatched"], [])
        self.assertTrue(r["audit"]["chain_ok"])
        self.assertIn("healthy", body["summary"])
        text = json.loads(res.content[0].text)          # fallback copy for clients without structuredContent
        self.assertEqual(text["result"]["server"], r["server"])

        self.assertTrue(bad.isError)                     # schema validation rejects a non-boolean


class StdoutPurity(unittest.TestCase):
    def test_every_stdout_line_is_json_rpc(self):
        proc = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=ENV)
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [lines.put(l) for l in iter(proc.stdout.readline, b"")], daemon=True).start()

        def send(msg):
            proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
            proc.stdin.flush()

        def until(msg_id, timeout=60):
            while True:
                raw = lines.get(timeout=timeout).decode("utf-8").strip()
                if not raw:
                    continue
                obj = json.loads(raw)                    # fails the test on any non-JSON output
                self.assertEqual(obj.get("jsonrpc"), "2.0")
                if obj.get("id") == msg_id:
                    return obj

        try:
            send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": LATEST_PROTOCOL_VERSION, "capabilities": {},
                             "clientInfo": {"name": "stdout-purity-test", "version": "0"}}})
            self.assertIn("result", until(1))
            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            self.assertEqual(sorted(t["name"] for t in until(2)["result"]["tools"]), sorted(EXPECTED_TOOLS))
            send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "server_status", "arguments": {}}})
            self.assertFalse(until(3)["result"].get("isError", False))
        finally:
            proc.stdin.close()
            proc.wait(timeout=30)
            proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
