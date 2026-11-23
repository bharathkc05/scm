"""M5 acceptance: the client config templates are portable and well-formed, render for this machine, install by
merging into existing client files, and every client's config launches the server and answers all 12 tools over
stdio (same command, args, env and cwd handling as the clients)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from local_mcp import smoke
from local_mcp.configs import render

MANIFEST = json.loads((render.REPO / "local_mcp" / "manifest.json").read_text(encoding="utf-8"))
T1_PRESENT = all((render.REPO / a["path"]).exists() for a in MANIFEST["artifacts"] if a["tier"] == "T1")


def template(name: str) -> tuple[str, dict]:
    raw = (render.TEMPLATES / f"{name}.json").read_text(encoding="utf-8")
    return raw, json.loads(raw)


class Templates(unittest.TestCase):
    def test_one_portable_template_per_client(self):
        self.assertEqual(sorted(p.stem for p in render.TEMPLATES.glob("*.json")), sorted(render.CLIENTS))
        for name, spec in render.CLIENTS.items():
            raw, cfg = template(name)
            self.assertEqual(list(cfg), [spec.key], name)
            entry = cfg[spec.key][render.SERVER_NAME]
            self.assertEqual(entry["command"], "{{PYTHON}}", name)
            self.assertEqual(entry["args"], ["-X", "utf8", "{{SCM_ROOT}}/local_mcp/server.py"], name)
            self.assertEqual(entry["env"], {"SCM_ROOT": "{{SCM_ROOT}}", "PYTHONUTF8": "1"}, name)
            self.assertNotRegex(raw, r"[A-Za-z]:[/\\]|Users|/home/", f"{name}: machine path committed")

    def test_client_specific_shapes(self):
        self.assertEqual(template("vscode")[1]["servers"][render.SERVER_NAME]["type"], "stdio")
        self.assertEqual(template("claude_code")[1]["mcpServers"][render.SERVER_NAME]["type"], "stdio")
        ag = template("antigravity")[1]["mcpServers"][render.SERVER_NAME]
        self.assertEqual(ag["cwd"], "{{SCM_ROOT}}")
        self.assertIsInstance(ag["timeoutSeconds"], int)


class Render(unittest.TestCase):
    def test_paths_are_absolute_forward_slash_and_real(self):
        for name in render.CLIENTS:
            entry = render.server_entry(render.render(name))
            self.assertEqual(entry["command"], Path(sys.executable).resolve().as_posix(), name)
            self.assertEqual(Path(entry["args"][-1]).resolve(), (render.REPO / "local_mcp" / "server.py").resolve())
            self.assertEqual(Path(entry["env"]["SCM_ROOT"]).resolve(), render.REPO.resolve())
            self.assertNotIn("\\", json.dumps(entry), name)
            self.assertNotIn("{{", json.dumps(entry), name)
        cwd = render.server_entry(render.render("antigravity"))["cwd"]
        self.assertEqual(Path(cwd).resolve(), render.REPO.resolve())

    def test_unknown_client_and_overrides(self):
        with self.assertRaises(KeyError):
            render.render("notepad")
        entry = render.server_entry(render.render("cursor", python="D:/py/python.exe", root="D:/elsewhere/SCM"))
        self.assertEqual(entry["command"], "D:/py/python.exe")
        self.assertEqual(entry["args"][-1], "D:/elsewhere/SCM/local_mcp/server.py")

    def test_interpreter_check(self):
        self.assertIsNone(render.check_interpreter(sys.executable))
        self.assertIsNotNone(render.check_interpreter("C:/no/such/python.exe"))


class Install(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_merge_keeps_other_servers_and_backs_up(self):
        target = self.tmp / "claude_desktop_config.json"
        old = json.dumps({"mcpServers": {"other": {"command": "node", "args": ["x.js"]}}, "theme": "dark"})
        target.write_text(old, encoding="utf-8")
        render.install("claude_desktop", target=target)
        cfg = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(cfg["theme"], "dark")
        self.assertEqual(cfg["mcpServers"]["other"], {"command": "node", "args": ["x.js"]})
        self.assertEqual(cfg["mcpServers"][render.SERVER_NAME], render.server_entry(render.render("claude_desktop")))
        self.assertEqual((self.tmp / "claude_desktop_config.json.bak").read_text(encoding="utf-8"), old)
        first = target.read_text(encoding="utf-8")
        render.install("claude_desktop", target=target)              # idempotent
        self.assertEqual(target.read_text(encoding="utf-8"), first)

    def test_default_target_and_key_per_client(self):
        for name, spec in render.CLIENTS.items():
            if spec.target is None:
                with self.assertRaises(ValueError):                  # outside the repo: needs an explicit target
                    render.install(name, root=self.tmp)
                continue
            path = render.install(name, root=self.tmp)
            self.assertEqual(path, self.tmp / spec.target)
            self.assertIn(render.SERVER_NAME, json.loads(path.read_text(encoding="utf-8"))[spec.key])
            self.assertFalse(path.with_name(path.name + ".bak").exists())   # nothing to back up for a new file

    def test_refuses_what_it_cannot_parse(self):
        for text in ("// JSONC comment\n{\"servers\": {}}", "[1, 2]", "{\"servers\": []}"):
            target = self.tmp / "mcp.json"
            target.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError, msg=text):
                render.install("vscode", target=target)
            self.assertEqual(target.read_text(encoding="utf-8"), text)


class LaunchEveryClient(unittest.TestCase):
    """The acceptance run: every client config, all 12 tools, the same answers."""

    @classmethod
    def setUpClass(cls):
        cls.launches = [smoke.run_entry(name, entry) for name, entry in smoke.load_entries()]

    def test_all_twelve_tools_answer_from_every_config(self):
        self.assertEqual([L.name for L in self.launches], list(render.CLIENTS))
        self.assertEqual({c[0] for c in smoke.SMOKE_CALLS}, set(smoke.EXPECTED_TOOLS))
        for L in self.launches:
            self.assertTrue(L.ok, f"{L.name}: {L.problems}\n{L.stderr_tail}")
            self.assertEqual(L.tools, sorted(smoke.EXPECTED_TOOLS))
            for c in L.calls:
                expected = "gated" if (c.tool == "cohort_summary" and not T1_PRESENT) else "ok"
                self.assertEqual(c.status, expected, f"{L.name}/{c.tool}: {c.detail}")
                if c.status == "ok":
                    self.assertLess(c.bytes, smoke.egress.MAX_PAYLOAD_BYTES)

    def test_same_server_same_answers(self):
        self.assertEqual(len({L.fingerprint for L in self.launches}), 1)

    def test_no_dependence_on_client_working_directory(self):
        by_name = {L.name: L for L in self.launches}
        self.assertNotEqual(Path(by_name["cursor"].cwd).resolve(), render.REPO.resolve())
        self.assertEqual(Path(by_name["antigravity"].cwd).resolve(), render.REPO.resolve())

    def test_cold_start_is_interactive(self):
        for L in self.launches:
            self.assertLess(L.start_ms, 15_000, L.name)


class SmokeCli(unittest.TestCase):
    def test_installed_file_quick_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mcp.json"
            render.install("cursor", target=target)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = smoke.main(["--config", str(target), "--quick"])
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("[PASS]", out.getvalue())

    def test_broken_config_fails_cleanly(self):
        entry = dict(render.server_entry(render.render("cursor")), command="C:/no/such/python.exe")
        launch = smoke.run_entry("broken", entry, smoke.QUICK_CALLS, timeout=30)
        self.assertFalse(launch.ok)
        self.assertTrue(launch.problems)
        with self.assertRaises(ValueError):
            smoke.params_from_entry({"type": "http", "command": "x"}, Path("."))


if __name__ == "__main__":
    unittest.main()
