"""Network guard and hash-chained audit log."""
import json
import shutil
import socket
import tempfile
import unittest
from pathlib import Path

from local_mcp import audit, guards


class NetworkGuard(unittest.TestCase):
    def setUp(self):
        guards.install()

    def tearDown(self):
        guards.uninstall()

    def test_non_loopback_connect_is_blocked_without_a_real_attempt(self):
        before = guards.state()["blocked_attempts"]
        s = socket.socket()
        try:
            with self.assertRaises(guards.NetworkBlocked):
                s.connect(("203.0.113.1", 80))          # TEST-NET-3, never routable
            with self.assertRaises(guards.NetworkBlocked):
                s.connect_ex(("203.0.113.1", 80))
        finally:
            s.close()
        self.assertEqual(guards.state()["blocked_attempts"], before + 2)

    def test_dns_lookup_is_blocked(self):
        with self.assertRaises(guards.NetworkBlocked):
            socket.getaddrinfo("example.com", 443)

    def test_loopback_and_socketpair_still_work(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        cli = socket.socket()
        cli.connect(srv.getsockname())
        a, b = socket.socketpair()        # asyncio on Windows needs this
        for s in (srv, cli, a, b):
            s.close()
        self.assertTrue(guards.state()["active"])


class AuditChain(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = audit.AuditLog(self.tmp / "audit.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_chain_verifies_and_breaks_on_edit(self):
        for i in range(3):
            self.log.record("server_status", {"verify_full": False}, "ok", 1.0 + i, f"out{i}")
        self.assertEqual(audit.verify(self.log.path), {"ok": True, "entries": 3, "first_bad_line": None})
        lines = self.log.path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[1])
        entry["outcome"] = "error"
        lines[1] = json.dumps(entry)
        self.log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertEqual(audit.verify(self.log.path)["first_bad_line"], 2)

    def test_deleting_a_line_breaks_the_chain(self):
        for i in range(3):
            self.log.record("t", {}, "ok", 1.0, f"o{i}")
        lines = self.log.path.read_text(encoding="utf-8").splitlines()
        self.log.path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
        self.assertFalse(audit.verify(self.log.path)["ok"])

    def test_sensitive_args_are_hashed_and_output_is_not_stored(self):
        self.log.record("explain_ecg", {"ecg_features": {"ST_Elev_I": 0.01}, "timing": "after_onset"},
                        "ok", 2.0, "secret-output", redact=("ecg_features",))
        entry = json.loads(self.log.path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(set(entry["args"]["ecg_features"]), {"sha256"})
        self.assertEqual(entry["args"]["timing"], "after_onset")
        self.assertNotIn("secret-output", self.log.path.read_text(encoding="utf-8"))

    def test_a_second_writer_continues_the_same_chain(self):
        other = audit.AuditLog(self.log.path)          # e.g. a second client's server process
        self.log.record("a", {}, "ok", 1.0, "x")
        other.record("b", {}, "ok", 1.0, "y")
        self.log.record("c", {}, "ok", 1.0, "z")
        self.assertEqual(audit.verify(self.log.path), {"ok": True, "entries": 3, "first_bad_line": None})


if __name__ == "__main__":
    unittest.main()
