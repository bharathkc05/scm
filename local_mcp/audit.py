"""Hash-chained JSONL audit log (local only, under the git-ignored data/ folder).

Each line stores the tool, its arguments (sensitive ones hashed), the outcome, the duration and a hash of the
output (never the output itself), plus the previous line's hash. Editing or deleting any line breaks the chain.
Several server processes (one per client) can append safely: writes are serialised with a file lock and each
entry chains to the file's current last line.
"""
import hashlib
import json
import os
import time
from pathlib import Path

GENESIS = "0" * 64


def _default(o):
    return o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o)


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _digest(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


class _FileLock:
    """Cross-process exclusive lock on a sidecar file (msvcrt on Windows, fcntl elsewhere)."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, "a+b")
        if os.name == "nt":
            import msvcrt
            self.f.seek(0)
            while True:
                try:
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
        else:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            import msvcrt
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _last_hash(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - 65536))
        lines = f.read().splitlines()
    for line in reversed(lines):
        if line.strip():
            return json.loads(line)["hash"]
    return GENESIS


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(".lock")
        self.session_entries = 0

    def record(self, tool: str, args: dict | None, outcome: str, duration_ms: float,
               output: str | None = None, redact: tuple[str, ...] = ()) -> str:
        safe_args = {k: ({"sha256": _digest(v)} if k in redact else v) for k, v in (args or {}).items()}
        with _FileLock(self.lock_path):
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": os.getpid(),
                "tool": tool,
                "args": json.loads(_canonical(safe_args)),
                "outcome": outcome,
                "duration_ms": round(duration_ms, 2),
                "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest() if output is not None else None,
                "prev": _last_hash(self.path),
            }
            entry["hash"] = _digest(entry)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(_canonical(entry) + "\n")
        self.session_entries += 1
        return entry["hash"]


def verify(path: Path) -> dict:
    """Walk the chain. Returns ok, number of valid entries and the first bad line (1-based) if any."""
    path = Path(path)
    prev, n = GENESIS, 0
    if not path.exists():
        return {"ok": True, "entries": 0, "first_bad_line": None}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            stored = entry.pop("hash")
        except (ValueError, KeyError):
            return {"ok": False, "entries": n, "first_bad_line": i}
        if entry.get("prev") != prev or _digest(entry) != stored:
            return {"ok": False, "entries": n, "first_bad_line": i}
        prev, n = stored, n + 1
    return {"ok": True, "entries": n, "first_bad_line": None}
