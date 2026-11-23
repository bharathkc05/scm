"""Manifest-driven registry: which artifacts exist on this machine, whether they are intact, and which
DuckDB views the sandbox may expose. Data possession is the access model: absent files are never served.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from local_mcp.build_manifest import content_hash
from local_mcp.paths import MANIFEST, ROOT

VIEW_NAME = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
TEMPLATE_NAME = re.compile(r"^[a-z][a-z0-9_]*__[a-z0-9_]+$")
READERS = {"parquet": "read_parquet", "csv": "read_csv_auto"}
TIERS = ("T1", "T2", "T3", "code")


class IntegrityError(Exception):
    """An artifact is missing, altered, or changed since the server started."""

    def __init__(self, paths: list[str], reason: str):
        self.paths, self.reason = paths, reason
        super().__init__(f"{reason}: {', '.join(paths)}")


@dataclass
class Artifact:
    path: str
    tier: str
    kind: str
    view: str | None
    sha256: str
    size: int
    mtime_ns: int
    abs_path: Path | None = None
    status: str = "unchecked"          # ok | missing | mismatch
    check: str = ""                    # hash | stat | stat_changed_then_hash | full_hash
    seen: tuple[int, int] | None = None  # (size, mtime_ns) when verified


class Registry:
    def __init__(self, manifest_path: Path = MANIFEST, root: Path = ROOT):
        self.root = Path(root).resolve()
        self.error: str | None = None
        self.version: str | None = None
        self.built_at: str | None = None
        self.artifacts: list[Artifact] = []
        try:
            data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.error = "manifest missing: run local_mcp/build_manifest.py"
            return
        self.version, self.built_at = data["version"], data["built_at"]
        for e in data["artifacts"]:
            a = Artifact(**e)
            if a.view is not None and not VIEW_NAME.match(a.view):
                raise ValueError(f"invalid view name in manifest: {a.view!r}")
            a.abs_path = self.root / a.path
            self.artifacts.append(a)
        self.verify(full=False)

    # ---------------------------------------------------------------- integrity
    def verify(self, full: bool = False) -> None:
        for a in self.artifacts:
            self._verify_one(a, full)

    def _verify_one(self, a: Artifact, full: bool) -> None:
        if not a.abs_path.is_file():
            a.status, a.check, a.seen = "missing", "", None
            return
        st = a.abs_path.stat()
        a.seen = (st.st_size, st.st_mtime_ns)
        if a.tier == "T1" and not full and a.seen == (a.size, a.mtime_ns):
            a.status, a.check = "ok", "stat"          # large binary file, unchanged since the manifest
            return
        if a.tier == "T1":
            a.check = "full_hash" if full else "stat_changed_then_hash"
        else:
            a.check = "hash"
        a.status = "ok" if content_hash(a.abs_path, a.kind) == a.sha256 else "mismatch"

    def tiers(self) -> dict[str, str]:
        out = {}
        for t in TIERS:
            arts = [a for a in self.artifacts if a.tier == t]
            ok = sum(a.status == "ok" for a in arts)
            out[t] = "absent" if ok == 0 else ("present" if ok == len(arts) else "partial")
        return out

    def by_status(self, status: str) -> list[str]:
        return [a.path for a in self.artifacts if a.status == status]

    def changed_since_start(self) -> list[str]:
        changed = []
        for a in self.artifacts:
            if a.status != "ok":
                continue
            if not a.abs_path.is_file():
                changed.append(a.path)
                continue
            st = a.abs_path.stat()
            if (st.st_size, st.st_mtime_ns) != a.seen:
                changed.append(a.path)
        return changed

    def assert_fresh(self) -> None:
        changed = self.changed_since_start()
        if changed:
            raise IntegrityError(changed, "changed since the server started; re-run build_manifest.py and restart")

    # ---------------------------------------------------------------- DuckDB inputs
    def queryable(self) -> list[Artifact]:
        return [a for a in self.artifacts if a.status == "ok" and a.kind in READERS]

    def allowlist(self) -> list[str]:
        return [a.abs_path.resolve().as_posix() for a in self.queryable()]

    def views(self) -> list[tuple[str, str, str]]:
        return [(a.view, a.abs_path.resolve().as_posix(), READERS[a.kind]) for a in self.queryable() if a.view]

    # ---------------------------------------------------------------- text artifacts
    def _artifact(self, rel: str) -> Artifact:
        for a in self.artifacts:
            if a.path == rel:
                return a
        raise IntegrityError([rel], "not in the manifest")

    def text(self, rel: str) -> str:
        a = self._artifact(rel)
        if a.status != "ok":
            raise IntegrityError([rel], f"artifact is {a.status}")
        self.assert_fresh()
        return a.abs_path.read_text(encoding="utf-8")

    def template(self, name: str) -> str:
        if not TEMPLATE_NAME.match(name):
            raise ValueError(f"invalid template name {name!r}")
        return self.text(f"local_mcp/sql/{name}.sql")

    def raw_mimic_registered(self) -> bool:
        return any("data/raw" in a.path.lower() or a.path.lower().endswith(".duckdb") for a in self.artifacts)
