"""Filesystem locations used by the server. Nothing here is user-controlled at runtime."""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SCM_ROOT", PKG.parent)).resolve()   # the SCM repository
MANIFEST = PKG / "manifest.json"                                  # committed: what the server may read
SQL_DIR = PKG / "sql"                                             # reviewed query templates
STATE_DIR = ROOT / "data" / "local_mcp"                          # git-ignored local state
AUDIT_LOG = STATE_DIR / "audit.jsonl"


def rel(path: Path) -> str:
    """Repository-relative POSIX path. Never send absolute local paths to a client: files outside the repo are
    shown by name only."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return f"<outside repo>/{p.name}"
