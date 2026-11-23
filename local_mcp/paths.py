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
    """Repository-relative POSIX path (never send absolute local paths to a client)."""
    return Path(path).resolve().relative_to(ROOT).as_posix()
