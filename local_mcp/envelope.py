"""Response envelope shared by every tool, and structured tool errors.

Every successful result carries a plain-language summary, caveats, assumptions, provenance and disclosure
info, so the client's LLM receives the interpretation together with the numbers.
Errors are raised as ToolError with a JSON body; MCP returns them as isError results the LLM can act on.
"""
import json
from typing import Generic, NoReturn, TypeVar

from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

T = TypeVar("T")

ERROR_CODES = ("INVALID_ARGUMENT", "DATA_NOT_PRESENT", "INTEGRITY_MISMATCH", "DISCLOSURE_BLOCKED",
               "NOT_SUPPORTED", "INTERNAL")


class ArtifactRef(BaseModel):
    path: str
    sha256: str


class Provenance(BaseModel):
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    manifest_version: str | None = None
    phase: int | None = None


class Assumption(BaseModel):
    id: str
    text: str


class Disclosure(BaseModel):
    tier: str = Field("none", description="none | aggregate | aggregate_from_patient_level | model_output")
    suppressed_cells: int = 0


class Envelope(BaseModel, Generic[T]):
    ok: bool = True
    tool: str
    result: T
    summary: str = Field(description="One plain-language sentence; repeat it rather than paraphrasing numbers.")
    caveats: list[str] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    provenance: Provenance
    disclosure: Disclosure = Field(default_factory=Disclosure)


def fail(code: str, message: str, **extra) -> NoReturn:
    if code not in ERROR_CODES:
        raise ValueError(f"unknown error code {code}")
    raise ToolError(json.dumps({"ok": False, "error": {"code": code, "message": message, **extra}}))
