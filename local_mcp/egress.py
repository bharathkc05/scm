"""Egress guard: the last check on every tool response before it leaves the server process.

Tools are written never to emit patient-level data; this is the second line of defence. A response is withheld
(DISCLOSURE_BLOCKED) if it contains any of:
  * a denylisted key (MIMIC identifiers and event timestamps),
  * an integer of 10,000,000 or more (the range of MIMIC subject, admission and ECG identifiers),
  * an 8-digit identifier-shaped number inside text (subject 1xxxxxxx, admission 2xxxxxxx, ECG study 4xxxxxxx),
  * a MIMIC shifted date (years 2100-2209),
  * a list longer than MAX_LIST_ITEMS, or a JSON payload larger than MAX_PAYLOAD_BYTES.
Images are excluded from the scan: they are fixed manifest assets verified by hash, not data-derived text.
Violations are reported by code and location only, never by value.
"""
import json
import math
import re

MAX_PAYLOAD_BYTES = 32 * 1024
MAX_LIST_ITEMS = 60
ID_MIN = 10_000_000
DENY_KEYS = {"subject_id", "subjectid", "hadm_id", "study_id", "stay_id", "note_id", "ecg_time", "admittime",
             "dischtime", "edregtime", "edouttime", "deathtime", "charttime", "storetime", "dod", "anchor_year",
             "waveform_path"}
ID_IN_TEXT = re.compile(r"(?<![\w.])[124]\d{7}(?![\w.])")
SHIFTED_DATE = re.compile(r"(?<!\d)2(?:1\d{2}|20\d)[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])")


def scan_text(s: str, where: str = "$") -> list[dict]:
    out = []
    if ID_IN_TEXT.search(s):
        out.append({"code": "identifier_in_text", "at": where})
    if SHIFTED_DATE.search(s):
        out.append({"code": "shifted_date", "at": where})
    return out


def check(payload) -> list[dict]:
    """Return a list of violations ({code, at}); empty means the payload may leave."""
    violations: list[dict] = []

    def walk(node, where):
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).lower() in DENY_KEYS:
                    violations.append({"code": "denylisted_key", "at": f"{where}.{k}"})
                violations.extend(scan_text(str(k), f"{where}.<key>"))
                walk(v, f"{where}.{k}")
        elif isinstance(node, (list, tuple)):
            if len(node) > MAX_LIST_ITEMS:
                violations.append({"code": "list_too_long", "at": where})
            for i, v in enumerate(node):
                walk(v, f"{where}[{i}]")
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            if math.isfinite(node) and abs(node) >= ID_MIN and float(node).is_integer():
                violations.append({"code": "large_integer", "at": where})
        elif isinstance(node, str):
            violations.extend(scan_text(node, where))

    walk(payload, "$")
    size = len(json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8"))
    if size > MAX_PAYLOAD_BYTES:
        violations.append({"code": "payload_too_large", "at": f"$ ({size} bytes)"})
    return violations


def payload_of(out) -> object:
    """The part of a tool result that would reach the client, minus images."""
    if hasattr(out, "structuredContent") and hasattr(out, "content"):        # CallToolResult
        return {"structured": out.structuredContent,
                "text": [c.text for c in out.content if getattr(c, "type", None) == "text"]}
    if hasattr(out, "model_dump"):
        return out.model_dump(mode="json")
    return out
