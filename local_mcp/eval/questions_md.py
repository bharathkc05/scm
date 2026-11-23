"""Write QUESTIONS.md from extended.json, so the readable list and the graded set never drift apart.

    python -X utf8 local_mcp/eval/questions_md.py

Edit the questions in extended.json, then run this. tests/test_eval.py fails if QUESTIONS.md is out of date.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

from local_mcp.eval import grading  # noqa: E402

OUT = Path(__file__).resolve().parent / "QUESTIONS.md"


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _tools(q: dict) -> str:
    names = []
    for exp in q.get("expect_tools", []):
        alts = exp["tool"] if isinstance(exp["tool"], list) else [exp["tool"]]
        names.append(" or ".join(f"`{n}`" for n in alts))
    return "; ".join(names) or "none required"


def render() -> str:
    cats = grading.categories()
    questions = grading.extended()
    by_cat = {k: [q for q in questions if q["category"] == k] for k in cats}
    lines = [
        "# Agent evaluation questions",
        "",
        f"{len(questions)} questions in {len(cats)} categories for testing how an AI client answers questions about the "
        "causal MI study through the `scm-causal-mi` MCP server. They are phrased as an end user would ask them.",
        "",
        "- **A good answer** is the rubric for a human reviewer. Numbers in it are the server's current values.",
        "- **Tools** lists the calls a good client makes. A client may use more.",
        "- **Automatic grading.** `python -X utf8 local_mcp/eval/run_claude.py --set extended` asks every question in "
        "headless Claude Code and grades each answer. It costs about 12 cents per question on your Claude login, about "
        "$11 for all 90. An answer passes when four things hold:",
        "    - the expected tools were called;",
        "    - the expected values appear in the answer;",
        "    - the required phrases are present;",
        "    - nothing unsafe leaked.",
        "- **Review the failures.** Phrase checks are deliberately lenient, so read each failure against its rubric "
        "before blaming the client.",
        "- **Where the checks live.** The questions and their machine-readable checks are in "
        "[extended.json](extended.json). This file is generated from it by `questions_md.py`. The 15 core questions "
        "and 8 red-team prompts from milestone M6 are in [golden.json](golden.json) and "
        "[redteam.json](redteam.json).",
        "",
        "| Category | Questions | IDs |",
        "|---|---|---|",
    ]
    for i, (key, cat) in enumerate(cats.items(), start=1):
        qs = by_cat[key]
        lines.append(f"| {i}. {cat['title']} | {len(qs)} | {qs[0]['id']}-{qs[-1]['id']} |")
    for i, (key, cat) in enumerate(cats.items(), start=1):
        lines += ["", f"## {i}. {cat['title']}", "", cat["about"], "",
                  "| ID | Question | A good answer | Tools |", "|---|---|---|---|"]
        for q in by_cat[key]:
            note = " *(needs the patient-level files)*" if q.get("requires_t1") else ""
            lines.append(f"| {q['id']} | {_cell(q['question'])} | {_cell(q['rubric'])}{note} | {_tools(q)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(grading.extended())} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
