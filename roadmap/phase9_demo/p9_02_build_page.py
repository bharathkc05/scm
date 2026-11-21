"""Phase 9, step 2 — build the self-contained explainer page (parameters + explainer.js inlined, no network)."""
import json

from roadmap.config import PHASE_DIRS

DOC = PHASE_DIRS[9]

if __name__ == "__main__":
    params = json.loads((DOC / "demo_params.json").read_text())
    page = (DOC / "template.html").read_text(encoding="utf-8")
    page = page.replace("/*__PARAMS__*/", "window.DEMO_PARAMS = " + json.dumps(params) + ";")
    page = page.replace("/*__EXPLAINER__*/", (DOC / "explainer.js").read_text(encoding="utf-8"))
    (DOC / "ecg_explainer.html").write_text(page, encoding="utf-8")
    print(f"wrote {DOC / 'ecg_explainer.html'} ({len(page) / 1024:.0f} KB)")
