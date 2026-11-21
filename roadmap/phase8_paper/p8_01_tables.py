"""Phase 8 — build the manuscript tables directly from phase outputs (no hand-copied numbers).

Writes TABLES.md (Tables 1-3 + baseline-model table) and TABLE1_COHORT.csv.
"""
import json

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, REPO
from roadmap.phase7_stress_test.common import ROBUST

DOC = PHASE_DIRS[8]


def md(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal markdown table (avoids an extra dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) if str(c) != "index" else "" for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def pct(s):
    return f"{100 * s.mean():.1f}"


def table1() -> pd.DataFrame:
    l1 = pd.read_parquet(DATA_OUT / "phase3/layer1.parquet")
    l2 = pd.read_parquet(DATA_OUT / "phase3/layer2.parquet")
    l2 = l2[l2["set_primary"] == 1]
    groups = {"Layer 1: acute MI": l1[l1["acute_mi"] == 1], "Layer 1: no acute MI": l1[l1["acute_mi"] == 0],
              "Layer 2: after-onset MI": l2[l2["acute_mi"] == 1], "Layer 2: controls": l2[l2["acute_mi"] == 0]}
    rows = {}
    for g, d in groups.items():
        rows[g] = {
            "Admissions": f"{len(d):,}", "Patients": f"{d['subject_id'].nunique():,}",
            "Age, mean (SD)": f"{d['age'].mean():.1f} ({d['age'].std():.1f})", "Male, %": pct(d["sex_male"]),
            "Diabetes, %": pct(d["diabetes_primary"]), "Hypertension, %": pct(d["hypertension_primary"]),
            "CKD, %": pct(d["ckd_primary"]), "Prior MI, %": pct(d["prior_mi_primary"]),
            "First admission, %": pct(d["n_prior_admissions"] == 0), "Arrived via ED, %": pct(d["arrived_via_ed"]),
            "Chest-pain code, %": pct(d["chest_pain"]),
            "STEMI / NSTEMI / type 2, %": (f"{pct(d['mi_stemi'])} / {pct(d['mi_nstemi'])} / {pct(d['mi_type2'])}"
                                           if d["acute_mi"].all() else "-"),
            "Hours arrival to ECG, median (IQR)": (
                f"{d['hours_from_arrival'].median():.2f} ({d['hours_from_arrival'].quantile(.25):.2f}-{d['hours_from_arrival'].quantile(.75):.2f})"
                if "hours_from_arrival" in d else "-"),
        }
    return pd.DataFrame(rows)


def table2() -> pd.DataFrame:
    e = pd.read_csv(PHASE_DIRS[5] / "LAYER1_ESTIMATES.csv")
    e = e[e["estimand"] == "ATE_overlap"]
    ev = pd.read_csv(PHASE_DIRS[7] / "EVALUES.csv").dropna(subset=["label"]).set_index("label")
    att = pd.read_csv(PHASE_DIRS[5] / "LAYER1_ESTIMATES.csv")
    att = att[att["estimand"] == "ATT"].set_index("label")
    verdict = {"diabetes": "survives", "hypertension": "survives", "CKD": "fragile", "prior MI": "survives (descriptive)"}
    return pd.DataFrame({
        "Exposure": e["label"].str.split(" | ", regex=False).str[0],
        "Adjusted for (+ age spline)": e["label"].str.split(" | ", regex=False).str[1].str.replace(" (roadmap)", "", regex=False),
        "Role": e["role"],
        "Admissions (overlap)": e["n"].map("{:,}".format), "Exposed": e["exposed"].map("{:,}".format),
        "Crude RD, pp": e["crude_rd_pp"].round(2), "Adjusted RD, pp (95% CI)":
            [f"{a:.2f} ({b:.2f} to {c:.2f})" for a, b, c in zip(e["rd_pp"], e["rd_ci_low_pp"], e["rd_ci_high_pp"])],
        "RR (95% CI)": [f"{a:.2f} ({b:.2f}-{c:.2f})" for a, b, c in zip(e["rr"], e["rr_ci_low"], e["rr_ci_high"])],
        "ATT RR": [f"{att.loc[l, 'rr']:.2f}" for l in e["label"]],
        "E-value (CI)": [f"{ev.loc[l, 'evalue_point']:.2f} ({ev.loc[l, 'evalue_ci']:.2f})" for l in e["label"]],
        "Phase 7 verdict": [verdict[l.split(" |")[0]] for l in e["label"]]})


def table3() -> pd.DataFrame:
    d = pd.read_csv(PHASE_DIRS[6] / "DECONFOUNDED_ECG_EFFECTS.csv").set_index("outcome")
    sel = pd.read_csv(PHASE_DIRS[7] / "SELECTION_WEIGHTED_LAYER2.csv").set_index("outcome")
    ev = pd.read_csv(PHASE_DIRS[7] / "EVALUES.csv").dropna(subset=["outcome"]).set_index("outcome")
    rows = []
    for o in ROBUST:
        r = d.loc[o]
        rows.append({"ECG measure": o.replace("_Elev_", " ").replace("_Amp_", " ").replace("_", " "),
                     "Crude (SD)": f"{r['crude_sd']:+.2f}",
                     "Adjusted ATT (SD, 95% CI)": f"{r['att_sd']:+.2f} ({r['att_sd_ci_low']:+.2f} to {r['att_sd_ci_high']:+.2f})",
                     "NSTEMI only": f"{r['nstemi_att_sd']:+.2f}", "Chest-pain comparator": f"{r['chest_pain_comparator_att_sd']:+.2f}",
                     "Selection-weighted": f"{sel.loc[o, 'att_sd']:+.2f}",
                     "Negative control": f"{r['negative_control_att_sd']:+.2f}",
                     "E-value (CI)": f"{ev.loc[o, 'evalue_point']:.2f} ({ev.loc[o, 'evalue_ci']:.2f})"})
    return pd.DataFrame(rows)


def baseline_models() -> pd.DataFrame:
    r = json.loads((REPO / "models/baselines/results.json").read_text())["models"]
    return pd.DataFrame([{"Model (legacy, Phase H)": v["name"], "Features": v["n_features"],
                          "AUROC": round(v["metrics"]["auroc"], 3), "AUPRC": round(v["metrics"]["auprc"], 3)}
                         for v in r.values()])


if __name__ == "__main__":
    t1 = table1()
    t1.to_csv(DOC / "TABLE1_COHORT.csv")
    parts = ["# Manuscript tables (generated by p8_01_tables.py — do not edit by hand)\n",
             "## Table 1. Cohort characteristics\n", md(t1, index=True), "\n",
             "## Table 2. Layer 1 — chronic conditions and acute MI at admission (ATE in overlap region)\n",
             md(table2()), "\n",
             "## Table 3. Layer 2 — robust ECG effects of MI (ATT, control-SD units)\n",
             md(table3()), "\n",
             "## Table S1. Legacy baseline prediction models (not re-estimated; see caveats in the manuscript)\n",
             md(baseline_models()), "\n"]
    (DOC / "TABLES.md").write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))
