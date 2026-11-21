"""Phase 4 — causal graphs as code, adjustment-set checks and implied independencies.

Nodes: A age, S sex, D diabetes, H hypertension, K CKD, P prior MI, M acute MI at admission,
       F ECG (one multivariate block: 95 features / 64 latents), R has a feature ECG (selection).
Latent nodes start with "U".

Writes ADJUSTMENT_TABLE.csv and IMPLIED_INDEPENDENCIES.csv into this folder.
"""
from itertools import combinations

import networkx as nx
import pandas as pd

from roadmap.config import PHASE_DIRS

BASE = ["A", "S"]
CHRONIC = ["D", "H", "K"]


def g0() -> nx.DiGraph:
    """Roadmap draft: baseline factors -> MI and ECG; MI -> ECG and selection. No D-H edge."""
    g = nx.DiGraph()
    for b in BASE:
        g.add_edges_from((b, v) for v in ["D", "H", "K", "P", "M", "F", "R"])
    g.add_edges_from([("D", "K"), ("H", "K"), ("D", "P"), ("H", "P"), ("K", "P")])
    g.add_edges_from((v, "M") for v in ["D", "H", "K", "P"])
    g.add_edges_from((v, "F") for v in ["D", "H", "K", "P"])        # e.g. HTN -> LVH voltage, old MI -> Q waves
    g.add_edges_from((v, "R") for v in ["D", "H", "K", "P"])        # sicker patients get more ECGs
    g.add_edges_from([("M", "F"), ("M", "R")])
    return g


def variant(name: str) -> nx.DiGraph:
    g = g0()
    if name == "G0_roadmap":
        pass
    elif name == "G_D_to_H":
        g.add_edge("D", "H")
    elif name == "G_H_to_D":
        g.add_edge("H", "D")
    elif name == "G_DH_common_cause":                   # metabolic syndrome / obesity
        g.add_edges_from([("U_dh", "D"), ("U_dh", "H")])
    elif name == "G_revised":                           # Phase 4 revision: D -> H plus shared metabolic cause
        g.add_edge("D", "H")
        g.add_edges_from([("U_dh", "D"), ("U_dh", "H")])
    elif name == "G_P_before_K":                        # prior MI -> CKD instead of CKD -> prior MI
        g.remove_edge("K", "P")
        g.add_edge("P", "K")
    elif name == "G_lifestyle_U":                       # smoking, lipids, genetics: affect disease and MI, not ECG
        g.add_edges_from([("U_life", v) for v in ["D", "H", "P", "M"]])
    elif name == "G_lifestyle_U_to_ECG":                # ... and also the ECG (e.g. BMI lowers QRS voltage)
        g.add_edges_from([("U_life", v) for v in ["D", "H", "P", "M", "F"]])
    elif name == "G_severity_selection":                # unmeasured acuity affects ECG recording and the ECG
        g.add_edges_from([("U_sev", "R"), ("U_sev", "F")])
    else:
        raise ValueError(name)
    return g


GRAPHS = ["G0_roadmap", "G_revised", "G_D_to_H", "G_H_to_D", "G_DH_common_cause", "G_P_before_K",
          "G_lifestyle_U", "G_lifestyle_U_to_ECG", "G_severity_selection"]

# (layer, exposure, outcome, label, adjustment set, selection nodes conditioned on by design)
QUERIES = [
    ("L1", "D", "M", "roadmap", {"A", "S"}, set()),
    ("L1", "D", "M", "alt_ordering", {"A", "S", "H"}, set()),
    ("L1", "H", "M", "roadmap", {"A", "S"}, set()),
    ("L1", "H", "M", "alt_ordering", {"A", "S", "D"}, set()),
    ("L1", "K", "M", "roadmap", {"A", "S", "D", "H"}, set()),
    ("L1", "K", "M", "alt_ordering", {"A", "S", "D", "H", "P"}, set()),
    ("L1", "P", "M", "roadmap", {"A", "S", "D", "H", "K"}, set()),
    ("L2", "M", "F", "roadmap", {"A", "S", "D", "H", "K", "P"}, {"R"}),
]


def check(g: nx.DiGraph, x: str, y: str, z: set, sel: set) -> dict:
    latent = {v for v in g if v.startswith("U")}
    assert not z & latent
    no_desc = not (z & nx.descendants(g, x))
    gx = g.copy()
    gx.remove_edges_from(list(g.out_edges(x)))
    backdoor = nx.is_d_separator(gx, {x}, {y}, z | sel)
    # Selection recoverability (s-backdoor): outcome independent of selection given exposure + Z
    s_ok = True if not sel else nx.is_d_separator(g, {y}, sel, {x} | z)
    return {"no_descendant_of_exposure": no_desc, "blocks_backdoor": backdoor,
            "selection_ignorable": s_ok, "valid": no_desc and backdoor and s_ok}


def adjustment_table() -> pd.DataFrame:
    rows = []
    for layer, x, y, label, z, sel in QUERIES:
        for name in GRAPHS:
            g = variant(name)
            r = check(g, x, y, z, sel)
            rows.append({"layer": layer, "exposure": x, "outcome": y, "set_label": label,
                         "adjustment_set": ",".join(sorted(z)), "conditioned_selection": ",".join(sorted(sel)),
                         "graph": name, **r})
    return pd.DataFrame(rows)


def implied_independencies(name: str) -> list[dict]:
    """Local Markov basis over observed nodes: V indep. of its non-descendant non-parents | parents."""
    g = variant(name)
    observed = [v for v in g if not v.startswith("U")]
    out = []
    for v in observed:
        parents = set(g.predecessors(v))
        if parents & {u for u in g if u.startswith("U")}:
            continue                                    # latent parent: no observable local statement
        others = set(observed) - nx.descendants(g, v) - parents - {v}
        for w in sorted(others):
            if not nx.is_d_separator(g, {v}, {w}, parents):
                continue
            testable = not ({"R", "F"} & {v, w} and "R" in {v, w})   # F unobserved when R = 0
            out.append({"graph": name, "independence": f"{v} _||_ {w} | {','.join(sorted(parents)) or '{}'}",
                        "x": v, "y": w, "given": ",".join(sorted(parents)), "testable": testable})
    # drop symmetric duplicates
    seen, uniq = set(), []
    for r in out:
        key = (frozenset({r["x"], r["y"]}), r["given"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


if __name__ == "__main__":
    adj = adjustment_table()
    adj.to_csv(PHASE_DIRS[4] / "ADJUSTMENT_TABLE.csv", index=False)
    imp = pd.DataFrame([r for n in GRAPHS for r in implied_independencies(n)])
    imp.to_csv(PHASE_DIRS[4] / "IMPLIED_INDEPENDENCIES.csv", index=False)
    print(adj.pivot_table(index=["exposure", "set_label", "adjustment_set"], columns="graph",
                          values="valid", aggfunc="first").to_string())
    print(imp.to_string())
