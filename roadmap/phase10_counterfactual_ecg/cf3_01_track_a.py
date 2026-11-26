"""Phase 10, CF3 — Track A: feature-level individual counterfactuals, validated on within-patient pairs.

Structural model per measure (the 16 robust measures of Phases 6-7), fitted on the Phase 10 TRAINING split only:
    no MI : y = m0(L) + u              m0 = OLS on training controls (one ECG per never-MI patient)
    MI    : y = m1(L, type) + u        m1 = OLS on the earliest QC-passing during-MI ECG of each training episode,
                                            with MI-type dummies (NSTEMI, STEMI by territory, type 2, other)
    L = age spline, sex, diabetes, hypertension, CKD, prior MI (Phase 5 design and specification).
Counterfactual (abduction, action, prediction): u = y_obs - m_obs(L); y_cf = m_cf(L) + rho * u.
    A1  rho = 1 (pre-specified, the additive model: y_cf = y_obs -/+ tau(L, type))
    A2  rho per measure and direction, fitted on pair_val only (how much of the patient's deviation persists)
Baselines: "no change" (copy the other ECG) and "population mean" (m_cf(L), today's explain_ecg).
Intervals: split conformal, 90%, calibrated on pair_val residuals; evaluated on pair_test.

Directions on each pair: add MI (baseline ECG -> during-MI ECG) and remove MI (during-MI ECG -> baseline ECG).
Primary set: pair_test, after-onset context, before baseline not after an earlier MI, both ECGs pass QC.

Outputs (committed, aggregate): CF3_PAIR_VALIDATION.csv, CF3_SUBGROUPS.csv, cf3_01_track_a.json
Local (git-ignored):           data/roadmap/phase10/cf3_predictions.parquet, cf3_track_a_model.pkl
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PHASE_DIRS, SEED
from roadmap.estimators import Design, ols, ols_predict, patient_bootstrap_ratio
from roadmap.phase5_estimation.queries import L2_COVS
from roadmap.phase7_stress_test.common import ROBUST, spec

OUT = DATA_OUT / "phase10"
DOC = PHASE_DIRS[10]
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LATERAL = ["I", "aVL", "V5", "V6"]
NON_AVR = [l for l in LEADS if l != "aVR"]
ST_MAX_ABS_MV = 2.0                     # beat-wise ECGdeli ST beyond 2 mV is an artefact (CF2) -> missing
ST_T = [m for m in ROBUST if not m.startswith("Q_")]   # 14 ST/T measures (gate CF-G1)
TYPES = ["nstemi", "stemi_anterior", "stemi_inferior", "stemi_other", "type2", "other"]   # nstemi = reference
ALPHA = 0.10
REPS = 1000
COV_MAP = {"age_at_ecg": "age", "diabetes": "diabetes_primary", "hypertension": "hypertension_primary",
           "ckd": "ckd_primary", "prior_mi": "prior_mi_primary"}


# ----------------------------------------------------------------------------- data
def measures() -> pd.DataFrame:
    """The 16 measures for every QC-passing MIMIC ECG, cleaned and winsorized like Phase 3."""
    idx = pd.read_parquet(OUT / "beats_index.parquet", columns=["record_id", "dataset", "qc_pass"])
    keep = idx[(idx.dataset == "mimic") & idx.qc_pass].record_id
    need = sorted({f"ST_Elev_{l}" for l in LEADS} | {m for m in ROBUST if m in ("Q_Amp_I", "Q_Amp_aVL")}
                  | {m for m in ROBUST if m.startswith("T_Amp")})
    f = pd.read_parquet(OUT / "ecgdeli_features.parquet", columns=["record_id", "dataset"] + need)
    f = f[(f.dataset == "mimic") & f.record_id.isin(keep)].drop(columns="dataset").set_index("record_id")
    st = [f"ST_Elev_{l}" for l in LEADS]
    f[st] = f[st].where(f[st].abs() <= ST_MAX_ABS_MV)
    f["ST_lateral_mean"] = f[[f"ST_Elev_{l}" for l in LATERAL]].mean(axis=1)
    f["ST_max_depression"] = f[[f"ST_Elev_{l}" for l in NON_AVR]].min(axis=1)
    bounds = json.loads((PHASE_DIRS[3] / "winsorization_bounds.json").read_text())["bounds"]
    out = f[ROBUST].copy()
    for m in ROBUST:
        out[m] = out[m].clip(*bounds[m])
    return out


def cf_type(mi_kind: pd.Series, territory: pd.Series) -> pd.Series:
    t = mi_kind.astype(object).copy()
    st = mi_kind == "stemi"
    t[st] = "stemi_" + territory[st].map(lambda x: x if x in ("anterior", "inferior") else "other").astype(str)
    return t.replace({"other_unspecified": "other"})


def episode_covs(ep: pd.DataFrame) -> pd.DataFrame:
    e = ep[["hadm_id", "subject_id", "age", "sex_male"] + L2_COVS[1:] + ["mi_kind", "territory"]].copy()
    e[["sex_male"] + L2_COVS[1:]] = e[["sex_male"] + L2_COVS[1:]].astype(float).fillna(0)
    e["cf_type"] = cf_type(e.mi_kind, e.territory)
    return e


def training_rows(meas: pd.DataFrame, ep: pd.DataFrame, idx: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    idx = idx[idx.study_id.isin(meas.index) & (idx.subject_id.map(split) == "train")]
    # MI rows: earliest QC-passing during-MI ECG of each training episode, with the episode's covariates and type
    dur = idx[idx.role.str.contains("during")].sort_values("ecg_time").drop_duplicates("during_hadm_id")
    mi = dur[["study_id", "during_hadm_id"]].merge(episode_covs(ep), left_on="during_hadm_id", right_on="hadm_id")
    mi["mi"] = 1
    # Control rows: one ECG per never-MI patient, covariates as of the ECG (no admission before it = none recorded)
    c = idx[idx.role == "control"].rename(columns=COV_MAP)
    c[L2_COVS] = c[L2_COVS].astype(float).fillna(0)
    c["mi"], c["cf_type"] = 0, None
    cols = ["study_id", "subject_id", "age", "mi", "cf_type"] + L2_COVS
    rows = pd.concat([mi[cols], c[cols]], ignore_index=True)
    rows = rows.join(meas, on="study_id")
    return rows.dropna(subset=ROBUST).reset_index(drop=True)


# ----------------------------------------------------------------------------- model
class TrackA:
    def fit(self, tr: pd.DataFrame, design: Design | None = None, rows: np.ndarray | None = None) -> "TrackA":
        self.des = design or Design(spec("layer2"), L2_COVS).fit(tr)
        rows = np.arange(len(tr)) if rows is None else rows
        t = tr.iloc[rows]
        x = self.des.transform(t)
        y = t[ROBUST].to_numpy(float)
        c = t.mi.to_numpy() == 0
        self.b0 = ols(x[c], y[c])
        self.b1 = ols(np.column_stack([x[~c], self._dummies(t.cf_type[~c])]), y[~c])
        return self

    @staticmethod
    def _dummies(types: pd.Series) -> np.ndarray:
        return np.column_stack([(types.to_numpy() == k).astype(float) for k in TYPES[1:]])

    def m0(self, covs: pd.DataFrame) -> np.ndarray:
        return ols_predict(self.b0, self.des.transform(covs))

    def m1(self, covs: pd.DataFrame) -> np.ndarray:
        return ols_predict(self.b1, np.column_stack([self.des.transform(covs), self._dummies(covs.cf_type)]))


# ----------------------------------------------------------------------------- pairs
def pair_table(meas: pd.DataFrame, ep: pd.DataFrame, pairs: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """One row per (episode, baseline side) with both ECGs passing QC."""
    p = pairs.merge(episode_covs(ep).drop(columns=["subject_id", "mi_kind", "territory"]), on="hadm_id")
    p["cf_split"] = p.subject_id.map(split)
    rows = []
    for side in ("before", "after"):
        s = p[p[f"{side}_study_id"].notna()].copy()
        s["side"], s["base_study_id"], s["gap_days"] = side, s[f"{side}_study_id"].astype("int64"), s[f"{side}_gap_days"]
        rows.append(s)
    p = pd.concat(rows, ignore_index=True)
    p = p[p.during_study_id.isin(meas.index) & p.base_study_id.isin(meas.index)].reset_index(drop=True)
    p["primary"] = ((p.side == "before") & p.after_onset_context.astype(bool)
                    & ~p.before_follows_earlier_mi.fillna(False).astype(bool))
    keep = ["subject_id", "hadm_id", "cf_split", "side", "primary", "mi_kind", "territory", "cf_type", "gap_days",
            "during_study_id", "base_study_id", "age"] + L2_COVS
    return p[keep]


def predict(model: TrackA, p: pd.DataFrame, meas: pd.DataFrame) -> dict:
    """Arrays (n_pairs x 16) for both directions: source, target, m_source, m_target."""
    dur = meas.loc[p.during_study_id].to_numpy(float)
    base = meas.loc[p.base_study_id].to_numpy(float)
    m0, m1 = model.m0(p), model.m1(p)
    return {"add": {"src": base, "tgt": dur, "m_src": m0, "m_tgt": m1},
            "remove": {"src": dur, "tgt": base, "m_src": m1, "m_tgt": m0}}


def fit_rho(d: dict) -> np.ndarray:
    """Least-squares rho through the origin: (target - m_target) ~ rho * (source - m_source), per measure."""
    a, b = d["src"] - d["m_src"], d["tgt"] - d["m_tgt"]
    ok = np.isfinite(a) & np.isfinite(b)
    rho = np.where(ok, a * b, 0).sum(0) / np.where(ok, a * a, 0).sum(0)
    return np.clip(rho, 0, 1)


def methods(d: dict, rho: np.ndarray) -> dict:
    return {"A1_additive": d["m_tgt"] + (d["src"] - d["m_src"]),
            "A2_shrunk": d["m_tgt"] + rho * (d["src"] - d["m_src"]),
            "no_change": d["src"],
            "population_mean": d["m_tgt"]}


def conformal_q(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    r = np.abs(tgt - pred)
    q = []
    for j in range(r.shape[1]):
        v = np.sort(r[np.isfinite(r[:, j]), j])
        k = min(int(np.ceil((len(v) + 1) * (1 - ALPHA))) - 1, len(v) - 1)
        q.append(v[k])
    return np.array(q)


# ----------------------------------------------------------------------------- metrics
def evaluate(sub: pd.DataFrame, d: dict, preds: dict, q: dict, sd: np.ndarray, seed: int) -> pd.DataFrame:
    sid = sub.subject_id.to_numpy()
    tgt, src = d["tgt"], d["src"]
    err = {k: np.abs(v - tgt) / sd for k, v in preds.items()}
    rows = []
    for j, m in enumerate(ROBUST):
        ok = np.isfinite(tgt[:, j]) & np.isfinite(src[:, j])
        e = {k: v[ok, j] for k, v in err.items()}
        obs_change = (tgt - src)[ok, j]
        boot = {}
        for ref in ("no_change", "population_mean"):
            num = np.column_stack([e["A1_additive"] - e[ref], e["A2_shrunk"] - e[ref]])
            boot[ref] = patient_bootstrap_ratio(sid[ok], num, np.ones(ok.sum()), REPS, seed + j)
        for k in preds:
            r = {"measure": m, "group": "ST" if m.startswith("ST") else ("T" if m.startswith("T_") else "Q"),
                 "method": k, "n": int(ok.sum()), "mae_sd": float(e[k].mean()), "median_ae_sd": float(np.median(e[k]))}
            if k != "no_change":
                pc = (preds[k] - src)[ok, j]
                nz = obs_change != 0
                r["direction_acc"] = float(np.mean(np.sign(pc[nz]) == np.sign(obs_change[nz])))
            if k in q:
                lo, hi = preds[k][ok, j] - q[k][j], preds[k][ok, j] + q[k][j]
                r["coverage_90"] = float(np.mean((tgt[ok, j] >= lo) & (tgt[ok, j] <= hi)))
                r["interval_halfwidth_sd"] = float(q[k][j] / sd[j])
            if k in ("A1_additive", "A2_shrunk"):
                col = 0 if k == "A1_additive" else 1
                for ref in ("no_change", "population_mean"):
                    diff = float(e[k].mean() - e[ref].mean())
                    r[f"diff_vs_{ref}"] = diff
                    r[f"diff_vs_{ref}_ci_low"] = float(np.percentile(boot[ref][:, col], 2.5))
                    r[f"diff_vs_{ref}_ci_high"] = float(np.percentile(boot[ref][:, col], 97.5))
            rows.append(r)
    return pd.DataFrame(rows)


def composite(sub: pd.DataFrame, d: dict, preds: dict, q: dict, sd: np.ndarray, seed: int) -> dict:
    """Mean over the 14 ST/T measures of the absolute error (SD units), per pair; patient bootstrap of differences."""
    j = [ROBUST.index(m) for m in ST_T]
    tgt, src = d["tgt"][:, j], d["src"][:, j]
    ok = np.isfinite(tgt).all(1) & np.isfinite(src).all(1)
    e = {k: (np.abs(v[:, j] - tgt) / sd[j])[ok].mean(1) for k, v in preds.items()}
    sid = sub.subject_id.to_numpy()[ok]
    out = {"n_pairs": int(ok.sum()), "n_patients": int(len(np.unique(sid))),
           "mae_sd": {k: round(float(v.mean()), 4) for k, v in e.items()}}
    for a_i, a in enumerate(("A1_additive", "A2_shrunk")):
        for ref in ("no_change", "population_mean"):
            b = patient_bootstrap_ratio(sid, (e[a] - e[ref])[:, None], np.ones(len(sid)), REPS, seed + 10 * a_i)[:, 0]
            out[f"{a}_minus_{ref}"] = {"diff": round(float((e[a] - e[ref]).mean()), 4),
                                       "ci": [round(float(np.percentile(b, 2.5)), 4),
                                              round(float(np.percentile(b, 97.5)), 4)]}
    for k in q:
        cov = [np.mean(np.abs(preds[k][ok, jj] - d["tgt"][ok, jj]) <= q[k][jj]) for jj in j]
        out[f"coverage_90_{k}"] = {"mean": round(float(np.mean(cov)), 3),
                                   "range": [round(float(min(cov)), 3), round(float(max(cov)), 3)]}
    return out


# ----------------------------------------------------------------------------- main
def main() -> dict:
    meas = measures()
    ep = pd.read_parquet(OUT / "episodes.parquet")
    idx = pd.read_parquet(OUT / "ecg_index.parquet")
    pairs = pd.read_parquet(OUT / "pairs.parquet")
    sp = pd.read_parquet(OUT / "cf_splits.parquet")
    split = sp.set_index("subject_id").cf_split.astype(object)

    tr = training_rows(meas, ep, idx, split)
    assert not set(tr.subject_id) & set(pairs.subject_id), "pair patient in training rows"
    model = TrackA().fit(tr)
    sd = tr.loc[tr.mi == 0, ROBUST].std(ddof=1).to_numpy()

    # bootstrap refits (patients resampled, design fixed) -> model uncertainty of tau
    rng = np.random.default_rng(SEED + 30)
    codes, uniq = pd.factorize(tr.subject_id)
    members = pd.Series(np.arange(len(tr))).groupby(codes).apply(np.array).to_numpy()
    boots = []
    for _ in range(200):
        rows = np.concatenate(members[rng.integers(0, len(uniq), len(uniq))])
        boots.append(TrackA().fit(tr, design=model.des, rows=rows))

    p = pair_table(meas, ep, pairs, split)
    val, test = p[p.cf_split == "pair_val"].reset_index(drop=True), p[p.cf_split == "pair_test"].reset_index(drop=True)

    # tuning and calibration on pair_val only, separately for before and after baselines
    tuned = {}
    for side in ("before", "after"):
        v = val[val.side == side].reset_index(drop=True)
        dv = predict(model, v, meas)
        tuned[side] = {}
        for direction, d in dv.items():
            rho = fit_rho(d)
            pv = methods(d, rho)
            tuned[side][direction] = {"rho": rho, "q": {k: conformal_q(pv[k], d["tgt"]) for k in ("A1_additive",
                                                                                                    "A2_shrunk")},
                                      "n_val": len(v)}

    sets = {"primary": test[test.primary], "all_before": test[test.side == "before"],
            "all_after": test[test.side == "after"]}
    results, comp, preds_out = [], {}, []
    for s_i, (name, sub) in enumerate(sets.items()):
        sub = sub.reset_index(drop=True)
        side = "after" if name == "all_after" else "before"
        dd = predict(model, sub, meas)
        comp[name] = {}
        for direction, d in dd.items():
            t = tuned[side][direction]
            pr = methods(d, t["rho"])
            res = evaluate(sub, d, pr, t["q"], sd, SEED + 100 * s_i + (0 if direction == "add" else 50))
            res.insert(0, "direction", direction)
            res.insert(0, "set", name)
            results.append(res)
            comp[name][direction] = composite(sub, d, pr, t["q"], sd, SEED + 1000 + 10 * s_i)
            if name in ("primary", "all_after"):
                df = sub[["subject_id", "hadm_id", "side", "cf_type", "gap_days"]].copy()
                df["direction"] = direction
                for k, v in pr.items():
                    df[[f"{k}__{m}" for m in ROBUST]] = v
                df[[f"target__{m}" for m in ROBUST]] = d["tgt"]
                preds_out.append(df)
    val_res = pd.concat(results, ignore_index=True)
    val_res.round(5).to_csv(DOC / "CF3_PAIR_VALIDATION.csv", index=False)

    # subgroups on the primary set (composite ST/T MAE)
    sub_rows = []
    prim = sets["primary"].reset_index(drop=True)
    gap_q = pd.qcut(prim.gap_days, 4)
    groups = {"mi_kind": prim.mi_kind.astype(str), "cf_type": prim.cf_type.astype(str),
              "gap_quartile_days": gap_q.astype(str)}
    for gname, g in groups.items():
        for level in sorted(g.unique()):
            m = (g == level).to_numpy()
            if m.sum() < 30:
                continue
            sub = prim[m].reset_index(drop=True)
            for direction, d in predict(model, sub, meas).items():
                t = tuned["before"][direction]
                c = composite(sub, d, methods(d, t["rho"]), t["q"], sd, SEED + 2000)
                sub_rows.append({"subgroup": gname, "level": level, "direction": direction, "n_pairs": c["n_pairs"],
                                 **{f"mae_{k}": v for k, v in c["mae_sd"].items()},
                                 "A1_minus_no_change": c["A1_additive_minus_no_change"]["diff"],
                                 "A1_minus_no_change_ci": c["A1_additive_minus_no_change"]["ci"],
                                 "A1_minus_population_mean": c["A1_additive_minus_population_mean"]["diff"],
                                 "A1_minus_population_mean_ci": c["A1_additive_minus_population_mean"]["ci"],
                                 "A2_minus_population_mean": c["A2_shrunk_minus_population_mean"]["diff"],
                                 "coverage_A1": c["coverage_90_A1_additive"]["mean"],
                                 "coverage_A2": c["coverage_90_A2_shrunk"]["mean"]})
    pd.DataFrame(sub_rows).to_csv(DOC / "CF3_SUBGROUPS.csv", index=False)

    # consistency with the Phase 5 ATT: average tau over the primary MI episodes vs the ATT (SD units)
    tau = model.m1(prim) - model.m0(prim)
    tau_boot = np.stack([b.m1(prim) - b.m0(prim) for b in boots])     # (200, n, 16)
    p5 = pd.read_csv(PHASE_DIRS[5] / "LAYER2_ESTIMATES.csv").set_index("outcome")
    att = {m: {"track_a_mean_tau_sd": round(float(tau[:, j].mean() / sd[j]), 3),
               "phase5_att_sd": round(float(p5.loc[f"w_{m}", "att_sd"]), 3),
               "tau_model_se_sd_median": round(float(np.median(tau_boot[:, :, j].std(0)) / sd[j]), 3)}
           for j, m in enumerate(ROBUST)}

    # gate CF-G1: on the ST/T measures (composite), each Track A variant vs both baselines, + coverage 85-95%
    gate = {}
    for a in ("A1_additive", "A2_shrunk"):
        g = {}
        for direction in ("add", "remove"):
            c = comp["primary"][direction]
            beats = all(c[f"{a}_minus_{ref}"]["ci"][1] < 0 for ref in ("no_change", "population_mean"))
            cov = c[f"coverage_90_{a}"]["mean"]
            g[direction] = {"beats_both_baselines": beats, "coverage_90": cov, "coverage_ok": 0.85 <= cov <= 0.95,
                            "pass": beats and 0.85 <= cov <= 0.95}
        g["pass"] = all(g[d]["pass"] for d in ("add", "remove"))
        gate[a] = g

    summary = {
        "training": {"rows": len(tr), "mi_episodes": int(tr.mi.sum()), "controls": int((tr.mi == 0).sum()),
                     "patients": int(tr.subject_id.nunique()),
                     "mi_by_type": tr.loc[tr.mi == 1, "cf_type"].value_counts().to_dict(),
                     "design": spec("layer2").name, "covariates": L2_COVS,
                     "st_artefact_rule": f"|ST_Elev| > {ST_MAX_ABS_MV} mV set to missing before composites"},
        "pairs": {"val": val.groupby("side").size().to_dict(), "test": test.groupby("side").size().to_dict(),
                  "primary_test_pairs": int(test.primary.sum()),
                  "primary_test_patients": int(test.loc[test.primary, "subject_id"].nunique())},
        "rho_pair_val": {side: {dr: dict(zip(ROBUST, np.round(t["rho"], 3).tolist())) for dr, t in v.items()}
                         for side, v in tuned.items()},
        "composite_st_t": comp,
        "consistency_with_phase5_att": att,
        "gate_cf_g1": gate,
    }
    (DOC / "cf3_01_track_a.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.concat(preds_out, ignore_index=True).to_parquet(OUT / "cf3_predictions.parquet", index=False)
    with open(OUT / "cf3_track_a_model.pkl", "wb") as fh:
        pickle.dump({"model": model, "sd": sd, "measures": ROBUST, "types": TYPES,
                     "tuned": tuned, "st_max_abs_mv": ST_MAX_ABS_MV}, fh)
    return summary


if __name__ == "__main__":
    # run through the importable module so the pickled TrackA class resolves outside this script
    from roadmap.phase10_counterfactual_ecg.cf3_01_track_a import main as _main
    s = _main()
    print(json.dumps({k: s[k] for k in ("training", "pairs", "composite_st_t", "gate_cf_g1")}, indent=2, default=str))
