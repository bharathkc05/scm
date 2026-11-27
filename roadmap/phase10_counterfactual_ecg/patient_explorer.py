"""Phase 10 — local patient explorer: individual counterfactual and interventional reasoning for one real patient.

LOCAL ONLY. The report contains one patient's MIMIC data and counterfactuals (tier T1). It is written to
data/roadmap/phase10/explorer/ (git-ignored) and must never be sent through the MCP server or any cloud service.
The command prints only the report path, never patient values.

    python -X utf8 -m roadmap.phase10_counterfactual_ecg.patient_explorer --random --kind pair
    python -X utf8 -m roadmap.phase10_counterfactual_ecg.patient_explorer --random --kind mi --seed 7
    python -X utf8 -m roadmap.phase10_counterfactual_ecg.patient_explorer --random --kind control
    python -X utf8 -m roadmap.phase10_counterfactual_ecg.patient_explorer --subject <subject_id>

Kinds (random selection draws from held-out patients unless --allow-training):
    pair     MI patient with a during-MI and a baseline ECG (pool: pair_test). Counterfactuals are checked against
             the patient's real other ECG.
    mi       other MI patient with a during-MI ECG (pool: test split). "Without MI" counterfactual.
    control  never-MI patient (pool: test split). "With MI" counterfactuals: NSTEMI, anterior and inferior STEMI.

What the report contains:
    Track A (CF3, model A2): the 16 robust ECG measures, counterfactual value with its 90% split-conformal interval.
    Track B (CF4): 12-lead median beat; counterfactual beat = observed + (decoder(z, c_cf) - decoder(z, c_obs)).
    Clinical do(): probability that an admission is an acute-MI admission under do(condition = no / yes) for this
    patient's covariate profile (Phase 5 Layer 1 design, refitted on the Phase 3 training split), and for MI patients
    with the condition, the probability of necessity PN = 1 - r0/r1 (monotonicity and no unmeasured confounding).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import pickle

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, REPO
from roadmap.estimators import Design, logistic
from roadmap.phase5_estimation.queries import L1_QUERIES
from roadmap.phase7_stress_test.common import spec
from roadmap.phase10_counterfactual_ecg import cf3_01_track_a as A
from roadmap.phase10_counterfactual_ecg import cf4_model as B

OUT = DATA_OUT / "phase10"
REPORTS = OUT / "explorer"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ADD_STATES = ["nstemi", "stemi_anterior", "stemi_inferior"]
STATE_LABEL = {"none": "no MI", "nstemi": "NSTEMI", "stemi_anterior": "anterior STEMI",
               "stemi_inferior": "inferior STEMI", "stemi_other": "other STEMI", "type2": "type 2 MI",
               "other": "other / unspecified MI"}
L1_PRIMARY = [q for q in L1_QUERIES if q["role"] == "primary" and q["exposure"] != "prior_mi_primary"]
EXPOSURE_LABEL = {"diabetes_primary": "diabetes", "hypertension_primary": "hypertension", "ckd_primary": "CKD"}


# ----------------------------------------------------------------------------- models and data
class Context:
    def __init__(self):
        import torch
        self.torch = torch
        self.meas = A.measures()
        self.ep = pd.read_parquet(OUT / "episodes.parquet")
        self.pairs = pd.read_parquet(OUT / "pairs.parquet")
        sp = pd.read_parquet(OUT / "cf_splits.parquet")
        self.split = sp.set_index("subject_id").cf_split.astype(object)
        self.phase3_split = sp.set_index("subject_id").phase3_split.astype(object)
        self.rows = B.build_rows(splits=None)
        with open(OUT / "cf3_track_a_model.pkl", "rb") as fh:
            self.ta = pickle.load(fh)
        ck = torch.load(REPO / "models/cf_vae/cvae_selected.pt", map_location="cpu", weights_only=False)
        self.vae = B.CondVAE(ck["z_dim"])
        self.vae.load_state_dict(ck["state_dict"])
        self.vae.eval()
        self.scale = ck["scale"]
        self.l1 = layer1_models()

    def beats(self, rows) -> np.ndarray:
        import h5py
        with h5py.File(OUT / "beats.h5", "r") as h:
            return np.stack([h["beats"][int(r)].astype(np.float32) for r in rows])


def layer1_models() -> dict:
    """Arm-specific logistic outcome models per primary Layer 1 query, Phase 3 training split, cached."""
    path = OUT / "explorer_layer1_models.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    l1 = pd.read_parquet(DATA_OUT / "phase3/layer1.parquet")
    tr = l1[l1.split == "train"].reset_index(drop=True)
    models = {}
    for q in L1_PRIMARY:
        des = Design(spec("layer1"), q["covs"]).fit(tr)
        x, y = tr[q["exposure"]].to_numpy(), tr["acute_mi"].to_numpy()
        xt = des.transform(tr)
        models[q["exposure"]] = {"design": des, "covs": q["covs"], "label": q["label"],
                                 "arm": {a: logistic(xt[x == a], y[x == a]) for a in (0, 1)}}
    with open(path, "wb") as fh:
        pickle.dump(models, fh)
    return models


# ----------------------------------------------------------------------------- selection
def pools(ctx: Context, allow_training: bool) -> dict:
    r = ctx.rows
    pair_ok = A.pair_table(ctx.meas, ctx.ep, ctx.pairs, ctx.split)
    pair_ok = pair_ok[pair_ok.during_study_id.isin(set(r.record_id)) & pair_ok.base_study_id.isin(set(r.record_id))]
    mi = r[r.mi_state != "none"]
    ctrl = r[r.role == "control"]
    held = {"pair": ["pair_test"], "mi": ["test"], "control": ["test"]}
    if allow_training:
        held = {"pair": ["pair_test", "pair_val"], "mi": ["test", "validation", "train"],
                "control": ["test", "validation", "train"]}
    return {"pair": np.sort(pair_ok[pair_ok.cf_split.isin(held["pair"])].subject_id.unique()),
            "mi": np.sort(mi[mi.split.isin(held["mi"])].subject_id.unique()),
            "control": np.sort(ctrl[ctrl.split.isin(held["control"])].subject_id.unique())}


def choose(ctx: Context, kind: str, seed: int, allow_training: bool) -> int:
    pool = pools(ctx, allow_training)[kind]
    return int(np.random.default_rng(seed).choice(pool))


def kind_of(ctx: Context, subject_id: int) -> str:
    if subject_id in set(ctx.pairs.subject_id):
        p = A.pair_table(ctx.meas, ctx.ep, ctx.pairs[ctx.pairs.subject_id == subject_id], ctx.split)
        if len(p):
            return "pair"
    r = ctx.rows[ctx.rows.subject_id == subject_id]
    if (r.mi_state != "none").any():
        return "mi"
    if (r.role == "control").any():
        return "control"
    return "other"


def training_use(ctx: Context, subject_id: int) -> dict:
    cf = ctx.split.get(subject_id, "none")
    p3 = ctx.phase3_split.get(subject_id, "none")
    return {"phase10_split": cf, "phase3_split": p3,
            "track_a_trained_on_patient": cf == "train", "track_b_trained_on_patient": cf == "train",
            "track_a_rho_and_intervals_tuned_on_patient": cf == "pair_val",
            "layer1_trained_on_patient": p3 == "train"}


# ----------------------------------------------------------------------------- Track A
def track_a_single(ctx: Context, row: pd.Series, action: str, state: str) -> pd.DataFrame:
    """Counterfactual for one ECG. action 'remove' (during-MI ECG -> no MI) or 'add' (ECG -> MI of type `state`).
    Tuning (rho, interval) from the primary before-baseline pairs."""
    m = ctx.ta["model"]
    t = ctx.ta["tuned"]["before"][action]
    cov = pd.DataFrame([{"age": row.age, "sex_male": row.sex_male, "diabetes_primary": row.diabetes,
                         "hypertension_primary": row.hypertension, "ckd_primary": row.ckd,
                         "prior_mi_primary": row.prior_mi,
                         "cf_type": row.mi_state if action == "remove" else state}])
    y = ctx.meas.loc[[row.record_id]].to_numpy(float)
    m0, m1 = m.m0(cov), m.m1(cov)
    src_m, tgt_m = (m1, m0) if action == "remove" else (m0, m1)
    cf = tgt_m + t["rho"] * (y - src_m)
    q = t["q"]["A2_shrunk"]
    return pd.DataFrame({"measure": A.ROBUST, "observed": y[0], "counterfactual": cf[0], "low": cf[0] - q,
                         "high": cf[0] + q, "population_mean": tgt_m[0], "change": (cf - y)[0],
                         "change_sd": ((cf - y)[0] / ctx.ta["sd"])})


def track_a_pair(ctx: Context, prow: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Exactly the CF3 validation computation for one pair row."""
    side = prow.side.iat[0]
    t = ctx.ta["tuned"][side][direction]
    d = A.predict(ctx.ta["model"], prow.reset_index(drop=True), ctx.meas)[direction]
    pr = A.methods(d, t["rho"])
    q = t["q"]["A2_shrunk"]
    sd = ctx.ta["sd"]
    a2 = pr["A2_shrunk"][0]
    return pd.DataFrame({"measure": A.ROBUST, "source_observed": d["src"][0], "counterfactual": a2,
                         "low": a2 - q, "high": a2 + q, "real_other_ecg": d["tgt"][0],
                         "population_mean": pr["population_mean"][0],
                         "error_sd": np.abs(a2 - d["tgt"][0]) / sd,
                         "error_no_change_sd": np.abs(d["src"][0] - d["tgt"][0]) / sd,
                         "error_population_mean_sd": np.abs(pr["population_mean"][0] - d["tgt"][0]) / sd,
                         "inside_interval": (d["tgt"][0] >= a2 - q) & (d["tgt"][0] <= a2 + q)})


# ----------------------------------------------------------------------------- Track B
def track_b(ctx: Context, row: pd.Series, state: str) -> dict:
    torch = ctx.torch
    x = ctx.beats([row.row])                                              # mV
    df = row.to_frame().T
    for c in B.COVS:
        df[c] = df[c].astype(float)
    c_obs = torch.from_numpy(B.condition(df))
    c_cf = torch.from_numpy(B.condition(df, state))
    xt = torch.from_numpy(x / ctx.scale)
    with torch.no_grad():
        rec = ctx.vae.counterfactual(xt, c_obs, c_obs).numpy() * ctx.scale
        cf = ctx.vae.counterfactual(xt, c_obs, c_cf).numpy() * ctx.scale
    out = x + (cf - rec)
    j = row.j_offset_ms if np.isfinite(row.j_offset_ms) else 100.0
    k = int(np.clip(B.QRS_ON + round((j + 60) / 2), 0, B.N_SAMPLES - 1))
    return {"observed": x[0], "counterfactual": out[0], "reconstruction_fvu": B.fvu_per_lead(x, rec),
            "st_j60_observed": x[0, :, k], "st_j60_counterfactual": out[0, :, k], "k": k}


# ----------------------------------------------------------------------------- clinical
def clinical(ctx: Context, profile: dict, had_mi: bool) -> pd.DataFrame:
    rows = []
    for exp, m in ctx.l1.items():
        df = pd.DataFrame([profile])
        xt = m["design"].transform(df)
        r0, r1 = (float(m["arm"][a].predict_proba(xt)[0, 1]) for a in (0, 1))
        has = bool(profile[exp])
        pn = max(0.0, 1 - r0 / r1) if (has and had_mi and r1 > 0) else np.nan
        rows.append({"condition": EXPOSURE_LABEL[exp], "patient_has_it": has, "adjusted_for": m["label"],
                     "risk_do_no": r0, "risk_do_yes": r1, "risk_difference_pp": 100 * (r1 - r0),
                     "risk_ratio": r1 / r0 if r0 > 0 else np.nan, "pn": pn})
    return pd.DataFrame(rows)


def profile_for(ctx: Context, subject_id: int, kind: str, hadm_id=None, ecg_row: pd.Series | None = None) -> dict:
    if kind in ("pair", "mi") and hadm_id is not None:
        e = ctx.ep.set_index("hadm_id").loc[hadm_id]
        p = {"age": float(e.age), "sex_male": float(e.sex_male)}
        for c in ("diabetes_primary", "hypertension_primary", "ckd_primary"):
            p[c] = float(e[c]) if pd.notna(e[c]) else 0.0
        return p
    r = ecg_row
    return {"age": float(r.age), "sex_male": float(r.sex_male), "diabetes_primary": float(r.diabetes),
            "hypertension_primary": float(r.hypertension), "ckd_primary": float(r.ckd)}


# ----------------------------------------------------------------------------- figures and HTML
def fig_beats(lines: list[tuple[str, np.ndarray, str, str]], k: int | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = (np.arange(B.N_SAMPLES) - B.QRS_ON) * 2.0                         # ms from QRS onset
    fig, axes = plt.subplots(3, 4, figsize=(13, 7), sharex=True)
    for i, ax in enumerate(axes.T.ravel()):                               # columns: I-III, aVR-aVF, V1-V3, V4-V6
        for label, x, color, style in lines:
            ax.plot(t, x[i], color=color, ls=style, lw=1.3, label=label)
        if k is not None:
            ax.axvline((k - B.QRS_ON) * 2.0, color="#999", lw=0.6, ls=":")
        ax.axhline(0, color="#ccc", lw=0.6)
        ax.set_title(LEADS[i], fontsize=10, loc="left")
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.supxlabel("ms from QRS onset (dotted line: J + 60 ms)", fontsize=9)
    fig.supylabel("mV (baseline at QRS onset)", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return f'<img alt="12-lead median beats" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                s = "yes" if v else "no"
            elif isinstance(v, (float, np.floating)):
                s = "–" if not np.isfinite(v) else fmt.get(c, "{:.3f}").format(v)
            else:
                s = html.escape(str(v))
            cells.append(f"<td>{s}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


CSS = """
body{font-family:system-ui,Segoe UI,sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#1d2433;
background:#fff;line-height:1.45}
h1{font-size:1.5rem;margin-bottom:4px} h2{font-size:1.15rem;margin-top:32px;border-bottom:1px solid #dde;padding-bottom:4px}
h3{font-size:1rem;margin-top:20px}
.warn{background:#fff6e5;border-left:4px solid #e0a000;padding:10px 14px;margin:12px 0}
.note{background:#eef4ff;border-left:4px solid #3b6fd8;padding:10px 14px;margin:12px 0}
table{border-collapse:collapse;font-size:.85rem;margin:8px 0;display:block;overflow-x:auto}
th,td{border:1px solid #dde;padding:3px 7px;text-align:right;white-space:nowrap} th{background:#f3f5f9}
td:first-child,th:first-child{text-align:left} img{max-width:100%} .muted{color:#667}
"""


def page(title: str, sections: list[str]) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style>"
            f"</head><body>{''.join(sections)}</body></html>")


A_FMT = {c: "{:.3f}" for c in ("observed", "counterfactual", "low", "high", "population_mean", "change",
                               "source_observed", "real_other_ecg")}
A_FMT.update({c: "{:.2f}" for c in ("change_sd", "error_sd", "error_no_change_sd", "error_population_mean_sd")})


# ----------------------------------------------------------------------------- report
def report(ctx: Context, subject_id: int) -> tuple[str, str]:
    kind = kind_of(ctx, subject_id)
    use = training_use(ctx, subject_id)
    rows = ctx.rows[ctx.rows.subject_id == subject_id].sort_values("ecg_time")
    held_out = not (use["track_a_trained_on_patient"] or use["track_b_trained_on_patient"])
    S = [f"<h1>Patient explorer: individual counterfactuals</h1>",
         f"<p class='muted'>MIMIC-IV subject {subject_id} · kind: <b>{kind}</b> · Phase 10 split: "
         f"<b>{use['phase10_split']}</b> · local report, do not share</p>",
         "<div class='warn'><b>Research prototype, not a diagnosis.</b> Counterfactuals are model-based ranges, not "
         "this patient's true alternative ECG. On held-out pairs the best model (A2) still has an average error of "
         "about 0.9 SD per ST/T measure when adding MI and 0.7 SD when removing it; 90% intervals covered 89–92%. "
         "Intervals are <b>too narrow for STEMI, especially anterior STEMI</b> (77% coverage).</div>"]
    if not held_out:
        S.append("<div class='warn'><b>In-sample patient.</b> The Track A and Track B models were trained on this "
                 "patient's ECGs, so the results below are optimistic.</div>")
    if use["track_a_rho_and_intervals_tuned_on_patient"]:
        S.append("<div class='note'>This patient is in <code>pair_val</code>, which tuned ρ and the intervals.</div>")

    # timeline
    tl = rows[["ecg_time", "role", "mi_state", "hr"]].copy()
    tl["ecg_time"] = tl.ecg_time.astype(str)
    tl["mi_state"] = tl.mi_state.map(STATE_LABEL)
    eps = ctx.ep[ctx.ep.subject_id == subject_id][["admittime", "dischtime", "mi_kind", "territory",
                                                    "revascularised"]].astype(str)
    S += ["<h2>Timeline</h2>", "<h3>QC-passing ECGs (median beats available)</h3>",
          table(tl, {"hr": "{:.0f}"})]
    if len(eps):
        S += ["<h3>Acute-MI admissions</h3>", table(eps)]

    hadm = None
    if kind == "pair":
        p = A.pair_table(ctx.meas, ctx.ep, ctx.pairs[ctx.pairs.subject_id == subject_id], ctx.split)
        p = p[p.during_study_id.isin(set(rows.record_id)) & p.base_study_id.isin(set(rows.record_id))]
        p = p.sort_values(["primary", "side"], ascending=[False, True])
        hadm = p.hadm_id.iat[0]
        S.append("<h2>Counterfactuals checked against the patient's real ECGs</h2>"
                 "<p>Each pair is one MI episode with a baseline ECG (before or after). <b>Remove MI</b> starts from the "
                 "during-MI ECG and predicts the baseline; <b>add MI</b> starts from the baseline and predicts the "
                 "during-MI ECG. Errors are in control-SD units; the other columns show how two baselines would "
                 "have done: copying the source ECG (no change) and the expected value for similar patients.</p>")
        for _, pr in p.iterrows():
            prow = p[(p.hadm_id == pr.hadm_id) & (p.side == pr.side)]
            dur = rows[rows.record_id == pr.during_study_id].iloc[0]
            base = rows[rows.record_id == pr.base_study_id].iloc[0]
            S.append(f"<h3>Episode {STATE_LABEL.get(pr.cf_type, pr.cf_type)} · {pr.side} baseline "
                     f"({pr.gap_days:.0f} days apart){' · primary test pair' if pr.primary else ''}</h3>")
            for direction, src, tgt, state in (("remove", dur, base, "none"), ("add", base, dur, dur.mi_state)):
                ta = track_a_pair(ctx, prow, direction)
                st = ta[ta.measure.isin(A.ST_T)]
                S.append(f"<p><b>{'Remove MI' if direction == 'remove' else 'Add MI'}</b> — Track A, 14 ST/T measures: "
                         f"mean error {st.error_sd.mean():.2f} SD (no change {st.error_no_change_sd.mean():.2f}, "
                         f"population mean {st.error_population_mean_sd.mean():.2f}); real value inside the 90% "
                         f"interval for {int(st.inside_interval.sum())} of 14.</p>")
                S.append(table(ta, A_FMT))
                tb = track_b(ctx, src, state)
                real = ctx.beats([tgt.row])[0]
                S.append(fig_beats([("observed (source)", tb["observed"], "#1d2433", "-"),
                                    ("counterfactual", tb["counterfactual"], "#d0342c", "-"),
                                    ("real other ECG", real, "#3b6fd8", "--")], tb["k"]))
                kk = tb["k"]
                stt = pd.DataFrame({"lead": LEADS, "source": tb["st_j60_observed"],
                                    "counterfactual": tb["st_j60_counterfactual"], "real_other_ecg": real[:, kk]})
                S.append("<p class='muted'>Track B, ST at J + 60 ms of the median beat (mV):</p>" + table(stt))
    elif kind == "mi":
        dur = rows[rows.mi_state != "none"].iloc[0]
        hadm = dur.during_hadm_id
        S.append(f"<h2>Without MI: counterfactual of the first during-MI ECG ({STATE_LABEL[dur.mi_state]})</h2>"
                 "<p>No real no-MI ECG of this episode is available to check against.</p>")
        S.append(table(track_a_single(ctx, dur, "remove", "none"), A_FMT))
        tb = track_b(ctx, dur, "none")
        S.append(fig_beats([("observed (during MI)", tb["observed"], "#1d2433", "-"),
                            ("counterfactual: no MI", tb["counterfactual"], "#d0342c", "-")], tb["k"]))
    elif kind == "control":
        base = rows[rows.role == "control"].iloc[0]
        S.append("<h2>With MI: counterfactuals of this patient's ECG</h2>"
                 "<p>The patient never had an acute-MI admission. Each MI type is applied in turn.</p>")
        for state in ADD_STATES:
            S.append(f"<h3>{STATE_LABEL[state]}</h3>" + table(track_a_single(ctx, base, "add", state), A_FMT))
            tb = track_b(ctx, base, state)
            S.append(fig_beats([("observed (no MI)", tb["observed"], "#1d2433", "-"),
                                (f"counterfactual: {STATE_LABEL[state]}", tb["counterfactual"], "#d0342c", "-")],
                               tb["k"]))
    else:
        S.append("<p>No QC-passing median beat for this patient, so no ECG counterfactual is possible.</p>")

    # clinical do()
    if kind != "other":
        prof = profile_for(ctx, subject_id, kind, hadm, rows.iloc[0])
        cl = clinical(ctx, prof, kind in ("pair", "mi"))
        S += ["<h2>Clinical interventions: do(condition)</h2>",
              f"<p>Profile used: age {prof['age']:.0f}, {'male' if prof['sex_male'] else 'female'}, diabetes "
              f"{'yes' if prof['diabetes_primary'] else 'no'}, hypertension "
              f"{'yes' if prof['hypertension_primary'] else 'no'}, CKD {'yes' if prof['ckd_primary'] else 'no'} "
              f"({'at the MI admission' if hadm is not None else 'as of the ECG'}). The risk is the probability that "
              "a hospital admission of a patient with this profile is an acute-MI admission, with the condition set "
              "to no or yes and the other covariates of each Phase 5 query held at the patient's values.</p>",
              table(cl, {"risk_do_no": "{:.4f}", "risk_do_yes": "{:.4f}", "risk_difference_pp": "{:.2f}",
                         "risk_ratio": "{:.2f}", "pn": "{:.2f}"}),
              "<div class='note'><b>PN</b> (only for an MI patient who has the condition): the probability that the "
              "condition was necessary for this MI, = 1 − r0/r1. It assumes the condition never prevents MI "
              "(monotonicity) and no unmeasured confounding given the query's covariates; otherwise it is a lower "
              "bound. Phase 7 bounded unmeasured confounding at about 38% of effect size. Point estimates only; "
              "population CIs are in Phase 5. Not validated per patient: nobody's diabetes switches off.</div>"]
    S.append("<p class='muted'>Generated by roadmap/phase10_counterfactual_ecg/patient_explorer.py · Track A: CF3 "
             "model A2 · Track B: CF4 cvae_b1.0_l10, counterfactual beat = observed + decoder difference · Clinical: "
             "Phase 5 Layer 1 design refitted on the Phase 3 training split.</p>")
    name = "report_" + hashlib.sha1(str(subject_id).encode()).hexdigest()[:10] + ".html"
    return name, page("Patient explorer", S)


def main(argv=None) -> str:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subject", type=int)
    g.add_argument("--random", action="store_true")
    ap.add_argument("--kind", choices=["pair", "mi", "control"], default="pair")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: a new random patient each run)")
    ap.add_argument("--allow-training", action="store_true", help="random selection may pick in-sample patients")
    a = ap.parse_args(argv)
    ctx = Context()
    sid = a.subject if a.subject is not None else choose(
        ctx, a.kind, a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 2**32), a.allow_training)
    name, doc = report(ctx, sid)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(doc, encoding="utf-8")
    return str(REPORTS / name)


if __name__ == "__main__":
    print("report written (local only, contains patient data):", main())
