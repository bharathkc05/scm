"""Phase 10, CF4 — train the Track B conditional beta-VAE (cf4_model.py) and check its acceptance criteria.

Data: QC-passing MIMIC median beats of the Phase 10 train split (validation split for early stopping and selection;
the test split is untouched until CF5). Pair patients are never seen. Sampling is reweighted by 1/sqrt(count) of the
MI state so that rare STEMI territories are seen more often.

Grid (pre-specified before the full runs; lambda 200 added after a 3-epoch smoke test showed a weak adversary):
beta in {0.5, 1, 2} x adversary weight lambda in {10, 50, 200}; z_dim 32; seed fixed.
Selection rule (pre-specified): among runs whose MI probe on z has AUROC <= 0.55 on the validation split (both a
logistic and a gradient-boosting probe), the lowest validation reconstruction error; otherwise the lowest AUROC,
and the gate fails.

Acceptance (ROADMAP CF4):
    reconstruction error <= the current VAE's (per-lead FVU 0.278, validation, from its training log)
    adversary on z: AUROC <= 0.55 (fresh probes trained on train-split z, scored on validation z)
    reproducible from a fixed seed (two short runs with the same seed give identical weights)

    python -X utf8 -m roadmap.phase10_counterfactual_ecg.cf4_01_train            # grid + selection + checks
Outputs: CF4_SWEEP.csv, cf4_01_train.json (committed); models/cf_vae/*.pt, data/roadmap/phase10/cf4_rows.parquet
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from roadmap.config import PHASE_DIRS, REPO, SEED
from roadmap.phase10_counterfactual_ecg.cf4_model import (MI_STATES, OUT, QRS_ON, CondVAE, build_rows, condition,
                                                          fvu_per_lead, load_beats, n_params)

DOC = PHASE_DIRS[10]
MODELS = REPO / "models" / "cf_vae"
GRID = [(b, l) for b in (0.5, 1.0, 2.0) for l in (10.0, 50.0, 200.0)]
Z_DIM, BATCH, LR, MAX_EPOCHS, PATIENCE, LAM_RAMP = 32, 256, 1e-3, 150, 15, 10
OLD_VAE_FVU = 16690.6014 / 60000      # current VAE: best-run validation recon loss / (12 x 5000), per-lead z-scored
AUROC_MAX = 0.55
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
LATERAL = [LEADS.index(l) for l in ("I", "aVL", "V5", "V6")]


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Data:
    def __init__(self, dev):
        rows = build_rows()
        rows.to_parquet(OUT / "cf4_rows.parquet", index=False)
        self.rows = rows
        x = load_beats(rows.row.to_numpy())
        tr = (rows.split == "train").to_numpy()
        self.scale = float(x[tr].std())
        self.dev = dev
        self.t = {}
        for s in ("train", "validation"):
            m = (rows.split == s).to_numpy()
            self.t[s] = {"x": torch.from_numpy(x[m] / self.scale).to(dev),
                         "c": torch.from_numpy(condition(rows[m])).to(dev),
                         "y": torch.from_numpy(rows[m].mi_state.map(MI_STATES.index).to_numpy().copy()).to(dev),
                         "df": rows[m].reset_index(drop=True)}
        y = self.t["train"]["y"].cpu().numpy()
        cnt = np.bincount(y, minlength=len(MI_STATES)).astype(float)
        self.weights = torch.tensor(1 / np.sqrt(cnt[y]), dtype=torch.float64)


def run(data: Data, beta: float, lam: float, seed: int, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
        verbose: bool = True) -> tuple[CondVAE, dict]:
    seed_all(seed)
    dev = data.dev
    model = CondVAE(Z_DIM).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    gen = torch.Generator().manual_seed(seed)
    tr, va = data.t["train"], data.t["validation"]
    n = len(tr["x"])
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(max_epochs):
        model.train()
        t0 = time.time()
        lam_t = lam * min(1.0, (ep + 1) / LAM_RAMP)
        order = torch.multinomial(data.weights, n, replacement=True, generator=gen).to(dev)
        agg = np.zeros(4)
        for s in range(0, n, BATCH):
            b = order[s:s + BATCH]
            x, c, y = tr["x"][b], tr["c"][b], tr["y"][b]
            xh, mu, lv, logits = model(x, c, lam_t)
            rec = ((xh - x) ** 2).sum((1, 2)).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum(1).mean()
            adv = F.cross_entropy(logits, y)
            loss = rec + beta * kl + adv          # gradient reversal scales the adversary's push on z by lam_t
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            agg += np.array([rec.item(), kl.item(), adv.item(), 1])
        v = evaluate(model, va, beta)
        sched.step(v["loss"])
        hist.append({"epoch": ep, "rec": agg[0] / agg[3], "kl": agg[1] / agg[3], "adv_ce": agg[2] / agg[3], **v,
                     "sec": round(time.time() - t0, 1)})
        if verbose and (ep % 10 == 0 or ep == max_epochs - 1):
            print(f"  beta={beta} lam={lam} ep{ep:3d} rec={agg[0]/agg[3]:.1f} kl={agg[1]/agg[3]:.1f} "
                  f"adv={agg[2]/agg[3]:.3f} | val loss={v['loss']:.1f} fvu={v['fvu']:.4f} ({time.time()-t0:.1f}s)",
                  flush=True)
        if v["loss"] < best - 1e-3:
            best, wait = v["loss"], 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            best_ep = ep
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep, "epochs_run": ep + 1, "history": hist}


@torch.no_grad()
def evaluate(model: CondVAE, d: dict, beta: float) -> dict:
    model.eval()
    rec, kl, xs, xhs = 0.0, 0.0, [], []
    for s in range(0, len(d["x"]), 1024):
        x, c = d["x"][s:s + 1024], d["c"][s:s + 1024]
        mu, lv = model.encode(x, c)
        xh = model.decode(mu, c)
        rec += ((xh - x) ** 2).sum().item()
        kl += (-0.5 * (1 + lv - mu ** 2 - lv.exp())).sum().item()
        xs.append(x.cpu().numpy())
        xhs.append(xh.cpu().numpy())
    n = len(d["x"])
    x, xh = np.concatenate(xs), np.concatenate(xhs)
    return {"loss": (rec + beta * kl) / n, "val_rec": rec / n, "val_kl": kl / n, "fvu": fvu_per_lead(x, xh),
            "fvu_global": float(((x - xh) ** 2).sum() / ((x - x.mean()) ** 2).sum())}


@torch.no_grad()
def latents(model: CondVAE, d: dict) -> np.ndarray:
    model.eval()
    return np.concatenate([model.encode(d["x"][s:s + 2048], d["c"][s:s + 2048])[0].cpu().numpy()
                           for s in range(0, len(d["x"]), 2048)])


def probe(model: CondVAE, data: Data) -> dict:
    """Fresh classifiers predict acute MI (any MI state) from z: fit on train z, AUROC on validation z."""
    ztr, zva = latents(model, data.t["train"]), latents(model, data.t["validation"])
    ytr = (data.t["train"]["y"].cpu().numpy() > 0).astype(int)
    yva = (data.t["validation"]["y"].cpu().numpy() > 0).astype(int)
    sc = StandardScaler().fit(ztr)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(ztr), ytr)
    hgb = HistGradientBoostingClassifier(max_iter=300, random_state=SEED, class_weight="balanced").fit(ztr, ytr)
    out = {"auroc_logistic": roc_auc_score(yva, lr.predict_proba(sc.transform(zva))[:, 1]),
           "auroc_boosting": roc_auc_score(yva, hgb.predict_proba(zva)[:, 1])}
    out["auroc_max"] = max(out.values())
    # STEMI vs no MI (the morphology most at risk of leaking into z)
    m = np.isin(data.t["validation"]["y"].cpu().numpy(), [0, 2, 3, 4])
    out["auroc_boosting_stemi_vs_none"] = roc_auc_score(yva[m], hgb.predict_proba(zva[m])[:, 1])
    return {k: round(float(v), 4) for k, v in out.items()}


@torch.no_grad()
def st_shift_check(model: CondVAE, data: Data) -> dict:
    """Sanity check for CF5: decode validation during-MI beats with their own state and with 'none'; ST at J+60 ms
    (median-beat J offset) in mV, lateral leads. Removing MI should move lateral ST up (Phase 5 ATT is negative)."""
    d = data.t["validation"]
    df = d["df"]
    m = (df.mi_state != "none").to_numpy()
    x, c_obs = d["x"][m], d["c"][m]
    c_cf = torch.from_numpy(condition(df[m], "none")).to(data.dev)
    xh_obs = model.counterfactual(x, c_obs, c_obs).cpu().numpy() * data.scale
    xh_cf = model.counterfactual(x, c_obs, c_cf).cpu().numpy() * data.scale
    j = df[m].j_offset_ms.to_numpy()
    k = np.clip(QRS_ON + np.round((np.nan_to_num(j, nan=100) + 60) / 2).astype(int), 0, 399)
    st = lambda a: a[np.arange(len(a)), :, k]          # noqa: E731  (n, 12)
    diff = st(xh_cf) - st(xh_obs)
    out = {"n": int(m.sum()), "lateral_st_shift_mv_remove_mi": round(float(diff[:, LATERAL].mean()), 4)}
    for state in ("nstemi", "stemi_anterior", "stemi_inferior"):
        s = (df[m].mi_state == state).to_numpy()
        out[f"st_shift_mv_by_lead_{state}"] = dict(zip(LEADS, np.round(diff[s].mean(0), 4).tolist()))
    return out


def reproducibility(data: Data, beta: float, lam: float) -> dict:
    a, _ = run(data, beta, lam, SEED, max_epochs=3, patience=99, verbose=False)
    b, _ = run(data, beta, lam, SEED, max_epochs=3, patience=99, verbose=False)
    diff = max(float((p - q).abs().max()) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
    return {"epochs": 3, "max_abs_weight_diff": diff, "identical": diff == 0.0}


def main():
    dev = torch.device("cuda")
    MODELS.mkdir(parents=True, exist_ok=True)
    data = Data(dev)
    counts = {s: data.t[s]["df"].mi_state.value_counts().to_dict() for s in ("train", "validation")}
    print("rows:", {s: len(data.t[s]["x"]) for s in data.t}, "scale (mV):", round(data.scale, 4), flush=True)
    results = []
    for beta, lam in GRID:
        t0 = time.time()
        model, info = run(data, beta, lam, SEED)
        v = evaluate(model, data.t["validation"], beta)
        pr = probe(model, data)
        name = f"cvae_b{beta}_l{lam:g}"
        torch.save({"state_dict": model.state_dict(), "z_dim": Z_DIM, "scale": data.scale, "beta": beta, "lam": lam,
                    "seed": SEED, "mi_states": MI_STATES}, MODELS / f"{name}.pt")
        pd.DataFrame(info["history"]).to_csv(OUT / f"cf4_history_{name}.csv", index=False)
        r = {"run": name, "beta": beta, "lambda": lam, "best_epoch": info["best_epoch"],
             "epochs_run": info["epochs_run"], "val_fvu_per_lead": round(v["fvu"], 4),
             "val_fvu_global": round(v["fvu_global"], 4), "val_kl": round(v["val_kl"], 2), **pr,
             "minutes": round((time.time() - t0) / 60, 1)}
        print(r, flush=True)
        results.append(r)
        pd.DataFrame(results).to_csv(DOC / "CF4_SWEEP.csv", index=False)

    sweep = pd.DataFrame(results)
    ok = sweep[sweep.auroc_max <= AUROC_MAX]
    chosen = (ok.sort_values("val_fvu_per_lead") if len(ok) else sweep.sort_values("auroc_max")).iloc[0]
    ck = torch.load(MODELS / f"{chosen.run}.pt", map_location=dev)
    model = CondVAE(Z_DIM).to(dev)
    model.load_state_dict(ck["state_dict"])
    torch.save(ck, MODELS / "cvae_selected.pt")
    shift = st_shift_check(model, data)
    repro = reproducibility(data, float(chosen.beta), float(chosen["lambda"]))
    gate = {"reconstruction": {"val_fvu_per_lead": float(chosen.val_fvu_per_lead), "current_vae": round(OLD_VAE_FVU, 4),
                               "pass": bool(chosen.val_fvu_per_lead <= OLD_VAE_FVU)},
            "adversary": {"auroc_max": float(chosen.auroc_max), "threshold": AUROC_MAX,
                          "pass": bool(chosen.auroc_max <= AUROC_MAX)},
            "reproducible": {**repro, "pass": repro["identical"]}}
    gate["pass"] = all(g["pass"] for g in gate.values())
    summary = {"data": {"rows": {s: len(data.t[s]["x"]) for s in data.t}, "mi_state_counts": counts,
                        "scale_mv": data.scale, "sampling": "1/sqrt(count) of MI state"},
               "model": {"params": n_params(model), "z_dim": Z_DIM, "c_dim": int(data.t["train"]["c"].shape[1]),
                         "batch": BATCH, "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
               "grid": [list(g) for g in GRID], "selection_rule": "auroc_max <= 0.55, then lowest val_fvu_per_lead",
               "chosen": chosen.run, "st_shift_check": shift, "gate_cf4": gate}
    (DOC / "cf4_01_train.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"chosen": chosen.run, "gate_cf4": gate, "st_shift": shift["lateral_st_shift_mv_remove_mi"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
