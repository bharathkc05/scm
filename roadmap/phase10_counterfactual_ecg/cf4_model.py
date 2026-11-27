"""Phase 10, CF4 — Track B model: conditional beta-VAE on 12-lead median beats, with an adversary on z.

Input   x : median beat, 12 leads x 400 samples (500 Hz, QRS onset at sample 100), mV / global train scale.
Condition c (decoder and encoder):
    MI state one-hot  : none, nstemi, stemi_anterior, stemi_inferior, stemi_other, type2, other
    age (z-scored), sex, diabetes, hypertension, CKD, prior MI (any earlier MI episode or prior-MI code)
    heart rate (z-scored): held fixed in counterfactuals, which change morphology, not rhythm
Latent  z : what is specific to the patient. A gradient-reversal adversary predicts the MI state from z, so MI
            information has to flow through c (assumption CF-A2).
Counterfactual: z = mu(x, c_obs); x_cf = decoder(z, c_obs with the MI state replaced).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from roadmap.config import DATA_OUT

OUT = DATA_OUT / "phase10"
MI_STATES = ["none", "nstemi", "stemi_anterior", "stemi_inferior", "stemi_other", "type2", "other"]
COVS = ["age_z", "sex_male", "diabetes", "hypertension", "ckd", "prior_mi_any", "hr_z"]
C_DIM = len(MI_STATES) + len(COVS)
N_LEADS, N_SAMPLES, QRS_ON = 12, 400, 100


# ----------------------------------------------------------------------------- data
def build_rows() -> pd.DataFrame:
    """One row per QC-passing MIMIC median beat in the Phase 10 train / validation / test splits (pair patients are
    in pair_test / pair_val and therefore excluded), with its condition."""
    from roadmap.phase10_counterfactual_ecg.cf3_01_track_a import episode_covs
    bi = pd.read_parquet(OUT / "beats_index.parquet")
    bi = bi[(bi.dataset == "mimic") & bi.qc_pass]
    idx = pd.read_parquet(OUT / "ecg_index.parquet")
    ep = pd.read_parquet(OUT / "episodes.parquet")
    split = pd.read_parquet(OUT / "cf_splits.parquet").set_index("subject_id").cf_split.astype(object)
    d = bi[["row", "record_id", "hr", "j_offset_ms"]].merge(idx, left_on="record_id", right_on="study_id")
    d["split"] = d.subject_id.map(split)
    d = d[d.split.isin(["train", "validation", "test"])].copy()

    # MI state: during-MI ECGs take the type of their episode; every other ECG is "none"
    e = episode_covs(ep).set_index("hadm_id")
    during = d.during_hadm_id.notna()
    d["mi_state"] = "none"
    d.loc[during, "mi_state"] = d.loc[during, "during_hadm_id"].map(e.cf_type).to_numpy()
    # covariates: the episode's for during-MI ECGs (as in Track A), as of the ECG otherwise
    d["age"] = d.age_at_ecg.astype(float)
    for col, src in (("diabetes", "diabetes_primary"), ("hypertension", "hypertension_primary"),
                     ("ckd", "ckd_primary"), ("prior_mi", "prior_mi_primary")):
        d[col] = d[col].astype(float)
        d.loc[during, col] = d.loc[during, "during_hadm_id"].map(e[src]).to_numpy()
    d.loc[during, "age"] = d.loc[during, "during_hadm_id"].map(e.age).to_numpy()
    # prior MI also counts any MI episode discharged before this ECG (the as-of admission can miss it)
    first_disch = ep.groupby("subject_id").dischtime.min()
    earlier = d.subject_id.map(first_disch) < d.ecg_time
    d["prior_mi_any"] = ((d.prior_mi.fillna(0) > 0) | earlier.fillna(False)).astype(float)
    d[["sex_male", "diabetes", "hypertension", "ckd"]] = d[["sex_male", "diabetes", "hypertension", "ckd"]].astype(
        float).fillna(0)
    d["age_z"] = (d.age - 65.0) / 15.0
    d["hr_z"] = (d.hr - 75.0) / 20.0
    keep = ["row", "record_id", "subject_id", "split", "role", "mi_state", "age", "hr", "j_offset_ms"] + COVS
    return d[keep].dropna(subset=COVS).sort_values("row").reset_index(drop=True)


def condition(df: pd.DataFrame, mi_state: str | None = None) -> np.ndarray:
    """Condition matrix; mi_state overrides the MI state for every row (the counterfactual action)."""
    states = df.mi_state.to_numpy() if mi_state is None else np.full(len(df), mi_state)
    onehot = (states[:, None] == np.array(MI_STATES)[None, :]).astype(np.float32)
    return np.column_stack([onehot, df[COVS].to_numpy(np.float32)])


def load_beats(rows: np.ndarray, block: int = 16384) -> np.ndarray:
    """Median beats for sorted beats.h5 rows, read in sequential blocks -> float32 [n, 12, 400] in mV."""
    import h5py
    rows = np.asarray(rows)
    assert (np.diff(rows) > 0).all(), "rows must be sorted and unique"
    out = np.empty((len(rows), N_LEADS, N_SAMPLES), np.float32)
    with h5py.File(OUT / "beats.h5", "r") as h:
        ds = h["beats"]
        for s in range(0, ds.shape[0], block):
            sel = (rows >= s) & (rows < s + block)
            if sel.any():
                out[sel] = ds[s:min(s + block, ds.shape[0])][rows[sel] - s].astype(np.float32)
    return out


# ----------------------------------------------------------------------------- model
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class FiLMUp(nn.Module):
    """Upsample x2, then GroupNorm modulated by the condition (FiLM), then LeakyReLU."""

    def __init__(self, cin: int, cout: int, c_dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, cout, affine=False)
        self.film = nn.Linear(c_dim, 2 * cout)

    def forward(self, h, c):
        h = self.norm(self.conv(h))
        g, b = self.film(c).unsqueeze(-1).chunk(2, dim=1)
        return F.leaky_relu(h * (1 + g) + b, 0.2)


class CondVAE(nn.Module):
    def __init__(self, z_dim: int = 32, c_dim: int = C_DIM, n_mi: int = len(MI_STATES)):
        super().__init__()
        self.z_dim = z_dim
        ch = [N_LEADS, 64, 128, 256, 256]                  # 400 -> 200 -> 100 -> 50 -> 25
        enc = []
        for i in range(4):
            enc += [nn.Conv1d(ch[i], ch[i + 1], kernel_size=7 if i == 0 else 5, stride=2,
                              padding=3 if i == 0 else 2), nn.GroupNorm(8, ch[i + 1]), nn.LeakyReLU(0.2)]
        self.enc = nn.Sequential(*enc)
        self.enc_fc = nn.Sequential(nn.Linear(256 * 25 + c_dim, 512), nn.LeakyReLU(0.2))
        self.mu, self.logvar = nn.Linear(512, z_dim), nn.Linear(512, z_dim)
        self.dec_fc = nn.Sequential(nn.Linear(z_dim + c_dim, 512), nn.LeakyReLU(0.2), nn.Linear(512, 256 * 25))
        self.up = nn.ModuleList([FiLMUp(256, 256, c_dim), FiLMUp(256, 128, c_dim), FiLMUp(128, 64, c_dim),
                                 FiLMUp(64, 64, c_dim)])
        self.out = nn.Conv1d(64, N_LEADS, kernel_size=7, padding=3)
        self.adv = nn.Sequential(nn.Linear(z_dim, 128), nn.LeakyReLU(0.2), nn.Linear(128, 128), nn.LeakyReLU(0.2),
                                 nn.Linear(128, n_mi))

    def encode(self, x, c):
        h = self.enc_fc(torch.cat([self.enc(x).flatten(1), c], dim=1))
        return self.mu(h), self.logvar(h).clamp(-10, 10)

    def decode(self, z, c):
        h = self.dec_fc(torch.cat([z, c], dim=1)).view(-1, 256, 25)
        for blk in self.up:
            h = blk(h, c)
        return self.out(h)

    def forward(self, x, c, lam: float = 0.0):
        mu, lv = self.encode(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        return self.decode(z, c), mu, lv, self.adv(GradReverse.apply(z, lam))

    @torch.no_grad()
    def counterfactual(self, x, c_obs, c_cf):
        mu, _ = self.encode(x, c_obs)
        return self.decode(mu, c_cf)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def fvu_per_lead(x: np.ndarray, xh: np.ndarray) -> float:
    """Mean over ECG-leads of SSE / SST with each lead of each beat centred on its own mean: comparable to the
    current VAE's loss on per-lead z-scored strips."""
    sst = ((x - x.mean(-1, keepdims=True)) ** 2).sum(-1)
    sse = ((x - xh) ** 2).sum(-1)
    ok = sst > 1e-8
    return float((sse[ok] / sst[ok]).mean())
