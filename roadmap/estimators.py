"""Shared estimation machinery (Phases 5-7).

Doubly robust (AIPW) estimators with patient-grouped cross-fitting and patient-level bootstrap.
- Layer 1: binary exposure -> binary outcome, ATE risk difference and risk ratio (logistic nuisances).
- Layer 2: binary exposure -> many continuous outcomes, ATT (primary) and ATE (multi-output least squares
  outcome models, so all outcomes share one propensity and are fitted in one pass).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

N_FOLDS = 5
CLIP = (0.01, 0.99)


# ----------------------------------------------------------------------------- design
@dataclass(frozen=True)
class Spec:
    age_df: int
    interactions: bool
    age_by_all: bool = False      # age spline x every binary covariate

    @property
    def name(self) -> str:
        return f"age_df{self.age_df}{'_int' if self.interactions else ''}{'_agexall' if self.age_by_all else ''}"


SPECS = [Spec(3, False), Spec(5, False), Spec(7, False), Spec(5, True), Spec(7, True), Spec(5, True, True)]


class Design:
    """Age spline + binary covariates (+ optional age x sex, sex x flag, flag x flag interactions).
    Fitted (knots, scaling) on the rows passed to fit() only."""

    def __init__(self, spec: Spec, binaries: list[str]):
        self.spec, self.binaries = spec, binaries

    def fit(self, df: pd.DataFrame) -> "Design":
        # cubic, no bias column: n_knots + 1 basis columns = age_df
        self.spline = SplineTransformer(n_knots=self.spec.age_df - 1, degree=3, knots="quantile",
                                        include_bias=False).fit(df[["age"]])
        self.scaler = StandardScaler().fit(self._raw(df))
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        age = self.spline.transform(df[["age"]])
        b = df[self.binaries].to_numpy(dtype=float)
        cols = [age, b]
        if self.spec.age_by_all and b.shape[1]:
            cols.extend(age * b[:, [i]] for i in range(b.shape[1]))
        if self.spec.interactions and b.shape[1]:
            sex = df[["sex_male"]].to_numpy(dtype=float) if "sex_male" in self.binaries else None
            if sex is not None and not self.spec.age_by_all:
                cols.append(age * sex)
            k = b.shape[1]
            cols.append(np.column_stack([b[:, i] * b[:, j] for i in range(k) for j in range(i + 1, k)])
                        if k > 1 else np.empty((len(df), 0)))
        return np.column_stack(cols)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(self._raw(df))


def logistic(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=1e3, max_iter=2000).fit(x, y)


def ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
    return beta


def ols_predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x]) @ beta


# ----------------------------------------------------------------------------- folds
def patient_folds(subject_id: pd.Series, seed: int, k: int = N_FOLDS) -> np.ndarray:
    pats = np.sort(subject_id.unique())
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(pats, rng.permutation(len(pats)) % k))
    return subject_id.map(fold_of).to_numpy()


# ----------------------------------------------------------------------------- nuisances
def crossfit_binary(df: pd.DataFrame, exposure: str, outcome: str, covs: list[str], spec: Spec,
                    folds: np.ndarray) -> dict:
    """Layer 1 nuisances: e(L)=P(X=1|L), m1(L), m0(L)=P(Y=1|X=x,L) via arm-specific logistic models."""
    n = len(df)
    e, m0, m1 = np.empty(n), np.empty(n), np.empty(n)
    x, y = df[exposure].to_numpy(), df[outcome].to_numpy()
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        des = Design(spec, covs).fit(df[tr])
        xtr, xte = des.transform(df[tr]), des.transform(df[te])
        e[te] = logistic(xtr, x[tr]).predict_proba(xte)[:, 1]
        for arm, store in ((1, m1), (0, m0)):
            sel = x[tr] == arm
            store[te] = logistic(xtr[sel], y[tr][sel]).predict_proba(xte)[:, 1]
    return {"e": e, "m0": m0, "m1": m1}


def crossfit_continuous(df: pd.DataFrame, exposure: str, outcomes: np.ndarray, covs: list[str], spec: Spec,
                        folds: np.ndarray, need_m1: bool = True) -> dict:
    """Layer 2 nuisances: e(L) logistic; m0(L), m1(L) multi-output least squares (all outcomes at once)."""
    n = len(df)
    e = np.empty(n)
    m0 = np.empty_like(outcomes, dtype=float)
    m1 = np.empty_like(outcomes, dtype=float) if need_m1 else None
    x = df[exposure].to_numpy()
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        des = Design(spec, covs).fit(df[tr])
        xtr, xte = des.transform(df[tr]), des.transform(df[te])
        e[te] = logistic(xtr, x[tr]).predict_proba(xte)[:, 1]
        c = x[tr] == 0
        m0[te] = ols_predict(ols(xtr[c], outcomes[tr][c]), xte)
        if need_m1:
            t = x[tr] == 1
            m1[te] = ols_predict(ols(xtr[t], outcomes[tr][t]), xte)
    return {"e": e, "m0": m0, "m1": m1}


# ----------------------------------------------------------------------------- scores
def ate_binary_terms(x, y, e, m0, m1) -> dict:
    ec = np.clip(e, *CLIP)
    mu1 = m1 + x * (y - m1) / ec
    mu0 = m0 + (1 - x) * (y - m0) / (1 - ec)
    return {"mu1": mu1, "mu0": mu0, "clipped_share": float(((e < CLIP[0]) | (e > CLIP[1])).mean())}


def att_terms(x, y, e, m0) -> tuple[np.ndarray, np.ndarray]:
    """Numerator terms a_i (n x k) and denominator d_i (n) so that ATT = sum(a)/sum(d)."""
    ec = np.clip(e, None, CLIP[1])[:, None]
    xi = x[:, None]
    a = xi * (y - m0) - (1 - xi) * ec / (1 - ec) * (y - m0)
    return a, x.astype(float)


def ate_cont_terms(x, y, e, m0, m1) -> np.ndarray:
    ec = np.clip(e, *CLIP)[:, None]
    xi = x[:, None]
    return m1 - m0 + xi * (y - m1) / ec - (1 - xi) * (y - m0) / (1 - ec)


# ----------------------------------------------------------------------------- bootstrap
def patient_bootstrap_ratio(subject_id: np.ndarray, num: np.ndarray, den: np.ndarray, reps: int, seed: int,
                            chunk: int = 50) -> np.ndarray:
    """Bootstrap of sum(num)/sum(den), resampling patients with all their rows. num: (n, k), den: (n,).
    Nuisance predictions are held fixed. Returns (reps, k)."""
    codes, uniq = pd.factorize(subject_id)
    P = len(uniq)
    num_p = np.zeros((P, num.shape[1]))
    np.add.at(num_p, codes, num)
    den_p = np.bincount(codes, weights=den, minlength=P)
    rng = np.random.default_rng(seed)
    out = np.empty((reps, num.shape[1]))
    for s in range(0, reps, chunk):
        b = min(chunk, reps - s)
        w = rng.multinomial(P, np.full(P, 1.0 / P), size=b).astype(float)  # (b, P)
        out[s:s + b] = (w @ num_p) / (w @ den_p)[:, None]
    return out


def summarize(est: np.ndarray, boot: np.ndarray) -> pd.DataFrame:
    se = boot.std(axis=0, ddof=1)
    z = est / np.where(se > 0, se, np.nan)
    from scipy.stats import norm
    return pd.DataFrame({"estimate": est, "se_boot": se,
                         "ci_low": np.percentile(boot, 2.5, axis=0), "ci_high": np.percentile(boot, 97.5, axis=0),
                         "p_value": 2 * norm.sf(np.abs(z))})


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


# ----------------------------------------------------------------------------- diagnostics
def weighted_smd(df: pd.DataFrame, covs: list[str], x: np.ndarray, w: np.ndarray) -> dict:
    out = {}
    for c in covs:
        v = df[c].to_numpy(float)
        t, cn = x == 1, x == 0
        m1, m0 = np.average(v[t], weights=w[t]), np.average(v[cn], weights=w[cn])
        s = np.sqrt((v[t].var() + v[cn].var()) / 2)
        out[c] = float((m1 - m0) / s) if s > 0 else 0.0
    return out


def ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())
