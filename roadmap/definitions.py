"""Locked variable definitions for the roadmap (Phase 1).

Every later phase must import these instead of re-typing code lists.
ICD codes in MIMIC-IV v2.2 are stored without dots (e.g. 'I214', '41071').
"""
from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# Outcome / Layer 2 exposure: acute MI coded at the current admission
# --------------------------------------------------------------------------
# ICD-9 410.xy : x = site, y = episode (0 unspecified, 1 initial, 2 subsequent).
#   410.x2 is a follow-up admission for an earlier MI -> NOT a new acute MI.
# ICD-10 I21.* (acute MI) and I22.* (subsequent MI within 4 weeks = new event).
ICD9_ACUTE_MI_PREFIX = "410"
ICD9_SUBSEQUENT_EPISODE_SUFFIX = "2"          # 5th character of 410.x2
ICD10_ACUTE_MI_PREFIXES = ("I21", "I22")

# Subtypes (for the STEMI/NSTEMI circularity check and type-2 sensitivity)
ICD9_NSTEMI_PREFIX = "4107"                   # subendocardial infarction
ICD9_UNSPECIFIED_PREFIX = "4109"
ICD10_STEMI_PREFIXES = ("I210", "I211", "I212", "I213")
ICD10_NSTEMI_PREFIXES = ("I214",)
ICD10_TYPE2_PREFIXES = ("I21A1",)
ICD10_OTHER_MI_PREFIXES = ("I219", "I21A9")

# Historical (old) MI - never an acute outcome
ICD9_OLD_MI = ("412",)
ICD10_OLD_MI_PREFIXES = ("I252",)

# --------------------------------------------------------------------------
# Chronic conditions (Layer 1 exposures, Layer 2 covariates)
# --------------------------------------------------------------------------
CHRONIC = {
    "diabetes": {9: ("250",), 10: ("E08", "E09", "E10", "E11", "E13")},
    "hypertension": {9: ("401", "402", "403", "404", "405"),
                     10: ("I10", "I11", "I12", "I13", "I15", "I16")},
    "ckd": {9: ("585",), 10: ("N18",)},
}

# --------------------------------------------------------------------------
# Active comparator for Layer 2: chest pain without acute MI
# --------------------------------------------------------------------------
CHEST_PAIN = {9: ("7865",), 10: ("R07",)}

# --------------------------------------------------------------------------
# Never use as covariates (roadmap "What not to use")
# --------------------------------------------------------------------------
FORBIDDEN_COVARIATES = (
    "Label_Troponin", "is_MI", "index_mi_time", "time_diff_hours",
    # in-admission treatments/support are post-onset (mediators / colliders)
    "statin_during", "statin_after", "beta_blocker_during", "acei_during",
    "aspirin_during", "vasopressor_during", "mech_vent_during",
)


def fix_lead_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Swap aVL <-> aVF in ECGdeli feature names (Phase 5 finding).

    MIMIC-IV-ECG WFDB files store leads as I, II, III, aVR, aVF, aVL, V1..V6, but
    src/matlab/extract_ecg_features.m labelled rows positionally as I, II, III, aVR, aVL, aVF.
    Verified against the limb-lead identities: the column labelled aVL tracks (II+III)/2 (r 0.87-0.90).
    Apply to every frame read from the raw ECGdeli feature file. Values are unchanged; only names swap.
    """
    def swap(c: str) -> str:
        return c.replace("_aVL", "\0").replace("_aVF", "_aVL").replace("\0", "_aVF")
    return df.rename(columns={c: swap(c) for c in df.columns})


def _norm(codes: pd.Series) -> pd.Series:
    return codes.astype(str).str.replace(".", "", regex=False).str.strip().str.upper()


def _starts(codes: pd.Series, prefixes) -> pd.Series:
    return codes.str.startswith(tuple(prefixes))


def flag_codes(dx: pd.DataFrame) -> pd.DataFrame:
    """Add boolean code-class columns to a diagnoses frame.

    Expects columns `icd_code`, `icd_version`. Returns a copy with:
    acute_mi, mi_stemi, mi_nstemi, mi_type2, mi_other_unspecified,
    mi_subsequent_episode_icd9, old_mi, diabetes, hypertension, ckd, chest_pain.
    """
    out = dx.copy()
    c = _norm(out["icd_code"])
    v9 = out["icd_version"].astype(int) == 9
    v10 = ~v9

    icd9_410 = v9 & c.str.startswith(ICD9_ACUTE_MI_PREFIX)
    icd9_subseq = icd9_410 & (c.str.len() == 5) & c.str.endswith(ICD9_SUBSEQUENT_EPISODE_SUFFIX)
    icd10_acute = v10 & _starts(c, ICD10_ACUTE_MI_PREFIXES)

    out["mi_subsequent_episode_icd9"] = icd9_subseq
    out["acute_mi"] = (icd9_410 & ~icd9_subseq) | icd10_acute

    out["mi_nstemi"] = (v9 & c.str.startswith(ICD9_NSTEMI_PREFIX) & ~icd9_subseq) | (
        v10 & _starts(c, ICD10_NSTEMI_PREFIXES))
    out["mi_type2"] = v10 & _starts(c, ICD10_TYPE2_PREFIXES)
    out["mi_other_unspecified"] = (v9 & c.str.startswith(ICD9_UNSPECIFIED_PREFIX) & ~icd9_subseq) | (
        v10 & (_starts(c, ICD10_OTHER_MI_PREFIXES) | c.str.startswith("I22")))
    out["mi_stemi"] = (out["acute_mi"] & v9 & ~c.str.startswith(ICD9_NSTEMI_PREFIX)
                       & ~c.str.startswith(ICD9_UNSPECIFIED_PREFIX)) | (v10 & _starts(c, ICD10_STEMI_PREFIXES))

    out["old_mi"] = (v9 & c.isin(ICD9_OLD_MI)) | (v10 & _starts(c, ICD10_OLD_MI_PREFIXES))
    for name, lists in CHRONIC.items():
        out[name] = (v9 & _starts(c, lists[9])) | (v10 & _starts(c, lists[10]))
    out["chest_pain"] = (v9 & _starts(c, CHEST_PAIN[9])) | (v10 & _starts(c, CHEST_PAIN[10]))
    return out


def age_at_admission(admittime: pd.Series, anchor_age: pd.Series, anchor_year: pd.Series) -> pd.Series:
    """MIMIC-IV age: anchor_age + (admission year - anchor_year)."""
    return anchor_age + (pd.to_datetime(admittime).dt.year - anchor_year)
