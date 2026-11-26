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


# --------------------------------------------------------------------------
# Phase 10: individual counterfactual ECGs (locked 2026-11-26, CF0)
# --------------------------------------------------------------------------
# MI type per admission, by priority when an admission carries several MI codes.
MI_KIND_PRIORITY = ("stemi", "nstemi", "type2", "other_unspecified")

# Infarct territory from the site digit (ICD-9 410.x) or subcategory (ICD-10 I21.x / I22.x).
# detail = what the code says; coarse = the grouping used for modelling (anterolateral -> anterior,
# inferolateral / inferoposterior -> inferior, as the dominant wall named first).
ICD9_TERRITORY = {  # 4th character of 410.x
    "0": ("anterolateral", "anterior"), "1": ("anterior", "anterior"), "2": ("inferolateral", "inferior"),
    "3": ("inferoposterior", "inferior"), "4": ("inferior", "inferior"), "5": ("lateral", "lateral"),
    "6": ("posterior", "posterior"), "7": ("subendocardial", "nstemi"), "8": ("other_site", "other_site"),
    "9": ("unspecified", "unspecified"),
}
ICD10_TERRITORY = (  # (prefix, detail, coarse), first match wins: longer prefixes first
    ("I2101", "left_main", "anterior"), ("I2102", "lad", "anterior"), ("I2109", "anterior_other", "anterior"),
    ("I2111", "rca", "inferior"), ("I2119", "inferior_other", "inferior"),
    ("I2121", "lcx", "lateral"), ("I2129", "other_site", "other_site"),
    ("I21A1", "type2", "type2"), ("I21A9", "other_type", "unspecified"),
    ("I210", "anterior", "anterior"), ("I211", "inferior", "inferior"), ("I212", "other_site", "other_site"),
    ("I213", "unspecified_site", "unspecified"), ("I214", "subendocardial", "nstemi"), ("I219", "unspecified", "unspecified"),
    ("I220", "anterior", "anterior"), ("I221", "inferior", "inferior"), ("I222", "subendocardial", "nstemi"),
    ("I228", "other_site", "other_site"), ("I229", "unspecified", "unspecified"),
)
TERRITORY_PRIORITY = ("anterior", "inferior", "lateral", "posterior", "other_site", "unspecified", "nstemi", "type2")


def mi_territory(icd_code: str, icd_version: int) -> tuple[str, str] | None:
    """(detail, coarse) territory of one acute-MI code, or None for non-MI codes and ICD-9 410.x2."""
    c = str(icd_code).replace(".", "").strip().upper()
    if int(icd_version) == 9:
        if not c.startswith(ICD9_ACUTE_MI_PREFIX) or len(c) < 4:
            return None
        if len(c) == 5 and c.endswith(ICD9_SUBSEQUENT_EPISODE_SUFFIX):
            return None
        return ICD9_TERRITORY.get(c[3])
    for prefix, detail, coarse in ICD10_TERRITORY:
        if c.startswith(prefix):
            return detail, coarse
    return None


# Within-patient pairs: an ECG recorded during an acute-MI stay (ED arrival to discharge) matched with the
# same patient's ECG outside any MI stay -- before (more than 1 day before arrival) or after (more than 30 days
# after discharge and more than 1 day before any later MI arrival).
PAIR_BEFORE_GAP_HOURS = 24
PAIR_AFTER_GAP_DAYS = 30

# Troponin onset anchor: first troponin above the lab's own upper reference limit (else flagged abnormal)
# between 12 h before ED arrival and discharge.
TROPONIN_ITEMIDS = (51002, 51003, 52642)
TROPONIN_LOOKBACK_HOURS = 12

# Coronary procedures during the stay (ICD-9 procedure / ICD-10-PCS prefixes).
REVASC_CODES = {
    9: ("0066", "3601", "3602", "3603", "3604", "3605", "3606", "3607", "3609",       # PCI / stents
        "3610", "3611", "3612", "3613", "3614", "3615", "3616", "3617", "3619",       # CABG
        "9910"),                                                                      # thrombolysis
    10: ("0270", "0271", "0272", "0273",                                              # coronary dilation (PCI)
         "0210", "0211", "0212", "0213",                                              # coronary bypass
         "3E03317", "3E04317", "3E06317"),                                            # thrombolysis
}

# Phase 10 split: pair patients are held out (80% pair_test, 20% pair_val, stratified by MI kind); every
# other patient keeps their Phase 3 split. Phase 10 models are fit on `train` only, so models from
# Phases 5-6 (fit on the Phase 3 train split, which contains some pair patients) must not be reused.
CF_PAIR_SHARES = {"pair_test": 0.80, "pair_val": 0.20}
