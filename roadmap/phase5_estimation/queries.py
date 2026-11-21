"""Locked Phase 5 analysis queries (from Phase 1 estimands as revised in Phase 4)."""
import json

from roadmap.config import PHASE_DIRS

S, D, H, K, P = "sex_male", "diabetes_primary", "hypertension_primary", "ckd_primary", "prior_mi_primary"

# Age enters every model through the spline; `covs` lists the binary covariates.
L1_QUERIES = [
    {"exposure": D, "label": "diabetes | age, sex", "covs": [S], "role": "primary",
     "note": "bracket lower set: valid if D -> H without shared cause"},
    {"exposure": D, "label": "diabetes | age, sex, HTN", "covs": [S, H], "role": "bracket",
     "note": "bracket upper set: valid under shared cause / H -> D"},
    {"exposure": H, "label": "hypertension | age, sex, diabetes", "covs": [S, D], "role": "primary",
     "note": "valid under revised graph (Q2)"},
    {"exposure": H, "label": "hypertension | age, sex (roadmap)", "covs": [S], "role": "sensitivity",
     "note": "valid only if H -> D"},
    {"exposure": K, "label": "CKD | age, sex, D, H", "covs": [S, D, H], "role": "primary", "note": ""},
    {"exposure": K, "label": "CKD | age, sex, D, H, prior MI", "covs": [S, D, H, P], "role": "sensitivity",
     "note": "valid if prior MI -> CKD"},
    {"exposure": P, "label": "prior MI | age, sex, D, H, CKD", "covs": [S, D, H, K], "role": "primary",
     "note": "descriptive only"},
]

L2_COVS = [S, D, H, K, P]


def l2_outcomes() -> dict:
    f = json.loads((PHASE_DIRS[2] / "ecg_feature_list.json").read_text())
    return {
        "point": [f"w_{c}" for c in f["point_53"]],                          # primary family (FDR)
        "iqr": [f"w_{c}" for c in f["iqr_41"]],                              # secondary family
        "st_composite": [f"w_{c}" for c in ["ST_inferior_mean", "ST_anterior_mean", "ST_lateral_mean",
                                            "ST_max_elevation", "ST_max_depression"]],
        "raw_point": f["point_53"],                                          # sensitivity (P4)
        "latent": [f"z_ecg_{i}" for i in range(1, 65)],                      # Phase 6 latent audit
    }


L2_MACHINE = ["machine_qrs_ms", "machine_rr_ms"]                              # validation outcomes (have NaN)
