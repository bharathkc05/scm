"""Generate server assets from the project's own definitions (build step, not run by the server).

M2: local_mcp/assets/variables.csv, the data dictionary behind describe_variable and name matching.
Run from the repo root, then rebuild the manifest:
    <project python> -X utf8 local_mcp/export_assets.py
    <project python> -X utf8 local_mcp/build_manifest.py
"""
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_mcp.paths import PKG, ROOT  # noqa: E402
from roadmap import definitions as D  # noqa: E402

ASSETS = PKG / "assets"
COLUMNS = ["name", "label", "synonyms", "category", "layer", "role", "definition", "icd9", "icd10", "units",
           "family", "grp", "source"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
WAVES = {"ST_Elev": ("ST level", ["st", "st elevation", "st depression", "st segment", "st level"], "ST", "mV"),
         "Q_Amp": ("Q-wave amplitude", ["q", "q wave", "q amplitude", "q wave amplitude"], "Q", "mV"),
         "R_Amp": ("R-wave amplitude", ["r", "r wave", "r amplitude", "r wave amplitude"], "R", "mV"),
         "S_Amp": ("S-wave amplitude", ["s", "s wave", "s amplitude", "s wave amplitude"], "S", "mV"),
         "T_Amp": ("T-wave amplitude", ["t", "t wave", "t amplitude", "t wave amplitude"], "T", "mV")}
ECG_COMMON = ("Measured by ECGdeli on the earliest ECG with extracted features in the admission; winsorized at the "
              "training-split 0.5th/99.5th percentiles.")


def icd(prefixes) -> str:
    """Stored codes have no dots (I214, 7865); show the clinical form (I21.4*, 786.5*)."""
    return ", ".join(f"{p[:3]}.{p[3:]}*" if len(p) > 3 else f"{p}*" for p in prefixes)


def row(**kw) -> dict:
    out = {c: "" for c in COLUMNS}
    out.update({k: ("; ".join(v) if isinstance(v, list) else v) for k, v in kw.items()})
    return out


def clinical_rows() -> list[dict]:
    rows = [
        row(name="acute_mi", label="Acute MI at this admission",
            synonyms=["acute myocardial infarction", "myocardial infarction", "mi", "ami", "heart attack", "acute mi"],
            category="exposure_outcome", layer="1, 2", role="outcome (Layer 1); exposure (Layer 2)",
            definition="Acute MI coded at the current admission (any diagnosis position). ICD-9 410.x2 (subsequent "
                       "episode of care) and old-MI codes are excluded.",
            icd9="410.x0, 410.x1", icd10=icd(D.ICD10_ACUTE_MI_PREFIXES), source="roadmap/definitions.py (Phase 1)"),
        row(name="mi_stemi", label="STEMI", synonyms=["st elevation myocardial infarction", "stemi"],
            category="mi_subtype", layer="2", role="subtype (sensitivity analyses)",
            definition="Acute MI coded as ST-elevation MI.", icd9="410.0-410.6, 410.8", icd10=icd(D.ICD10_STEMI_PREFIXES),
            source="roadmap/definitions.py"),
        row(name="mi_nstemi", label="NSTEMI", synonyms=["non st elevation myocardial infarction", "nstemi", "subendocardial mi"],
            category="mi_subtype", layer="2", role="subtype; NSTEMI-only analysis is the label-circularity check",
            definition="Acute MI coded as non-ST-elevation (subendocardial) MI.", icd9="410.7",
            icd10=icd(D.ICD10_NSTEMI_PREFIXES), source="roadmap/definitions.py"),
        row(name="mi_type2", label="Type 2 MI", synonyms=["type 2 myocardial infarction", "demand ischemia mi", "type ii mi"],
            category="mi_subtype", layer="2", role="subtype (excluded in a sensitivity analysis)",
            definition="Supply-demand MI; ICD-10 only.", icd10=icd(D.ICD10_TYPE2_PREFIXES), source="roadmap/definitions.py"),
        row(name="old_mi", label="Old (historical) MI code", synonyms=["old mi", "historical mi", "previous mi code"],
            category="history", layer="1, 2", role="part of the prior-MI definition; never an outcome",
            definition="Code for a past MI.", icd9=", ".join(D.ICD9_OLD_MI), icd10=icd(D.ICD10_OLD_MI_PREFIXES),
            source="roadmap/definitions.py"),
        row(name="recent_mi_followup", label="ICD-9 410.x2 follow-up admission",
            synonyms=["410.x2", "subsequent episode of care", "mi follow up admission"],
            category="exclusion", layer="2", role="excluded from Layer 2 controls",
            definition="Follow-up admission for an earlier MI (ICD-9 410.x2) without a new acute MI code.", icd9="410.x2",
            source="Phase 3 decision P1"),
        row(name="age", label="Age at admission (years)", synonyms=["age", "patient age"], category="covariate",
            layer="1, 2", role="adjusted for in every analysis (cubic spline, 5 df)",
            definition="anchor_age + (admission year - anchor_year), MIMIC-IV convention.", units="years",
            source="Phase 1"),
        row(name="sex_male", label="Sex (male)", synonyms=["sex", "gender", "male"], category="covariate", layer="1, 2",
            role="adjusted for in every analysis", definition="Recorded sex; 1 = male.", source="Phase 1"),
        row(name="prior_mi", label="Prior MI", synonyms=["prior mi", "previous mi", "history of mi", "prior myocardial infarction"],
            category="exposure_covariate", layer="1, 2", role="descriptive exposure (Layer 1); covariate (Layer 2)",
            definition="Acute MI code on an earlier admission, or an old-MI code on this or an earlier admission. "
                       "Descriptive marker of coronary disease, not a cause.", source="Phase 1"),
        row(name="chest_pain", label="Chest-pain code", synonyms=["chest pain", "chest pain comparator"],
            category="comparator", layer="2", role="active comparator group (chest pain without acute MI)",
            definition="Chest-pain diagnosis code at the admission.", icd9=icd(D.CHEST_PAIN[9]), icd10=icd(D.CHEST_PAIN[10]),
            source="roadmap/definitions.py"),
        row(name="timing_class", label="ECG timing relative to MI onset",
            synonyms=["timing", "after onset", "ecg timing", "onset timing"], category="design", layer="2",
            role="selects after-onset MI ECGs for the primary analysis",
            definition="after = MI is the principal diagnosis of an unplanned admission; uncertain = MI coded "
                       "secondary or planned admission; before = ECG from an earlier admission of a patient who later "
                       "has an MI (negative control). Admission context, not troponin.", source="Phase 2"),
        row(name="hours_from_arrival", label="Hours from arrival to ECG", synonyms=["time to ecg", "hours to ecg"],
            category="modifier", layer="2", role="descriptive only (a consequence of MI and presentation)",
            definition="Hours from ED registration (or admission) to the analysed ECG.", units="hours", source="Phase 4 Q8"),
        row(name="n_prior_admissions", label="Number of earlier admissions (look-back)",
            synonyms=["lookback", "look back", "prior admissions"], category="sensitivity", layer="1",
            role="sensitivity only (restriction or covariate)",
            definition="Count of the patient's earlier admissions; 0 means history flags are 0 by construction.",
            source="Phase 2/4"),
    ]
    names = {"diabetes": ("Diabetes", ["diabetes", "diabetes mellitus", "dm"]),
             "hypertension": ("Hypertension", ["hypertension", "high blood pressure", "htn"]),
             "ckd": ("Chronic kidney disease", ["ckd", "chronic kidney disease", "kidney disease", "renal disease"])}
    for key, (label, syn) in names.items():
        lists = D.CHRONIC[key]
        rows.append(row(name=key, label=label, synonyms=syn, category="exposure_covariate", layer="1, 2",
                        role="exposure (Layer 1); covariate (Layer 2)",
                        definition=f"{label} code on an earlier admission or a chronic code at the current admission "
                                   "(primary flag); prior-admission-only flag is a sensitivity version.",
                        icd9=icd(lists[9]), icd10=icd(lists[10]), source="roadmap/definitions.py"))
    for f in D.FORBIDDEN_COVARIATES:
        rows.append(row(name=f, label=f"{f} (never a covariate)", synonyms=[f.lower().replace("_", " ")],
                        category="forbidden", role="forbidden",
                        definition="Post-onset or troponin-derived variable; never used as a covariate (roadmap rule).",
                        source="roadmap/definitions.py FORBIDDEN_COVARIATES"))
    return rows


def ecg_rows() -> list[dict]:
    feats = json.loads((ROOT / "roadmap/phase2_data_audit/ecg_feature_list.json").read_text())
    rows = []
    for name in feats["point_53"] + feats["iqr_41"]:
        base = name.removesuffix("_iqr")
        is_iqr = name.endswith("_iqr")
        if base == "QRS_Dur_Global":
            label, syn, grp, units = "QRS duration (global)", ["qrs duration", "qrs width", "qrs"], "QRS", "ms"
            extra = " Not validated: machine-measured QRS showed no MI effect."
        elif base == "RR_Mean_Global":
            label, syn, grp, units = "Mean RR interval", ["rr interval", "rr", "cycle length", "heart rate interval"], "RR", "ms"
            extra = " Equally present before MI (negative control), so not MI-specific."
        else:
            wave, lead = base.rsplit("_", 1)
            wlabel, wsyn, grp, units = WAVES[wave]
            label = f"{wlabel}, lead {lead}"
            lo = lead.lower()
            syn = [f"{s} {lo}" for s in wsyn] + [f"{s} lead {lo}" for s in wsyn] + [f"{lo} {wsyn[1]}"]
            extra = " ST level = amplitude 60 ms after the J point minus the amplitude at QRS onset." if wave == "ST_Elev" else ""
            if lead in ("aVL", "aVF"):
                extra += " Lead label corrected (raw ECGdeli output had aVL and aVF swapped)."
        if is_iqr:
            label, syn = f"{label}: beat-to-beat IQR", [f"{s} variability" for s in syn[:3]] + [name.lower()]
            fam, role = "iqr", "secondary ECG outcome (beat-to-beat spread)"
        else:
            fam, role = "point", "primary ECG outcome (Layer 2)"
        rows.append(row(name=name, label=label, synonyms=syn, category="ecg_feature", layer="2", role=role,
                        definition=ECG_COMMON + extra, units=units, family=fam, grp=grp, source="ECGdeli; Phase 3"))
    rows.append(row(name="QRS_Dur_Global_count", label="Beats detected", synonyms=["beat count", "beats detected"],
                    category="ecg_quality", layer="2", role="quality flag; not an adjustment covariate (Phase 1 D5)",
                    definition="Number of beats ECGdeli detected in the 10-s ECG.", units="beats", source="Phase 1/3"))
    comps = {"ST_inferior_mean": ("Inferior ST level (mean of II, III, aVF)", ["inferior st", "st inferior"]),
             "ST_anterior_mean": ("Anterior ST level (mean of V1-V4)", ["anterior st", "st anterior"]),
             "ST_lateral_mean": ("Lateral ST level (mean of I, aVL, V5, V6)",
                                 ["lateral st", "st lateral", "lateral st depression", "lateral st level"]),
             "ST_max_elevation": ("Maximum ST elevation (non-aVR leads)", ["max st elevation", "maximum st elevation", "st elevation max"]),
             "ST_max_depression": ("Maximum ST depression (non-aVR leads)", ["max st depression", "maximum st depression", "st depression max"])}
    for name, (label, syn) in comps.items():
        rows.append(row(name=name, label=label, synonyms=syn, category="ecg_composite", layer="2",
                        role="territory composite outcome (Layer 2)", definition=f"{label}, from the ECGdeli ST levels. " + ECG_COMMON,
                        units="mV", family="st_composite", grp="ST", source="Phase 3"))
    rows.append(row(name="machine_qrs_ms", label="Machine-measured QRS duration", synonyms=["machine qrs", "cart qrs"],
                    category="validation", layer="2", role="validation outcome", definition="QRS duration from the ECG cart's own measurements.",
                    units="ms", family="machine", grp="machine", source="MIMIC-IV-ECG machine_measurements"))
    rows.append(row(name="machine_rr_ms", label="Machine-measured RR interval", synonyms=["machine rr", "cart rr"],
                    category="validation", layer="2", role="validation outcome", definition="RR interval from the ECG cart's own measurements.",
                    units="ms", family="machine", grp="machine", source="MIMIC-IV-ECG machine_measurements"))
    for k in range(1, 65):
        rows.append(row(name=f"z_ecg_{k}", label=f"beta-VAE latent dimension {k}",
                        synonyms=[f"z{k}", f"latent {k}", f"z ecg {k}", f"latent dimension {k}"], category="latent",
                        layer="2", role="latent audit (secondary)",
                        definition="Mean of the 64-dimensional beta-VAE encoding of the analysed ECG (unsupervised; the encoder "
                                   "saw acute-MI ECGs during training).", units="latent units", family="latent",
                        grp="latent", source="Phase 2/3/6"))
    return rows


def main() -> Path:
    ASSETS.mkdir(exist_ok=True)
    rows = clinical_rows() + ecg_rows()
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names)), "duplicate dictionary names"
    out = ASSETS / "variables.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return out


if __name__ == "__main__":
    p = main()
    print(f"wrote {p} ({sum(1 for _ in p.open(encoding='utf-8')) - 1} variables)")
