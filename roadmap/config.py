"""Single source of truth for every path used by the roadmap phases.

Import this in every phase script instead of hard-coding paths:
    from roadmap.config import PATHS, SEED
Run `python -m roadmap.config` to check that every input exists.
"""
from pathlib import Path

SEED = 20261121

REPO = Path(__file__).resolve().parents[1]                     # .../Capstone/SCM
RAW = Path(r"C:\Users\Acer\Desktop\Capstone\Capstone\data\raw")  # untouched source data
ROADMAP = REPO / "roadmap"

PATHS = {
    # --- Reused foundation (do not rebuild; roadmap says reuse) ---
    "master_train": REPO / "data/modeling/master/train.parquet",
    "master_validation": REPO / "data/modeling/master/validation.parquet",
    "master_test": REPO / "data/modeling/master/test.parquet",
    "split_manifest": REPO / "data/modeling/split_manifest.csv",
    "admissions_chronology": REPO / "data/longitudinal/admissions_chronology.parquet",
    "diagnoses_all": REPO / "data/longitudinal/diagnoses_all.parquet",
    "diagnoses_mi_codes": REPO / "data/longitudinal/diagnoses_mi_codes.parquet",
    "mi_admissions": REPO / "data/longitudinal/mi_admissions.parquet",
    "patients_demographics": REPO / "data/longitudinal/patients_demographics.parquet",
    "observation_end": REPO / "data/longitudinal/observation_end.parquet",
    # --- β-VAE (provenance check / latent audit) ---
    "vae_run_dir": REPO / "models/checkpoints/vae_zdim64_beta4.0_20260821_030245",
    "vae_best_model": REPO / "models/checkpoints/vae_zdim64_beta4.0_20260821_030245/best_model.pt",
    "ecg_embeddings_dir": REPO / "data/modeling/ecg_embeddings",
    "ecg_signals_h5": REPO / "data/processed/ecg_signals.h5",
    "vae_labels_dir": REPO / "data/modeling/labels",
    # --- Legacy label/feature files (audited in Phase 2, not trusted blindly) ---
    "ecg_features_full": REPO / "data/processed/extracted_ecg_features_with_study_id.csv",  # raw, all ECGs
    "legacy_troponin_labels": REPO / "data/processed/final_ecg_cohorts_troponin.csv",
    "mimic_duckdb": RAW.parent / "mimic_database.duckdb",  # read-only; has labevents, record_list
    # --- Raw MIMIC-IV v2.2 (only the tables the roadmap needs) ---
    "raw_admissions": RAW / "MIMIC-IV-2.2/hosp/admissions.csv",
    "raw_patients": RAW / "MIMIC-IV-2.2/hosp/patients.csv",
    "raw_diagnoses_icd": RAW / "MIMIC-IV-2.2/hosp/diagnoses_icd.csv",
    "raw_d_icd_diagnoses": RAW / "MIMIC-IV-2.2/hosp/d_icd_diagnoses.csv",
    "raw_omr": RAW / "MIMIC-IV-2.2/hosp/omr.csv",
    "raw_ecg_record_list": RAW / "MIMIC-IV-ECG-1.0/record_list.csv",
    # --- Phase 10 (counterfactual ECGs): raw waveforms, ECG-linked diagnoses, external data ---
    "raw_ecg_dir": RAW / "MIMIC-IV-ECG-1.0",
    "raw_ecg_ext_icd": RAW / "MIMIC-IV-ECG-Ext-ICD-1.0.1/records_w_diag_icd10.csv",
    "ptbxl_dir": RAW / "PTB-XL-1.0.3",
    "ptbxl_plus_dir": RAW / "PTB-XL+-1.0.1",
}

# Patient-level outputs go here (under data/, so git-ignored). Reports/aggregates go in PHASE_DIRS.
DATA_OUT = REPO / "data/roadmap"

# One output folder per phase; each phase also writes GATE_DECISION.md.
PHASE_DIRS = {
    1: ROADMAP / "phase1_questions",
    2: ROADMAP / "phase2_data_audit",
    3: ROADMAP / "phase3_dataset",
    4: ROADMAP / "phase4_graph",
    5: ROADMAP / "phase5_estimation",
    6: ROADMAP / "phase6_novelty",
    7: ROADMAP / "phase7_stress_test",
    8: ROADMAP / "phase8_paper",
    9: ROADMAP / "phase9_demo",
    10: ROADMAP / "phase10_counterfactual_ecg",
}


def check() -> bool:
    ok = True
    for name, p in PATHS.items():
        exists = p.exists()
        ok &= exists
        print(f"{'OK  ' if exists else 'MISSING'}  {name:24s} {p}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
