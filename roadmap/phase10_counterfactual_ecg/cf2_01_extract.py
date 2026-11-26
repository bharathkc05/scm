"""Phase 10, CF2 — task lists for the ECGdeli extractor (MATLAB), pilot check, and merge of its shards.

The extraction itself is the legacy ECGdeli method, run in MATLAB by cf2_extract_ecgdeli.m. This script:

    --write-tasks   writes data/roadmap/phase10/cf2_tasks_pilot.csv (300 ECGs: 150 pair, 100 control, 50 PTB-XL)
                    and cf2_tasks_full.csv (every CF1-indexed MIMIC ECG + all PTB-XL, in priority order:
                    0 pair, 1 during-MI, 2 other MI-patient ECGs, 3 controls, 4 PTB-XL)
    --check DIR     checks a shard folder: success and QC rates, speed, lead order (limb-lead identities on the
                    median beats), ST from the median beats vs ECGdeli ST_Elev, and agreement with the legacy feature
                    file for the same ECGs -> cf2_00_pilot.json
    --merge DIR...  streams one or more stage shard folders into beats.h5 (float16 [N, 12, 400], mV, QRS onset at sample 100),
                    beats_index.parquet and ecgdeli_features.parquet; aggregate QC -> cf2_01_extract.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS, SEED
from roadmap.definitions import fix_lead_labels

OUT = DATA_OUT / "phase10"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
N_SAMPLES, QRS_ON = 400, 100           # QRS onset at sample index 100 (MATLAB 101), 500 Hz
QC = {"min_beats": 3, "min_template_corr": 0.80, "hr_range": (30, 180), "max_abs_mv": 5.0}


def tasks() -> pd.DataFrame:
    idx = pd.read_parquet(OUT / "ecg_index.parquet", columns=["study_id", "path", "role"])
    pairs = pd.read_parquet(OUT / "pairs.parquet", columns=["during_study_id", "before_study_id", "after_study_id"])
    pair_ids = set(pairs.during_study_id) | set(pairs.before_study_id.dropna().astype(int)) | set(
        pairs.after_study_id.dropna().astype(int))
    prio = np.where(idx.study_id.isin(pair_ids), 0, np.where(idx.role.str.contains("during"), 1,
                    np.where(idx.role == "control", 3, 2)))
    root = PATHS["raw_ecg_dir"]
    mimic = pd.DataFrame({"record_id": idx.study_id.astype("int64"), "dataset": "mimic",
                          "path": [str(root / p) for p in idx.path], "priority": prio})
    db = pd.read_csv(PATHS["ptbxl_dir"] / "ptbxl_database.csv", usecols=["ecg_id", "filename_hr"])
    ptb = pd.DataFrame({"record_id": db.ecg_id.astype("int64"), "dataset": "ptbxl",
                        "path": [str(PATHS["ptbxl_dir"] / f) for f in db.filename_hr], "priority": 4})
    t = pd.concat([mimic, ptb], ignore_index=True)
    t["path"] = t.path.str.replace("/", "\\", regex=False)
    return t.sort_values(["priority", "dataset", "record_id"], kind="stable").reset_index(drop=True)


def write_tasks() -> dict:
    t = tasks()
    pilot = pd.concat([t[t.priority == 0].sample(150, random_state=SEED), t[t.priority == 3].sample(100, random_state=SEED),
                       t[t.priority == 4].sample(50, random_state=SEED)])
    pilot.to_csv(OUT / "cf2_tasks_pilot.csv", index=False)
    t.to_csv(OUT / "cf2_tasks_full.csv", index=False)
    return {"pilot": len(pilot), "full": len(t), "full_by_priority": t.priority.value_counts().sort_index().to_dict()}


STAGE3_CONTROLS = {"train": 25_000, "validation": 5_000, "test": 5_000}


def write_stage3_reduced() -> dict:
    """Stage 3 limited to what Track B can use. Pair patients are held out of training, and the pairs themselves are
    the evaluation, so their other ECGs are skipped. Controls without a Phase 3 split cannot be placed and are skipped.
    Order: non-pair MI patients' other ECGs, sampled controls, then PTB-XL, so a run stopped early is still useful."""
    t = tasks()
    idx = pd.read_parquet(OUT / "ecg_index.parquet", columns=["study_id", "subject_id"])
    sp = pd.read_parquet(OUT / "cf_splits.parquet", columns=["subject_id", "cf_split"])
    split = idx.merge(sp, on="subject_id", how="left").set_index("study_id").cf_split.astype(object)
    mimic = t[t.dataset == "mimic"].copy()
    mimic["cf_split"] = mimic.record_id.map(split)
    other = mimic[(mimic.priority == 2) & mimic.cf_split.isin(["train", "validation", "test"])]
    ctrl = mimic[mimic.priority == 3]
    ctrl = pd.concat([ctrl[ctrl.cf_split == s].sample(n, random_state=SEED) for s, n in STAGE3_CONTROLS.items()])
    ptb = t[t.dataset == "ptbxl"]
    out = pd.concat([other, ctrl.sort_values("record_id"), ptb], ignore_index=True)[list(t.columns)]
    out.to_csv(OUT / "cf2_tasks_stage3_reduced.csv", index=False)
    return {"stage3_reduced": len(out), "other_mi_non_pair": len(other), "controls": len(ctrl), "ptbxl": len(ptb),
            "skipped_vs_full_stage3": 240_203 - len(out)}


def iter_shards(shard_dir: Path):
    """Yield (meta, beats) per finished chunk, with QC columns added."""
    for csv in sorted(shard_dir.glob("*_features.csv")):
        m = pd.read_csv(csv).copy()
        b = np.fromfile(csv.with_name(csv.name.replace("_features.csv", "_beats.bin")), dtype="<f4")
        b = b.reshape(len(m), 12, N_SAMPLES)
        m["shard"] = f"{shard_dir.name}/{csv.name.split('_features')[0]}"
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            m["max_abs_mv"] = np.nanmax(np.abs(b), axis=(1, 2))
        m["qc_pass"] = ((m.status == "ok") & (m.n_beats_used >= QC["min_beats"])
                        & (m.template_corr >= QC["min_template_corr"]) & m.hr.between(*QC["hr_range"])
                        & (m.max_abs_mv <= QC["max_abs_mv"]))   # > 5 mV in a median beat = artefact
        yield m, b


def read_shards(shard_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    parts = list(iter_shards(shard_dir))
    return pd.concat([m for m, _ in parts], ignore_index=True).copy(), np.concatenate([b for _, b in parts])


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def check(shard_dir: Path) -> dict:
    meta, beats = read_shards(shard_dir)
    L = LEADS.index
    ok = np.flatnonzero(meta.qc_pass.to_numpy())
    ident = {"aVF~(II+III)/2": [], "aVL~(I-III)/2": [], "aVR~-(I+II)/2": [], "aVF~(I-III)/2 (swapped order)": []}
    for i in ok:
        m = beats[i]
        ident["aVF~(II+III)/2"].append(_corr(m[L("aVF")], (m[L("II")] + m[L("III")]) / 2))
        ident["aVL~(I-III)/2"].append(_corr(m[L("aVL")], (m[L("I")] - m[L("III")]) / 2))
        ident["aVR~-(I+II)/2"].append(_corr(m[L("aVR")], -(m[L("I")] + m[L("II")]) / 2))
        ident["aVF~(I-III)/2 (swapped order)"].append(_corr(m[L("aVF")], (m[L("I")] - m[L("III")]) / 2))

    # ST from the median beat at J + 60 ms (J offset from ECGdeli) vs ECGdeli's beat-wise ST_Elev, same ECGs
    st_med = {lead: [] for lead in LEADS}
    st_feat = {lead: [] for lead in LEADS}
    for i in ok:
        j = meta.j_offset_ms.iat[i]
        if not np.isfinite(j):
            continue
        k = min(QRS_ON + int(round((j + 60) / 2)), N_SAMPLES - 1)
        for lead in LEADS:
            st_med[lead].append(beats[i, L(lead), k])
            st_feat[lead].append(meta[f"ST_Elev_{lead}"].iat[i])
    st_r = {lead: round(float(pd.Series(st_med[lead]).corr(pd.Series(st_feat[lead]))), 3) for lead in LEADS}

    # Agreement with the legacy feature file (labels fixed there), for MIMIC ECGs present in both
    cols = ["study_id"] + [f"ST_Elev_{l}" for l in LEADS] + [f"T_Amp_{l}" for l in LEADS]
    legacy = fix_lead_labels(pd.read_csv(PATHS["ecg_features_full"], usecols=lambda c: c in set(cols)))
    legacy = legacy.drop_duplicates("study_id").set_index("study_id")
    mm = meta[(meta.dataset == "mimic") & (meta.status != "error")].set_index("record_id")
    both = mm.join(legacy, how="inner", rsuffix="_legacy")
    legacy_r = {f: round(float(both[f].corr(both[f + "_legacy"])), 3)
                for f in [f"ST_Elev_{l}" for l in LEADS] + [f"T_Amp_{l}" for l in ("I", "aVL", "aVF", "V5")]}

    by = meta.groupby("dataset")
    s = {"ecgs": len(meta), "by_dataset": by.size().to_dict(),
         "status": meta.status.value_counts().to_dict(),
         "qc_pass_rate": {k: round(float(g.qc_pass.mean()), 3) for k, g in by},
         "errors": meta.error.dropna().astype(str).str[:60].value_counts().head(5).to_dict(),
         "beats_used_median": float(meta.n_beats_used.median()),
         "limb_identity_median_r": {k: round(float(np.median(v)), 3) for k, v in ident.items()},
         "limb_identity_share_r_above_0.9": {k: round(float(np.mean(np.array(v) > 0.9)), 3) for k, v in ident.items()},
         "st_median_beat_vs_ecgdeli_r": st_r,
         "vs_legacy_feature_file": {"n": int(len(both)), "pearson_r": legacy_r}}
    name = "cf2_00_pilot.json" if "pilot" in shard_dir.name else f"cf2_02_check_{shard_dir.name.replace('shards_', '')}.json"
    (PHASE_DIRS[10] / name).write_text(json.dumps(s, indent=2))
    return s


def merge(shard_dirs: list[Path]) -> dict:
    """Stream every finished chunk of the given stage folders into one beats.h5 (low memory)."""
    import h5py
    metas, n = [], 0
    with h5py.File(OUT / "beats.h5.tmp", "w") as h:
        ds = h.create_dataset("beats", shape=(0, 12, N_SAMPLES), maxshape=(None, 12, N_SAMPLES), dtype=np.float16,
                              chunks=(256, 12, N_SAMPLES), compression="lzf")
        for d in shard_dirs:
            for m, b in iter_shards(d):
                ds.resize(n + len(m), axis=0)
                ds[n:n + len(m)] = b.astype(np.float16)
                m["row"] = np.arange(n, n + len(m))
                n += len(m)
                metas.append(m)
        h.attrs.update(leads=",".join(LEADS), fs=500, qrs_onset_index=QRS_ON, units="mV",
                       method="ECGdeli legacy chain; median of beats correlating >= 0.90; baseline at QRS onset")
    meta = pd.concat(metas, ignore_index=True).copy()
    dup = meta.duplicated(["dataset", "record_id"]).sum()
    if dup:
        raise ValueError(f"{dup} ECGs appear in more than one shard")
    (OUT / "beats.h5.tmp").replace(OUT / "beats.h5")
    keep = ["row", "record_id", "dataset", "status", "error", "n_beats_detected", "n_beats_used", "template_corr",
            "hr", "j_offset_ms", "t_off_offset_ms", "max_abs_mv", "qc_pass", "shard"]
    meta[keep].to_parquet(OUT / "beats_index.parquet", index=False)
    meta.drop(columns=[c for c in keep if c not in ("record_id", "dataset", "row")]).to_parquet(
        OUT / "ecgdeli_features.parquet", index=False)
    s = {"stages": [d.name for d in shard_dirs], "ecgs": len(meta),
         "by_dataset": meta.dataset.value_counts().to_dict(),
         "status": meta.status.value_counts().to_dict(),
         "qc_pass_rate_by_dataset": meta.groupby("dataset").qc_pass.mean().round(4).to_dict()}
    (PHASE_DIRS[10] / "cf2_01_extract.json").write_text(json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-tasks", action="store_true")
    g.add_argument("--write-stage3-reduced", action="store_true")
    g.add_argument("--check", type=Path)
    g.add_argument("--merge", type=Path, nargs="+", help="one or more stage shard folders")
    a = ap.parse_args()
    out = (write_tasks() if a.write_tasks else write_stage3_reduced() if a.write_stage3_reduced
           else check(a.check) if a.check else merge(a.merge))
    print(json.dumps(out, indent=2, default=str))
