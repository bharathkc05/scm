"""Phase 2, step 5 — β-VAE provenance check (decides whether the 64 latents may be used).

Checks: (1) checkpoint identity and training config vs split files; (2) training data composition;
(3) stored embeddings reproduce when re-encoding signals from ecg_signals.h5 with best_model.pt;
(4) HDF5 signals match the raw WFDB waveforms after the documented per-lead z-score.

Summary: roadmap/phase2_data_audit/p2_05_vae_provenance.json
"""
import json
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import wfdb

from roadmap.config import PATHS, PHASE_DIRS, RAW, REPO, SEED

sys.path.insert(0, str(REPO / "src"))
from models.vae_conv1d import Conv1DVAE  # noqa: E402

N_PER_SPLIT = 256
N_WFDB = 12


def main() -> dict:
    rng = np.random.default_rng(SEED)
    run = PATHS["vae_run_dir"]
    cfg = json.loads((run / "config.json").read_text())
    labels = {s: pd.read_csv(PATHS["vae_labels_dir"] / f"{s}.csv") for s in ("train", "validation", "test")}
    emb = {s: pd.read_csv(PATHS["ecg_embeddings_dir"] / f"{s}.csv") for s in ("train", "validation", "test")}

    s = {"run_dir": run.name, "config": {k: cfg[k] for k in ("z_dim", "beta", "seed", "epochs", "num_train", "num_val", "num_test")},
         "split_rows": {k: len(v) for k, v in labels.items()},
         "embedding_rows": {k: len(v) for k, v in emb.items()},
         "config_counts_match_split_files": cfg["num_train"] == len(labels["train"]) and cfg["num_val"] == len(labels["validation"])
                                            and cfg["num_test"] == len(labels["test"]),
         "embedding_ids_match_split_ids": {k: bool(set(emb[k]["study_id"]) == set(labels[k]["study_id"])) for k in labels},
         "train_label_composition": labels["train"]["Label_Troponin"].value_counts().to_dict(),
         "training_loss_uses_labels": False}  # verified by code review of train_vae.py / vae_conv1d.py

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(PATHS["vae_best_model"], map_location=device, weights_only=False)
    model = Conv1DVAE(z_dim=cfg["z_dim"], beta=cfg["beta"], free_bits=2.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    s["best_checkpoint_epoch"] = int(ckpt["epoch"]) + 1

    zcols = [f"z_ecg_{i}" for i in range(1, cfg["z_dim"] + 1)]
    with h5py.File(PATHS["ecg_signals_h5"], "r") as h5:
        h5_ids = h5["study_ids"][:]
        idx_of = {int(v): i for i, v in enumerate(h5_ids)}
        repro = {}
        for split, df in emb.items():
            sample = df.sample(N_PER_SPLIT, random_state=int(rng.integers(1e9)))
            idx = np.array([idx_of[int(v)] for v in sample["study_id"]])
            order = np.argsort(idx)
            sig = h5["signals"][np.sort(idx)]                       # h5py needs increasing indices
            with torch.no_grad():
                mu = model.get_latent_embeddings(torch.from_numpy(sig).to(device)).cpu().numpy()
            stored = sample[zcols].to_numpy()[order]
            diff = np.abs(mu - stored)
            repro[split] = {"n": N_PER_SPLIT, "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
                            "min_dim_correlation": float(min(np.corrcoef(mu[:, j], stored[:, j])[0, 1] for j in range(len(zcols))))}
        s["reencode_vs_stored"] = repro

        # HDF5 vs raw WFDB (documented preprocessing: 500 Hz, 5000 samples, NaN->0, per-lead z-score)
        rl = pd.read_csv(PATHS["raw_ecg_record_list"], usecols=["study_id", "path"]).set_index("study_id")
        pick = rng.choice(h5_ids, N_WFDB, replace=False)
        wf = []
        for sid in pick:
            rec = wfdb.rdrecord(str(RAW / "MIMIC-IV-ECG-1.0" / rl.loc[int(sid), "path"]))
            x = np.nan_to_num(rec.p_signal.T[:, :5000].astype(np.float64))
            m, sd = x.mean(axis=1, keepdims=True), x.std(axis=1, keepdims=True)
            x = np.where(sd > 1e-8, (x - m) / np.where(sd > 1e-8, sd, 1), x - m)
            wf.append(float(np.abs(x - h5["signals"][idx_of[int(sid)]]).max()))
        s["hdf5_vs_wfdb_max_abs_diff"] = {"n": N_WFDB, "worst": max(wf), "median": float(np.median(wf))}
    s["h5_rows"] = int(len(h5_ids))
    return s


if __name__ == "__main__":
    s = main()
    (PHASE_DIRS[2] / "p2_05_vae_provenance.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))
