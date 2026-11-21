"""Phase 3, step 2 — 64 β-VAE latents for every selected (earliest feature) ECG.

Stored embeddings are used where they exist (provenance verified in Phase 2). ECGs missing from the
HDF5 file (mostly post-onset ECGs the legacy master dropped) are encoded from raw WFDB with the
project's own loader (`ECGDataset._load_ecg`: 500 Hz, 5000 samples, NaN->0, per-lead z-score) and the
same checkpoint. Equivalence is checked first by WFDB-encoding ECGs that already have stored values.

Output (git-ignored): data/roadmap/phase3/latents_selected.parquet
Summary:             roadmap/phase3_dataset/p3_02_latents.json
"""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from roadmap.config import DATA_OUT, PATHS, PHASE_DIRS, RAW, REPO, SEED

sys.path.insert(0, str(REPO / "src"))
from data.ecg_dataset import ECGDataset  # noqa: E402
from models.vae_conv1d import Conv1DVAE  # noqa: E402

OUT = DATA_OUT / "phase3"
Z = [f"z_ecg_{i}" for i in range(1, 65)]
LOADER = SimpleNamespace(target_length=5000, normalize=True)  # attributes _load_ecg reads
BATCH = 64


def encode(model, device, paths: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(paths), BATCH):
        sig = torch.stack([ECGDataset._load_ecg(LOADER, p) for p in paths[i:i + BATCH]]).to(device)
        with torch.no_grad():
            out.append(model.get_latent_embeddings(sig).cpu().numpy())
    return np.concatenate(out)


def main() -> dict:
    ae = pd.read_parquet(DATA_OUT / "phase2/admission_ecg.parquet", columns=["first_feat_study_id"]).dropna()
    want = set(ae["first_feat_study_id"].astype("int64"))
    stored = pd.concat([pd.read_csv(PATHS["ecg_embeddings_dir"] / f"{s}.csv", usecols=["study_id"] + Z)
                        for s in ("train", "validation", "test")]).drop_duplicates("study_id")
    have = stored[stored["study_id"].isin(want)].assign(latent_source="stored")
    missing = sorted(want - set(have["study_id"]))

    rl = pd.read_csv(PATHS["raw_ecg_record_list"], usecols=["study_id", "path"]).set_index("study_id")["path"]
    to_path = lambda sid: str(RAW / "MIMIC-IV-ECG-1.0" / rl.loc[int(sid)])  # noqa: E731

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(PATHS["vae_best_model"], map_location=device, weights_only=False)
    model = Conv1DVAE(z_dim=64, beta=4.0, free_bits=2.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Equivalence check: WFDB path vs stored values on 128 ECGs that have both.
    chk = have.sample(128, random_state=SEED)
    chk_mu = encode(model, device, [to_path(s) for s in chk["study_id"]])
    diff = np.abs(chk_mu - chk[Z].to_numpy())

    new_mu = encode(model, device, [to_path(s) for s in missing])
    new = pd.DataFrame(new_mu, columns=Z).assign(study_id=missing, latent_source="encoded_wfdb")
    lat = pd.concat([have, new], ignore_index=True)
    lat.to_parquet(OUT / "latents_selected.parquet", index=False)
    return {"selected_ecgs": len(want), "stored": len(have), "encoded_from_wfdb": len(new),
            "wfdb_vs_stored_check": {"n": len(chk), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())},
            "checkpoint_epoch": int(ckpt["epoch"]) + 1, "all_selected_covered": len(lat) == len(want)}


if __name__ == "__main__":
    s = main()
    (PHASE_DIRS[3] / "p3_02_latents.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
