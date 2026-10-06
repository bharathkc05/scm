"""
Extract VAE Latent Embeddings from trained β-VAE checkpoint.

Loads the best model checkpoint, runs the encoder on all ECGs in
train/val/test splits, and saves 64-dim latent mean vectors as CSV.

Output: data/modeling/ecg_embeddings/{train,validation,test}.csv
Each file contains: study_id, SubjectID, z_ecg_1, ..., z_ecg_64

Usage:
    python src/models/extract_embeddings.py --checkpoint models/checkpoints/<run>/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.vae_conv1d import Conv1DVAE
from data.ecg_dataset import (
    load_split_datasets, ecg_collate_fn,
    load_split_datasets_hdf5,
)
from torch.utils.data import DataLoader


def extract_embeddings(
    model: Conv1DVAE,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> pd.DataFrame:
    """
    Run encoder on all ECGs in the dataset and return embeddings DataFrame.

    Returns DataFrame with columns: study_id, SubjectID, z_ecg_1, ..., z_ecg_{z_dim}
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ecg_collate_fn,
        pin_memory=True,
    )

    all_study_ids = []
    all_subject_ids = []
    all_embeddings = []

    with torch.no_grad():
        for signals, metadata in tqdm(loader, desc="Extracting"):
            if signals is None:
                continue

            signals = signals.to(device)
            mu = model.get_latent_embeddings(signals)  # (batch, z_dim)
            embeddings = mu.cpu().numpy()

            for i, meta in enumerate(metadata):
                all_study_ids.append(meta["study_id"])
                all_subject_ids.append(meta["SubjectID"])
                all_embeddings.append(embeddings[i])

    # Build DataFrame
    z_dim = all_embeddings[0].shape[0]
    z_cols = [f"z_ecg_{j+1}" for j in range(z_dim)]

    df = pd.DataFrame(all_embeddings, columns=z_cols)
    df.insert(0, "study_id", all_study_ids)
    df.insert(1, "SubjectID", all_subject_ids)

    return df


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n-> Using device: {device}")

    # Load checkpoint
    print(f"-> Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Initialize model
    z_dim = args.z_dim
    model = Conv1DVAE(z_dim=z_dim, beta=4.0, free_bits=2.0).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"-> Model loaded (z_dim={z_dim}, epoch={checkpoint['epoch']+1})")

    # Load datasets — auto-detect HDF5
    scm_dir = Path(args.scm_dir)
    modeling_dir = scm_dir / "data" / "modeling"
    cohort_csv = scm_dir / "data" / "processed" / "final_ecg_cohorts_troponin.csv"
    base_path = Path(args.base_path)

    h5_path = None
    if args.hdf5_path:
        h5_path = Path(args.hdf5_path)
    else:
        default_h5 = scm_dir / "data" / "processed" / "ecg_signals.h5"
        if default_h5.exists():
            h5_path = default_h5

    if h5_path and h5_path.exists():
        print(f"-> HDF5 MODE: Loading from {h5_path}")
        train_ds, val_ds, test_ds = load_split_datasets_hdf5(
            modeling_dir, h5_path
        )
    else:
        print(f"-> WFDB MODE: Loading from {base_path}")
        train_ds, val_ds, test_ds = load_split_datasets(
            modeling_dir, cohort_csv, base_path, normalize=True
        )

    # Extract embeddings for each split
    output_dir = modeling_dir / "ecg_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, dataset in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
        print(f"\n-> Extracting {split_name} embeddings ({len(dataset):,} ECGs)...")
        df = extract_embeddings(
            model, dataset, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        out_path = output_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"   Saved: {out_path} ({len(df):,} rows, {df.shape[1]} cols)")

        # Report failures
        n_failed = len(dataset) - len(df)
        if n_failed > 0:
            print(f"   WARNING: {n_failed} ECGs failed to load ({n_failed/len(dataset)*100:.1f}%)")

    # Summary
    print("\n" + "=" * 60)
    print("Embedding extraction complete!")
    print(f"Output directory: {output_dir}")
    for f in sorted(output_dir.glob("*.csv")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract VAE latent embeddings")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--scm_dir", type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent),
                        help="Path to SCM project root")
    parser.add_argument("--base_path", type=str,
                        default=r"C:\Users\Acer\Desktop\Capstone\Capstone\data\raw",
                        help="Root path to raw data")
    parser.add_argument("--hdf5_path", type=str, default=None,
                        help="Path to ecg_signals.h5 (auto-detected)")
    parser.add_argument("--z_dim", type=int, default=64,
                        help="Latent dimension size (must match checkpoint)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers")

    args = parser.parse_args()
    main(args)
