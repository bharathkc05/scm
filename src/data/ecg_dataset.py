"""
ECG Dataset — PyTorch Dataset for loading 12-lead WFDB waveforms.

Loads raw ECG signals from MIMIC-IV-ECG WFDB files (.dat/.hea), keyed
by study_id via the cohort labels file.

Expected waveform directory layout:
    base_path / MIMIC-IV-ECG-1.0 / files / pXXXX / pXXXXXXXX / sYYYYYYYY / YYYYYYYY.{dat,hea}

Usage:
    from data.ecg_dataset import ECGDataset, create_dataloaders

    dataset = ECGDataset(labels_csv, cohort_csv, base_path)
    loader  = DataLoader(dataset, batch_size=64, collate_fn=ecg_collate_fn)
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wfdb
except ImportError:
    raise ImportError("wfdb is required: pip install wfdb")

try:
    import h5py
except ImportError:
    h5py = None  # HDF5 features will be unavailable


# ============================================================================
# CONSTANTS
# ============================================================================

EXPECTED_LEADS = 12
EXPECTED_LENGTH = 5000  # 10 seconds at 500 Hz
TARGET_FS = 500


# ============================================================================
# DATASET
# ============================================================================


class ECGDataset(Dataset):
    """
    PyTorch Dataset for 12-lead ECG waveforms from MIMIC-IV-ECG.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Must contain at least: study_id, SubjectID, Label_Troponin.
        Typically loaded from data/modeling/labels/{split}.csv.
    cohort_df : pd.DataFrame
        Must contain: study_id, waveform_path.
        Loaded from data/processed/final_ecg_cohorts_troponin.csv.
    base_path : str or Path
        Root path to raw data (parent of MIMIC-IV-ECG-1.0/).
    normalize : bool
        If True, z-score normalize each lead independently.
    target_length : int
        Expected signal length. Signals are padded/truncated to this length.
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        cohort_df: pd.DataFrame,
        base_path: str | Path,
        normalize: bool = True,
        target_length: int = EXPECTED_LENGTH,
    ):
        self.base_path = Path(base_path)
        self.normalize = normalize
        self.target_length = target_length

        # Join labels with waveform paths
        cohort_subset = cohort_df[["study_id", "waveform_path"]].drop_duplicates(subset="study_id")
        self.df = labels_df.merge(cohort_subset, on="study_id", how="inner")

        if len(self.df) < len(labels_df):
            n_missing = len(labels_df) - len(self.df)
            warnings.warn(
                f"{n_missing} study_ids from labels have no waveform_path. "
                f"Keeping {len(self.df)}/{len(labels_df)} rows."
            )

        self.df = self.df.reset_index(drop=True)

        # Pre-resolve full paths for fast lookup
        self.paths = []
        for _, row in self.df.iterrows():
            full_path = self.base_path / row["waveform_path"]
            self.paths.append(str(full_path))

        # Track loading failures for diagnostics
        self._load_failures = 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        Returns
        -------
        signal : torch.Tensor or None
            Shape (12, target_length), float32. None if file is corrupt/missing.
        metadata : dict
            Contains study_id, SubjectID, Label_Troponin, waveform_path.
        """
        row = self.df.iloc[idx]
        record_path = self.paths[idx]

        metadata = {
            "study_id": int(row["study_id"]),
            "SubjectID": int(row["SubjectID"]),
            "Label_Troponin": str(row["Label_Troponin"]),
            "waveform_path": row["waveform_path"],
        }

        try:
            signal = self._load_ecg(record_path)
            return signal, metadata
        except Exception as e:
            self._load_failures += 1
            if self._load_failures <= 10:
                warnings.warn(f"Failed to load ECG at {record_path}: {e}")
            return None, metadata

    def _load_ecg(self, record_path: str) -> torch.Tensor:
        """Load a single ECG from WFDB files and return as tensor."""
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # (n_samples, n_leads)

        if signal is None:
            raise ValueError("wfdb.rdrecord returned None signal")

        # Transpose to (leads, samples)
        signal = signal.T  # (n_leads, n_samples)

        # Handle lead count mismatch
        n_leads, n_samples = signal.shape
        if n_leads != EXPECTED_LEADS:
            raise ValueError(f"Expected {EXPECTED_LEADS} leads, got {n_leads}")

        # Resample if not at target frequency
        if record.fs != TARGET_FS:
            from scipy.signal import resample
            target_samples = int(n_samples * TARGET_FS / record.fs)
            signal = resample(signal, target_samples, axis=1)
            n_samples = target_samples

        # Pad or truncate to target length
        if n_samples < self.target_length:
            pad_width = self.target_length - n_samples
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
        elif n_samples > self.target_length:
            signal = signal[:, : self.target_length]

        # Replace NaN/Inf with 0
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize per lead (zero-mean, unit-variance)
        if self.normalize:
            for lead in range(EXPECTED_LEADS):
                lead_std = signal[lead].std()
                lead_mean = signal[lead].mean()
                if lead_std > 1e-8:
                    signal[lead] = (signal[lead] - lead_mean) / lead_std
                else:
                    signal[lead] = signal[lead] - lead_mean

        return torch.tensor(signal, dtype=torch.float32)

    @property
    def label_distribution(self) -> dict:
        """Return label counts."""
        return self.df["Label_Troponin"].value_counts().to_dict()

    @property
    def load_failure_count(self) -> int:
        return self._load_failures


# ============================================================================
# COLLATE FUNCTION
# ============================================================================


def ecg_collate_fn(batch: list) -> Tuple[Optional[torch.Tensor], list]:
    """
    Custom collate that filters out None (corrupt) ECGs.

    Returns
    -------
    signals : torch.Tensor or None
        Shape (batch_size_valid, 12, 5000). None if entire batch is corrupt.
    metadata_list : list of dicts
    """
    valid = [(signal, meta) for signal, meta in batch if signal is not None]

    if len(valid) == 0:
        return None, [meta for _, meta in batch]

    signals = torch.stack([s for s, _ in valid])
    metadata = [m for _, m in valid]

    return signals, metadata


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def load_split_datasets(
    modeling_dir: str | Path,
    cohort_csv: str | Path,
    base_path: str | Path,
    normalize: bool = True,
) -> Tuple[ECGDataset, ECGDataset, ECGDataset]:
    """
    Load train/validation/test datasets from the modeling directory.

    Parameters
    ----------
    modeling_dir : Path
        Path to data/modeling/ (contains labels/ subdirectory).
    cohort_csv : Path
        Path to data/processed/final_ecg_cohorts_troponin.csv.
    base_path : Path
        Root path to raw ECG data.
    normalize : bool
        Whether to z-score normalize each lead.

    Returns
    -------
    train_dataset, val_dataset, test_dataset : ECGDataset
    """
    modeling_dir = Path(modeling_dir)
    cohort_df = pd.read_csv(cohort_csv)

    datasets = {}
    for split in ["train", "validation", "test"]:
        labels_df = pd.read_csv(modeling_dir / "labels" / f"{split}.csv")
        datasets[split] = ECGDataset(
            labels_df=labels_df,
            cohort_df=cohort_df,
            base_path=base_path,
            normalize=normalize,
        )

    return datasets["train"], datasets["validation"], datasets["test"]


def create_dataloaders(
    train_dataset: ECGDataset,
    val_dataset: ECGDataset,
    test_dataset: ECGDataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test with proper settings."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ecg_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ecg_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ecg_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ============================================================================
# HDF5 DATASET (FAST)
# ============================================================================


class ECGDatasetHDF5(Dataset):
    """
    Fast PyTorch Dataset that reads pre-converted ECGs from an HDF5 file.

    The HDF5 file contains:
      - signals:   (N, 12, 5000) float32  — all normalized ECG signals
      - study_ids: (N,) int64             — parallel array of study_ids

    This dataset maps each study_id from a split labels CSV to its
    index in the HDF5 file via a hash map for O(1) lookup.

    Parameters
    ----------
    h5_path : str or Path
        Path to ecg_signals.h5.
    labels_df : pd.DataFrame
        Must contain at least: study_id, SubjectID, Label_Troponin.
    """

    def __init__(self, h5_path: str | Path, labels_df: pd.DataFrame):
        self.h5_path = str(h5_path)
        self.labels_df = labels_df.reset_index(drop=True)

        # Lazy-open pattern: don't keep h5py handle in __init__
        # (h5py objects can't be pickled for multiprocessing workers)
        self._h5f = None
        self._signals = None

        # Temporarily open to build the study_id mapping, then close
        with h5py.File(self.h5_path, "r") as h5f:
            h5_study_ids = h5f["study_ids"][:]  # Load full array to RAM (~3 MB)

        # Build study_id → h5_index hash map
        self._sid_to_h5idx = {}
        for h5_idx, sid in enumerate(h5_study_ids):
            self._sid_to_h5idx[int(sid)] = h5_idx

        # Resolve which label rows have a valid HDF5 entry
        self._valid_indices = []
        self._h5_indices = []
        n_missing = 0

        for row_idx in range(len(self.labels_df)):
            sid = int(self.labels_df.iloc[row_idx]["study_id"])
            h5_idx = self._sid_to_h5idx.get(sid)
            if h5_idx is not None:
                self._valid_indices.append(row_idx)
                self._h5_indices.append(h5_idx)
            else:
                n_missing += 1

        if n_missing > 0:
            warnings.warn(
                f"{n_missing} study_ids from labels not found in HDF5. "
                f"Using {len(self._valid_indices)}/{len(self.labels_df)} rows."
            )

    def _ensure_h5_open(self):
        """Lazily open the HDF5 file (safe for multiprocessing workers)."""
        if self._h5f is None:
            self._h5f = h5py.File(self.h5_path, "r")
            self._signals = self._h5f["signals"]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns
        -------
        signal : torch.Tensor
            Shape (12, 5000), float32. Always valid (pre-validated in HDF5).
        metadata : dict
            Contains study_id, SubjectID, Label_Troponin.
        """
        self._ensure_h5_open()

        row_idx = self._valid_indices[idx]
        h5_idx = self._h5_indices[idx]
        row = self.labels_df.iloc[row_idx]

        signal = torch.tensor(self._signals[h5_idx], dtype=torch.float32)

        metadata = {
            "study_id": int(row["study_id"]),
            "SubjectID": int(row["SubjectID"]),
            "Label_Troponin": str(row["Label_Troponin"]),
        }

        return signal, metadata

    @property
    def label_distribution(self) -> dict:
        """Return label counts for the valid subset."""
        valid_labels = self.labels_df.iloc[self._valid_indices]["Label_Troponin"]
        return valid_labels.value_counts().to_dict()

    def close(self):
        """Close the HDF5 file handle."""
        if getattr(self, '_h5f', None) is not None:
            self._h5f.close()
            self._h5f = None
            self._signals = None

    def __del__(self):
        self.close()


def load_split_datasets_hdf5(
    modeling_dir: str | Path,
    h5_path: str | Path,
) -> Tuple[ECGDatasetHDF5, ECGDatasetHDF5, ECGDatasetHDF5]:
    """
    Load train/validation/test HDF5 datasets from the modeling directory.

    Parameters
    ----------
    modeling_dir : Path
        Path to data/modeling/ (contains labels/ subdirectory).
    h5_path : Path
        Path to ecg_signals.h5.

    Returns
    -------
    train_dataset, val_dataset, test_dataset : ECGDatasetHDF5
    """
    modeling_dir = Path(modeling_dir)
    datasets = {}
    for split in ["train", "validation", "test"]:
        labels_df = pd.read_csv(modeling_dir / "labels" / f"{split}.csv")
        datasets[split] = ECGDatasetHDF5(h5_path=h5_path, labels_df=labels_df)

    return datasets["train"], datasets["validation"], datasets["test"]


def create_dataloaders_hdf5(
    train_dataset: ECGDatasetHDF5,
    val_dataset: ECGDatasetHDF5,
    test_dataset: ECGDatasetHDF5,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for HDF5 datasets (no custom collate needed)."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("ECG Dataset Self-Test")
    print("=" * 70)

    # Paths
    scm_dir = Path(__file__).resolve().parent.parent.parent
    modeling_dir = scm_dir / "data" / "modeling"
    cohort_csv = scm_dir / "data" / "processed" / "final_ecg_cohorts_troponin.csv"
    base_path = Path(r"C:\Users\Acer\Desktop\Capstone\Capstone\data\raw")

    print(f"\n  SCM dir:      {scm_dir}")
    print(f"  Modeling dir: {modeling_dir}")
    print(f"  Cohort CSV:   {cohort_csv}")
    print(f"  Base path:    {base_path}")

    # Check files exist
    for p in [modeling_dir / "labels" / "train.csv", cohort_csv]:
        if not p.exists():
            print(f"\n  ERROR: {p} not found!")
            sys.exit(1)

    # Load datasets
    print("\n  Loading datasets...")
    train_ds, val_ds, test_ds = load_split_datasets(
        modeling_dir, cohort_csv, base_path, normalize=True
    )
    print(f"  Train: {len(train_ds):,} ECGs")
    print(f"  Val:   {len(val_ds):,} ECGs")
    print(f"  Test:  {len(test_ds):,} ECGs")

    # Test loading a few samples
    print("\n  Loading 10 sample ECGs...")
    n_success = 0
    n_fail = 0
    for i in range(min(10, len(train_ds))):
        signal, meta = train_ds[i]
        if signal is not None:
            n_success += 1
            if i < 3:
                print(f"    [{i}] shape={tuple(signal.shape)}, "
                      f"mean={signal.mean():.4f}, std={signal.std():.4f}, "
                      f"label={meta['Label_Troponin']}")
        else:
            n_fail += 1
            print(f"    [{i}] FAILED: {meta['waveform_path']}")

    print(f"\n  Results: {n_success}/10 loaded, {n_fail} failed")

    # Test DataLoader
    print("\n  Testing DataLoader (batch_size=4)...")
    loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=ecg_collate_fn)
    batch_signals, batch_meta = next(iter(loader))
    if batch_signals is not None:
        print(f"    Batch shape: {tuple(batch_signals.shape)}")
        print(f"    Batch dtype: {batch_signals.dtype}")
        print(f"    Labels: {[m['Label_Troponin'] for m in batch_meta]}")
        print("\n  ALL TESTS PASSED!")
    else:
        print("    ERROR: Entire batch returned None")
        sys.exit(1)

    print("=" * 70)
