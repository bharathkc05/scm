"""
Convert WFDB ECG files to a single HDF5 file for fast VAE training.

Reads all modeling ECGs (MI_Post-Incident excluded), loads each WFDB
waveform, normalizes per-lead (zero-mean, unit-variance), and writes
them into a contiguous HDF5 dataset.

Output: data/processed/ecg_signals.h5
  - signals:    (N, 12, 5000) float32
  - study_ids:  (N,) int64

Usage:
    python src/data/convert_to_hdf5.py

    # Or with custom paths:
    python src/data/convert_to_hdf5.py \
        --cohort data/processed/final_ecg_cohorts_troponin.csv \
        --base_path C:/Users/Acer/Desktop/Capstone/Capstone/data/raw \
        --output data/processed/ecg_signals.h5
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import wfdb
from tqdm import tqdm


# ============================================================================
# CONSTANTS
# ============================================================================

EXPECTED_LEADS = 12
EXPECTED_LENGTH = 5000  # 10s at 500 Hz
TARGET_FS = 500


# ============================================================================
# CORE FUNCTIONS
# ============================================================================


def load_and_normalize_ecg(record_path: str) -> np.ndarray | None:
    """
    Load a single ECG from WFDB files, normalize per-lead.

    Returns
    -------
    signal : np.ndarray of shape (12, 5000), float32
        Normalized ECG signal. None if loading fails.
    """
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # (n_samples, n_leads)

        if signal is None:
            return None

        signal = signal.T  # (n_leads, n_samples)
        n_leads, n_samples = signal.shape

        if n_leads != EXPECTED_LEADS:
            return None

        # Resample if needed
        if record.fs != TARGET_FS:
            from scipy.signal import resample
            target_samples = int(n_samples * TARGET_FS / record.fs)
            signal = resample(signal, target_samples, axis=1)
            n_samples = target_samples

        # Pad or truncate
        if n_samples < EXPECTED_LENGTH:
            pad_width = EXPECTED_LENGTH - n_samples
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
        elif n_samples > EXPECTED_LENGTH:
            signal = signal[:, :EXPECTED_LENGTH]

        # Replace NaN/Inf
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize per lead (zero-mean, unit-variance)
        for lead in range(EXPECTED_LEADS):
            std = signal[lead].std()
            mean = signal[lead].mean()
            if std > 1e-8:
                signal[lead] = (signal[lead] - mean) / std
            else:
                signal[lead] = signal[lead] - mean

        return signal.astype(np.float32)

    except Exception:
        return None


def convert_to_hdf5(
    cohort_csv: str,
    base_path: str,
    output_path: str,
    chunk_size: int = 256,
):
    """
    Convert all modeling ECGs to a single HDF5 file.

    Parameters
    ----------
    cohort_csv : str
        Path to final_ecg_cohorts_troponin.csv.
    base_path : str
        Root path to raw data (parent of MIMIC-IV-ECG-1.0/).
    output_path : str
        Output HDF5 file path.
    chunk_size : int
        HDF5 chunk size along the first axis (for I/O optimization).
    """
    base_path = Path(base_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cohort and exclude MI_Post-Incident
    print("Loading cohort CSV...")
    cohort = pd.read_csv(cohort_csv)
    cohort = cohort[cohort["Label_Troponin"] != "MI_Post-Incident"]
    cohort = cohort.drop_duplicates(subset="study_id").reset_index(drop=True)

    total = len(cohort)
    print(f"Total modeling ECGs to convert: {total:,}")
    print(f"Output file: {output_path}")
    print(f"Estimated file size: ~{total * 12 * 5000 * 4 / 1e9:.1f} GB")

    # Create HDF5 file
    # Use maxshape=(None,...) so we can resize if some ECGs fail
    with h5py.File(str(output_path), "w") as h5f:
        # Pre-allocate datasets at max size
        signals_ds = h5f.create_dataset(
            "signals",
            shape=(total, EXPECTED_LEADS, EXPECTED_LENGTH),
            dtype=np.float32,
            chunks=(min(chunk_size, total), EXPECTED_LEADS, EXPECTED_LENGTH),
            maxshape=(total, EXPECTED_LEADS, EXPECTED_LENGTH),
            compression=None,  # No compression for speed
        )

        study_ids_ds = h5f.create_dataset(
            "study_ids",
            shape=(total,),
            dtype=np.int64,
            maxshape=(total,),
        )

        # Store metadata as attributes
        h5f.attrs["n_leads"] = EXPECTED_LEADS
        h5f.attrs["signal_length"] = EXPECTED_LENGTH
        h5f.attrs["sampling_rate"] = TARGET_FS
        h5f.attrs["normalized"] = True
        h5f.attrs["normalization"] = "per-lead z-score (zero-mean, unit-variance)"

        # Convert
        write_idx = 0
        n_failed = 0
        failed_ids = []

        start_time = time.time()
        pbar = tqdm(range(total), desc="Converting", unit="ECG")

        for i in pbar:
            row = cohort.iloc[i]
            study_id = int(row["study_id"])
            waveform_path = row["waveform_path"]
            full_path = str(base_path / waveform_path)

            signal = load_and_normalize_ecg(full_path)

            if signal is not None:
                signals_ds[write_idx] = signal
                study_ids_ds[write_idx] = study_id
                write_idx += 1
            else:
                n_failed += 1
                if len(failed_ids) < 100:  # Don't store too many
                    failed_ids.append(study_id)

            # Progress update
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate
                pbar.set_postfix({
                    "saved": f"{write_idx:,}",
                    "failed": n_failed,
                    "rate": f"{rate:.0f}/s",
                    "ETA": f"{eta/60:.0f}m",
                })

        # Resize to actual number of valid ECGs
        if write_idx < total:
            signals_ds.resize(write_idx, axis=0)
            study_ids_ds.resize(write_idx, axis=0)

        # Store final counts
        h5f.attrs["n_valid"] = write_idx
        h5f.attrs["n_failed"] = n_failed

    # Summary
    elapsed = time.time() - start_time
    file_size_gb = output_path.stat().st_size / 1e9

    print("\n" + "=" * 70)
    print("HDF5 Conversion Complete!")
    print("=" * 70)
    print(f"  Output:      {output_path}")
    print(f"  File size:   {file_size_gb:.2f} GB")
    print(f"  Valid ECGs:  {write_idx:,} / {total:,}")
    print(f"  Failed:      {n_failed:,} ({n_failed/total*100:.2f}%)")
    print(f"  Time:        {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"  Rate:        {write_idx/elapsed:.0f} ECGs/second")
    print("=" * 70)

    if failed_ids:
        print(f"\n  First {min(len(failed_ids), 10)} failed study_ids:")
        for sid in failed_ids[:10]:
            print(f"    {sid}")

    return write_idx, n_failed


# ============================================================================
# VERIFICATION
# ============================================================================


def verify_hdf5(h5_path: str, cohort_csv: str, base_path: str, n_samples: int = 10):
    """Verify HDF5 content matches WFDB source for a random sample."""
    print(f"\nVerifying {n_samples} random samples against WFDB source...")

    h5f = h5py.File(h5_path, "r")
    n_total = h5f["signals"].shape[0]
    study_ids = h5f["study_ids"][:]

    # Pick random indices
    rng = np.random.RandomState(42)
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)

    cohort = pd.read_csv(cohort_csv)
    path_map = dict(zip(cohort["study_id"], cohort["waveform_path"]))

    n_match = 0
    for idx in indices:
        sid = study_ids[idx]
        h5_signal = h5f["signals"][idx]

        waveform_path = path_map.get(sid)
        if waveform_path is None:
            print(f"  [{idx}] study_id={sid}: NOT FOUND in cohort!")
            continue

        full_path = str(Path(base_path) / waveform_path)
        wfdb_signal = load_and_normalize_ecg(full_path)

        if wfdb_signal is None:
            print(f"  [{idx}] study_id={sid}: WFDB load failed")
            continue

        max_diff = np.abs(h5_signal - wfdb_signal).max()
        match = max_diff < 1e-5
        n_match += int(match)
        status = "MATCH" if match else f"MISMATCH (max_diff={max_diff:.6f})"
        print(f"  [{idx}] study_id={sid}: {status}")

    h5f.close()
    print(f"\n  Result: {n_match}/{len(indices)} samples verified")
    return n_match == len(indices)


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WFDB ECGs to HDF5")

    scm_dir = Path(__file__).resolve().parent.parent.parent

    parser.add_argument("--cohort", type=str,
                        default=str(scm_dir / "data" / "processed" / "final_ecg_cohorts_troponin.csv"),
                        help="Path to cohort CSV")
    parser.add_argument("--base_path", type=str,
                        default=r"C:\Users\Acer\Desktop\Capstone\Capstone\data\raw",
                        help="Root path to raw data")
    parser.add_argument("--output", type=str,
                        default=str(scm_dir / "data" / "processed" / "ecg_signals.h5"),
                        help="Output HDF5 file path")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification after conversion")

    args = parser.parse_args()

    n_valid, n_failed = convert_to_hdf5(
        cohort_csv=args.cohort,
        base_path=args.base_path,
        output_path=args.output,
    )

    if args.verify:
        verify_hdf5(args.output, args.cohort, args.base_path, n_samples=20)
