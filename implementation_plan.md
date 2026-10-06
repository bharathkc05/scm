# Implementation Plan: HDF5 ECG Dataset Conversion

## 1. Goal
Speed up VAE training by 10-20x by converting the 375,985 individual WFDB `.dat` and `.hea` files into a single, contiguous, memory-mapped HDF5 file (`ecg_signals.h5`). This eliminates random file I/O overhead.

## 2. Ensuring Proper Mapping
The most critical requirement is ensuring that the pre-built `train.csv`, `validation.csv`, and `test.csv` splits map exactly to the correct signals in the HDF5 file.

**Mechanism:**
1. **Conversion Phase:** The converter reads `data/processed/final_ecg_cohorts_troponin.csv`, which defines all 377,113 modeling ECGs. It iterates through them, saving each valid signal to `ecg_signals.h5` at index `i`, and simultaneously saving the corresponding `study_id` to an array at index `i`.
2. **Loading Phase:** The `ECGDataset` reads the `study_id` array from the HDF5 file and builds a fast hash map: `{study_id: hdf5_index}`. When the PyTorch DataLoader requests an ECG for a `study_id` (from `train.csv`), it instantly looks up the `hdf5_index` and reads directly from that row in the HDF5 signal array.

## 3. Proposed Changes

### [NEW] `src/data/convert_to_hdf5.py`
A robust conversion script that:
- Reads `data/processed/final_ecg_cohorts_troponin.csv` (excluding `MI_Post-Incident`).
- Uses `h5py` to create a new `ecg_signals.h5` file.
- Allocates large datasets: `signals` `(N, 12, 5000)` and `study_ids` `(N,)`.
- Uses multiprocessing or a fast loop to load WFDB files, normalize them (to save CPU during training), and write them sequentially into the HDF5 file.
- Handles corrupt files and tracks which `study_ids` were successfully saved.

### [MODIFY] `src/data/ecg_dataset.py`
Refactor the dataset to support HDF5:
- Rename `ECGDataset` to `ECGDatasetWFDB` (keep it around for reference).
- Create a new `ECGDatasetHDF5`.
- `__init__` opens the HDF5 file in read-only mode (`'r'`), loads the `study_ids` array, and creates the `study_id -> h5_index` mapping.
- `__getitem__` uses the mapping to perform a fast `self.h5_signals[h5_index]` read.

### [MODIFY] `src/models/train_vae.py` & `extract_embeddings.py`
- Update the data loading calls to use `ECGDatasetHDF5` instead of the WFDB version.
- Add an `--hdf5_path` argument to point to the new `ecg_signals.h5` file.

## 4. Open Questions & User Review
> [!IMPORTANT]
> The HDF5 file will be roughly **85 GB** in size (377,000 ECGs × 12 leads × 5000 samples × 4 bytes/float32). Do you have at least 100 GB of free space on your `C:\` drive (or whichever drive `Capstone` is on) to store this file?

> [!NOTE]
> I recommend creating the HDF5 file in `C:\Users\Acer\Desktop\Capstone\SCM\data\processed\ecg_signals.h5`. Is this location acceptable?

## 5. Verification Plan
1. Create a tiny test HDF5 file with 1,000 ECGs to verify the conversion script logic.
2. Verify that `ECGDatasetHDF5` can load from the test HDF5 and that the signals match the WFDB output exactly for the same `study_id`.
3. Run the VAE smoke test using the HDF5 dataset to confirm integration.
4. Execute the full conversion script (which will take ~30-60 minutes).
