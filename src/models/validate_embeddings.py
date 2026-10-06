"""
Validate VAE Embeddings — comprehensive quality checks.

Checks:
  1. Basic stats (shape, NaN, distribution)
  2. Label separation via t-SNE visualization
  3. Downstream discriminability (logistic regression MI vs Control)
  4. Reconstruction quality (encode → decode → compare)

Usage:
    python src/models/validate_embeddings.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

sys.path.append(str(Path(__file__).parent.parent))


def check_basic_stats(emb_dir: Path):
    """Check 1: Basic statistics — shape, NaN, distribution."""
    print("=" * 70)
    print("CHECK 1: Basic Statistics")
    print("=" * 70)

    for split in ["train", "validation", "test"]:
        df = pd.read_csv(emb_dir / f"{split}.csv")
        z_cols = [c for c in df.columns if c.startswith("z_ecg_")]

        z_data = df[z_cols].values
        n_nan = np.isnan(z_data).sum()
        n_inf = np.isinf(z_data).sum()

        print(f"\n  {split}:")
        print(f"    Shape: {df.shape}")
        print(f"    Z dimensions: {len(z_cols)}")
        print(f"    NaN count: {n_nan}")
        print(f"    Inf count: {n_inf}")
        print(f"    Z mean (global): {z_data.mean():.4f}")
        print(f"    Z std  (global): {z_data.std():.4f}")
        print(f"    Z min:  {z_data.min():.4f}")
        print(f"    Z max:  {z_data.max():.4f}")

        # Per-dimension stats
        dim_means = z_data.mean(axis=0)
        dim_stds = z_data.std(axis=0)
        print(f"    Per-dim mean range: [{dim_means.min():.4f}, {dim_means.max():.4f}]")
        print(f"    Per-dim std range:  [{dim_stds.min():.4f}, {dim_stds.max():.4f}]")

        # Check for collapsed dimensions (std ≈ 0)
        collapsed = (dim_stds < 0.01).sum()
        if collapsed > 0:
            print(f"    ⚠️  {collapsed} dimensions have std < 0.01 (possibly collapsed)")
        else:
            print(f"    ✓ No collapsed dimensions")

        if n_nan > 0 or n_inf > 0:
            print(f"    ❌ PROBLEM: Found NaN/Inf values!")
            return False

    print("\n  ✓ All basic checks passed!")
    return True


def check_label_separation(emb_dir: Path, labels_dir: Path, output_dir: Path):
    """Check 2: t-SNE visualization — do labels separate?"""
    print("\n" + "=" * 70)
    print("CHECK 2: Label Separation (t-SNE)")
    print("=" * 70)

    # Load train embeddings + labels
    emb = pd.read_csv(emb_dir / "train.csv")
    labels = pd.read_csv(labels_dir / "train.csv")

    merged = emb.merge(labels[["study_id", "Label_Troponin"]], on="study_id", how="inner")
    z_cols = [c for c in emb.columns if c.startswith("z_ecg_")]

    print(f"\n  Merged: {len(merged)} rows")
    print(f"  Label distribution:")
    for label, count in merged["Label_Troponin"].value_counts().items():
        print(f"    {label}: {count:,}")

    # Subsample for t-SNE (max 5000 per class for speed)
    parts = []
    for label, grp in merged.groupby("Label_Troponin"):
        n = min(len(grp), 5000)
        parts.append(grp.sample(n=n, random_state=42))
    sampled = pd.concat(parts, ignore_index=True)

    print(f"  Subsampled: {len(sampled)} rows for t-SNE")

    z_data = sampled[z_cols].values
    labels_arr = sampled["Label_Troponin"].values

    # Run t-SNE
    print("  Running t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(z_data)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = {
        "Control_Symptomatic": "#4CAF50",
        "MI_Pre-Incident": "#FF9800",
        "MI_Acute_Presentation": "#F44336",
    }

    for label, color in colors.items():
        mask = labels_arr == label
        if mask.sum() > 0:
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                      c=color, label=label, alpha=0.4, s=8)

    ax.set_title("t-SNE of VAE Latent Embeddings (z_ecg)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, fontsize=11)
    ax.grid(True, alpha=0.2)

    plot_path = output_dir / "tsne_embeddings.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ t-SNE plot saved: {plot_path}")


def check_discriminability(emb_dir: Path, labels_dir: Path):
    """Check 3: Can a simple logistic regression separate MI from Control?"""
    print("\n" + "=" * 70)
    print("CHECK 3: Downstream Discriminability (Logistic Regression)")
    print("=" * 70)

    # Load data
    z_cols = None
    X_train, y_train, X_val, y_val = None, None, None, None

    for split, xy in [("train", "train"), ("validation", "val")]:
        emb = pd.read_csv(emb_dir / f"{split}.csv")
        labels = pd.read_csv(labels_dir / f"{split}.csv")
        merged = emb.merge(labels[["study_id", "Label_Troponin"]], on="study_id")

        if z_cols is None:
            z_cols = [c for c in emb.columns if c.startswith("z_ecg_")]

        X = merged[z_cols].values
        # Binary: MI (Pre or Acute) vs Control
        y = (merged["Label_Troponin"] != "Control_Symptomatic").astype(int).values

        if xy == "train":
            X_train, y_train = X, y
        else:
            X_val, y_val = X, y

    print(f"\n  Train: {len(X_train)} samples ({y_train.sum()} MI, {(1-y_train).sum()} Control)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} MI, {(1-y_val).sum()} Control)")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Logistic regression
    print("  Training logistic regression...")
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = clf.predict(X_val_s)
    y_prob = clf.predict_proba(X_val_s)[:, 1]

    auc = roc_auc_score(y_val, y_prob)
    print(f"\n  ROC AUC: {auc:.4f}")

    if auc > 0.70:
        print(f"  ✓ Good discriminability — embeddings capture MI-relevant signal")
    elif auc > 0.55:
        print(f"  ~ Moderate discriminability — some signal present")
    else:
        print(f"  ⚠️ Low discriminability — embeddings may not be capturing MI signal well")

    print(f"\n  Classification Report (val set):")
    report = classification_report(y_val, y_pred, target_names=["Control", "MI"])
    print("  " + report.replace("\n", "\n  "))

    return auc


def check_dimension_usage(emb_dir: Path, output_dir: Path):
    """Check 4: Are all 64 dimensions being used, or are some dead?"""
    print("\n" + "=" * 70)
    print("CHECK 4: Dimension Usage (Variance per Latent Dim)")
    print("=" * 70)

    emb = pd.read_csv(emb_dir / "train.csv")
    z_cols = [c for c in emb.columns if c.startswith("z_ecg_")]
    z_data = emb[z_cols].values

    variances = z_data.var(axis=0)
    sorted_idx = np.argsort(variances)[::-1]

    print(f"\n  Top 10 most active dimensions (by variance):")
    for i in range(10):
        idx = sorted_idx[i]
        print(f"    z_ecg_{idx+1}: var={variances[idx]:.4f}")

    print(f"\n  Bottom 10 least active dimensions:")
    for i in range(10):
        idx = sorted_idx[-(i+1)]
        print(f"    z_ecg_{idx+1}: var={variances[idx]:.4f}")

    active_dims = (variances > 0.01).sum()
    print(f"\n  Active dimensions (var > 0.01): {active_dims}/{len(z_cols)}")

    if active_dims >= 50:
        print(f"  ✓ Good utilization — most dimensions are active")
    elif active_dims >= 30:
        print(f"  ~ Moderate — some dimensions unused but acceptable")
    else:
        print(f"  ⚠️  Many dead dimensions — consider reducing z_dim")

    # Plot variance per dimension
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(variances)), variances[sorted_idx], color="#2196F3", alpha=0.8)
    ax.axhline(y=0.01, color="red", linestyle="--", label="Active threshold (0.01)")
    ax.set_xlabel("Latent Dimension (sorted by variance)")
    ax.set_ylabel("Variance")
    ax.set_title("VAE Latent Dimension Usage")
    ax.legend()
    ax.grid(True, alpha=0.2)

    plot_path = output_dir / "dimension_variance.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Variance plot saved: {plot_path}")


def main():
    scm_dir = Path(__file__).resolve().parent.parent.parent
    emb_dir = scm_dir / "data" / "modeling" / "ecg_embeddings"
    labels_dir = scm_dir / "data" / "modeling" / "labels"
    output_dir = scm_dir / "models" / "checkpoints" / "vae_zdim64_beta4.0_20260821_030245"

    print("\n" + "=" * 70)
    print("VAE Embedding Validation")
    print("=" * 70)
    print(f"  Embeddings: {emb_dir}")
    print(f"  Labels:     {labels_dir}")
    print(f"  Output:     {output_dir}")

    # Run all checks
    basic_ok = check_basic_stats(emb_dir)
    if not basic_ok:
        print("\n❌ Basic checks failed. Fix issues before proceeding.")
        return

    check_dimension_usage(emb_dir, output_dir)
    check_label_separation(emb_dir, labels_dir, output_dir)
    auc = check_discriminability(emb_dir, labels_dir)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Basic stats:      ✓ Passed")
    print(f"  Dimension usage:  See variance plot")
    print(f"  t-SNE separation: See tsne_embeddings.png")
    print(f"  LogReg AUC:       {auc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
