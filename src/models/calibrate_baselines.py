"""
Phase H.7: Post-Hoc Model Calibration

Applies isotonic regression calibration to each baseline model using the
validation set, then re-evaluates calibration on the test set.

Usage:
    python src/models/calibrate_baselines.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

SCM_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_DIR = SCM_DIR / "data" / "modeling" / "master"
BASELINE_DIR = SCM_DIR / "models" / "baselines"

TARGET_COL = "Label_Troponin"
POSITIVE_LABEL = "MI_Acute_Presentation"

# Same feature definitions as train_baselines.py
Z_ECG_COLS = [f"z_ecg_{i}" for i in range(1, 65)]

_ECG_GLOBAL = [
    "QRS_Dur_Global", "QRS_Dur_Global_iqr", "QRS_Dur_Global_count",
    "RR_Mean_Global", "RR_Mean_Global_iqr",
]
_ECG_S_AMP = [
    "S_Amp_V1", "S_Amp_V1_iqr", "S_Amp_V2", "S_Amp_V2_iqr",
    "S_Amp_V3", "S_Amp_V3_iqr",
]
_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_ECG_LEAD = []
for lead in _LEADS:
    for feat in ["Q_Amp", "R_Amp", "T_Amp"]:
        _ECG_LEAD.append(f"{feat}_{lead}")
        _ECG_LEAD.append(f"{feat}_{lead}_iqr")
    _ECG_LEAD.append(f"ST_Elev_{lead}")
ECG_FEATURE_COLS = _ECG_GLOBAL + _ECG_S_AMP + _ECG_LEAD

COMORBIDITY_COLS = ["diabetes", "hypertension", "ckd", "prior_mi"]
TREATMENT_COLS = [
    "statin_before", "statin_during", "statin_after",
    "beta_blocker_before", "beta_blocker_during",
    "acei_before", "acei_during",
    "aspirin_before", "aspirin_during",
    "vasopressor_during", "mech_vent_during",
]
DEMOGRAPHIC_COLS = ["anchor_age", "sex_male"]
CLINICAL_COLS = COMORBIDITY_COLS + TREATMENT_COLS + DEMOGRAPHIC_COLS

MODEL_CONFIGS = {
    "logreg_clinical": {
        "name": "Logistic Regression (Clinical Only)",
        "feature_cols": CLINICAL_COLS,
    },
    "xgb_ecg_features_clinical": {
        "name": "XGBoost (ECG Features + Clinical)",
        "feature_cols": ECG_FEATURE_COLS + CLINICAL_COLS,
    },
    "xgb_embeddings_clinical": {
        "name": "XGBoost (Embeddings + Clinical)",
        "feature_cols": Z_ECG_COLS + CLINICAL_COLS,
    },
    "xgb_all_features": {
        "name": "XGBoost (All Features)",
        "feature_cols": ECG_FEATURE_COLS + Z_ECG_COLS + CLINICAL_COLS,
    },
}


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


def get_probs(model, scaler, X):
    """Get predicted probabilities from a model."""
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]


def main():
    print("=" * 70)
    print("Phase H.7: Post-Hoc Calibration (Isotonic Regression)")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df_val = pd.read_parquet(MASTER_DIR / "validation.parquet")
    df_test = pd.read_parquet(MASTER_DIR / "test.parquet")
    y_val = (df_val[TARGET_COL] == POSITIVE_LABEL).astype(int).values
    y_test = (df_test[TARGET_COL] == POSITIVE_LABEL).astype(int).values
    print(f"  Validation: {len(df_val):,} rows ({y_val.sum():,} positives)")
    print(f"  Test:       {len(df_test):,} rows ({y_test.sum():,} positives)")

    # Process each model
    results_before = {}
    results_after = {}
    all_probs_before = {}
    all_probs_after = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n--- {cfg['name']} ---")
        feature_cols = cfg["feature_cols"]

        # Load trained model
        pkl_path = BASELINE_DIR / f"{model_key}.pkl"
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        scaler = saved["scaler"]

        X_val = df_val[feature_cols].values.astype(np.float32)
        X_test = df_test[feature_cols].values.astype(np.float32)

        # Get uncalibrated probabilities
        prob_val_raw = get_probs(model, scaler, X_val)
        prob_test_raw = get_probs(model, scaler, X_test)

        # Before calibration metrics
        auroc_before = roc_auc_score(y_test, prob_test_raw)
        auprc_before = average_precision_score(y_test, prob_test_raw)
        brier_before = brier_score_loss(y_test, prob_test_raw)
        ece_before = compute_ece(y_test, prob_test_raw)

        results_before[model_key] = {
            "auroc": auroc_before, "auprc": auprc_before,
            "brier": brier_before, "ece": ece_before,
        }
        all_probs_before[model_key] = prob_test_raw

        # Fit isotonic calibration on validation set predicted probabilities
        iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso_reg.fit(prob_val_raw, y_val)

        # Get calibrated test probabilities
        prob_test_cal = iso_reg.predict(prob_test_raw)

        # After calibration metrics
        auroc_after = roc_auc_score(y_test, prob_test_cal)
        auprc_after = average_precision_score(y_test, prob_test_cal)
        brier_after = brier_score_loss(y_test, prob_test_cal)
        ece_after = compute_ece(y_test, prob_test_cal)

        results_after[model_key] = {
            "auroc": auroc_after, "auprc": auprc_after,
            "brier": brier_after, "ece": ece_after,
        }
        all_probs_after[model_key] = prob_test_cal

        print(f"  AUROC:  {auroc_before:.4f} -> {auroc_after:.4f} ({auroc_after - auroc_before:+.4f})")
        print(f"  AUPRC:  {auprc_before:.4f} -> {auprc_after:.4f} ({auprc_after - auprc_before:+.4f})")
        print(f"  Brier:  {brier_before:.4f} -> {brier_after:.4f} ({brier_after - brier_before:+.4f})")
        print(f"  ECE:    {ece_before:.4f} -> {ece_after:.4f} ({ece_after - ece_before:+.4f})")

        # Save calibrated model (original model + isotonic regressor)
        cal_path = BASELINE_DIR / f"{model_key}_calibrated.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "isotonic": iso_reg}, f)

    # -----------------------------------------------------------------------
    # Comparison plot: Before vs After on SAME axes (cleaner)
    # -----------------------------------------------------------------------
    print("\nGenerating calibration comparison plots...")

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    colors = ["#e74c3c", "#27ae60", "#2980b9", "#8e44ad"]

    for i, (model_key, cfg) in enumerate(MODEL_CONFIGS.items()):
        ax = axes[i]
        name = cfg["name"]

        prob_before = all_probs_before[model_key]
        prob_after = all_probs_after[model_key]

        # Calibration curves (quantile strategy, 10 bins)
        cal_pred_b, cal_frac_b = calibration_curve(
            y_test, prob_before, n_bins=10, strategy="quantile"
        )
        cal_pred_a, cal_frac_a = calibration_curve(
            y_test, prob_after, n_bins=10, strategy="quantile"
        )

        # Metrics
        ece_b = results_before[model_key]["ece"]
        ece_a = results_after[model_key]["ece"]
        brier_b = results_before[model_key]["brier"]
        brier_a = results_after[model_key]["brier"]

        # Plot BOTH on same axes
        ax.plot(cal_pred_b, cal_frac_b, "o--", color=colors[i], lw=2,
                markersize=7, alpha=0.5,
                label=f"Before (ECE={ece_b:.3f}, Brier={brier_b:.3f})")
        ax.plot(cal_pred_a, cal_frac_a, "s-", color=colors[i], lw=2.5,
                markersize=7,
                label=f"After (ECE={ece_a:.4f}, Brier={brier_a:.3f})")

        # Diagonal = perfect calibration
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        if i == 0:
            ax.set_ylabel("Observed Fraction of Positives", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.15)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect("equal")

    fig.suptitle(
        "Reliability Diagrams: Before vs After Isotonic Calibration\n"
        "(Quantile bins — each bin has equal number of samples)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(BASELINE_DIR / "calibration_before_after.png", dpi=200,
                bbox_inches="tight")
    plt.close()
    print("  Saved: calibration_before_after.png")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CALIBRATION COMPARISON (Test Set)")
    print("=" * 80)
    print(f"  {'Model':<40s} {'ECE Before':>10s} {'ECE After':>10s} {'Change':>8s} {'Brier Before':>12s} {'Brier After':>12s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*12} {'-'*12}")
    for model_key in MODEL_CONFIGS:
        name = MODEL_CONFIGS[model_key]["name"][:40]
        eb = results_before[model_key]["ece"]
        ea = results_after[model_key]["ece"]
        bb = results_before[model_key]["brier"]
        ba = results_after[model_key]["brier"]
        print(f"  {name:<40s} {eb:>10.4f} {ea:>10.4f} {ea - eb:>+8.4f} {bb:>12.4f} {ba:>12.4f}")
    print("=" * 80)

    # Save results
    cal_results = {
        "method": "isotonic_regression",
        "calibration_set": "validation",
        "test_set": "test",
        "before": {k: v for k, v in results_before.items()},
        "after": {k: v for k, v in results_after.items()},
    }
    with open(BASELINE_DIR / "calibration_results.json", "w") as f:
        json.dump(cal_results, f, indent=2)
    print("\nSaved: calibration_results.json")


if __name__ == "__main__":
    main()
