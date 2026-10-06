"""
Phase H: Baseline Predictive Models

Trains and evaluates 4 baseline classifiers for MI_Acute_Presentation
(binary classification) on the master dataset.

Models:
  1. Logistic Regression (clinical features only)
  2. XGBoost (ECG morphological features + clinical)
  3. XGBoost (VAE embeddings + clinical)
  4. XGBoost (all features combined)

Evaluation:
  - AUROC, AUPRC, Brier Score, ECE
  - Calibration plots, ROC/PR curves
  - Subgroup fairness (age, sex, diabetes, prior_mi)
  - Feature importance (top 30)

Usage:
    python src/models/train_baselines.py

Output:
    models/baselines/
        results.json
        model_comparison.png
        calibration_plots.png
        subgroup_fairness.png
        feature_importance.png
        xgb_ecg_features_clinical.json    (XGBoost model)
        xgb_embeddings_clinical.json      (XGBoost model)
        xgb_all_features.json             (XGBoost model)
        logreg_clinical.pkl               (Logistic regression model)
"""

import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCM_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_DIR = SCM_DIR / "data" / "modeling" / "master"
OUTPUT_DIR = SCM_DIR / "models" / "baselines"

# ---------------------------------------------------------------------------
# Feature column definitions (derived from actual master parquet schema)
# ---------------------------------------------------------------------------

# 64 VAE latent dimensions
Z_ECG_COLS = [f"z_ecg_{i}" for i in range(1, 65)]

# 95 morphological ECG features (from ecg_features)
# Global timing features
_ECG_GLOBAL = [
    "QRS_Dur_Global", "QRS_Dur_Global_iqr", "QRS_Dur_Global_count",
    "RR_Mean_Global", "RR_Mean_Global_iqr",
]
# S-wave amplitudes (V1-V3) with IQR
_ECG_S_AMP = [
    "S_Amp_V1", "S_Amp_V1_iqr",
    "S_Amp_V2", "S_Amp_V2_iqr",
    "S_Amp_V3", "S_Amp_V3_iqr",
]
# Per-lead features: Q_Amp, R_Amp, T_Amp (each with _iqr), ST_Elev
_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_ECG_LEAD = []
for lead in _LEADS:
    for feat in ["Q_Amp", "R_Amp", "T_Amp"]:
        _ECG_LEAD.append(f"{feat}_{lead}")
        _ECG_LEAD.append(f"{feat}_{lead}_iqr")
    _ECG_LEAD.append(f"ST_Elev_{lead}")

ECG_FEATURE_COLS = _ECG_GLOBAL + _ECG_S_AMP + _ECG_LEAD

# 17 clinical features
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

# Target
TARGET_COL = "Label_Troponin"
POSITIVE_LABEL = "MI_Acute_Presentation"

# Subgroup definitions for fairness
SUBGROUPS = {
    "age_group": {
        "col": "anchor_age",
        "bins": [0, 50, 70, 200],
        "labels": ["<50", "50-70", ">70"],
    },
    "sex": {
        "col": "sex_male",
        "map": {1: "Male", 0: "Female"},
    },
    "diabetes": {
        "col": "diabetes",
        "map": {1: "Yes", 0: "No"},
    },
    "prior_mi": {
        "col": "prior_mi",
        "map": {1: "Yes", 0: "No"},
    },
}

# Model configurations
MODEL_CONFIGS = {
    "logreg_clinical": {
        "name": "Logistic Regression (Clinical Only)",
        "feature_cols": CLINICAL_COLS,
        "model_type": "logreg",
    },
    "xgb_ecg_features_clinical": {
        "name": "XGBoost (ECG Features + Clinical)",
        "feature_cols": ECG_FEATURE_COLS + CLINICAL_COLS,
        "model_type": "xgboost",
    },
    "xgb_embeddings_clinical": {
        "name": "XGBoost (Embeddings + Clinical)",
        "feature_cols": Z_ECG_COLS + CLINICAL_COLS,
        "model_type": "xgboost",
    },
    "xgb_all_features": {
        "name": "XGBoost (All Features)",
        "feature_cols": ECG_FEATURE_COLS + Z_ECG_COLS + CLINICAL_COLS,
        "model_type": "xgboost",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load train/validation/test splits from master parquet files."""
    splits = {}
    for split in ["train", "validation", "test"]:
        path = MASTER_DIR / f"{split}.parquet"
        df = pd.read_parquet(path)
        splits[split] = df
        print(f"  {split}: {len(df):>8,} rows")
    return splits


def prepare_xy(df, feature_cols):
    """Extract feature matrix X and binary target y from a dataframe.

    Target: MI_Acute_Presentation vs rest (binary).
    MI_Pre-Incident and Control_Symptomatic are both treated as negative.
    """
    X = df[feature_cols].values.astype(np.float32)
    y = (df[TARGET_COL] == POSITIVE_LABEL).astype(int).values
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_logreg(X_train, y_train, X_val, y_val):
    """Train a logistic regression with class balancing."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        l1_ratio=0,
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    val_prob = model.predict_proba(X_val_s)[:, 1]
    val_auc = roc_auc_score(y_val, val_prob)
    print(f"    Val AUROC: {val_auc:.4f}")

    return model, scaler


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with early stopping on validation set."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=50,
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",     # fast CPU training
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_prob = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_prob)
    best_iter = model.best_iteration
    print(f"    Val AUROC: {val_auc:.4f} (best_iteration={best_iter})")

    return model, None  # no scaler needed for XGBoost


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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


def evaluate_model(model, scaler, X_test, y_test, feature_cols):
    """Compute all evaluation metrics for a trained model."""
    if scaler is not None:
        X_test_s = scaler.transform(X_test)
        y_prob = model.predict_proba(X_test_s)[:, 1]
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    # Discrimination
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)

    # Calibration
    brier = brier_score_loss(y_test, y_prob)
    ece = compute_ece(y_test, y_prob)

    # Clinical utility at 90% sensitivity threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    idx_90 = np.argmin(np.abs(tpr - 0.90))
    threshold_90 = thresholds[idx_90]
    sensitivity_90 = tpr[idx_90]
    specificity_90 = 1 - fpr[idx_90]

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    idx_opt = np.argmax(j_scores)
    threshold_opt = thresholds[idx_opt]
    y_pred_opt = (y_prob >= threshold_opt).astype(int)
    cm = confusion_matrix(y_test, y_pred_opt)
    tn, fp, fn, tp = cm.ravel()
    ppv_opt = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv_opt = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Curves for plotting
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
    cal_prob, cal_frac = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier_score": float(brier),
        "ece": float(ece),
        "sensitivity_at_90": float(sensitivity_90),
        "specificity_at_90": float(specificity_90),
        "threshold_at_90_sens": float(threshold_90),
        "threshold_optimal": float(threshold_opt),
        "ppv_optimal": float(ppv_opt),
        "npv_optimal": float(npv_opt),
        "n_features": len(feature_cols),
    }

    curves = {
        "roc": (fpr_curve, tpr_curve),
        "pr": (rec_curve, prec_curve),
        "calibration": (cal_prob, cal_frac),
    }

    return metrics, curves, y_prob


def subgroup_auroc(df_test, y_prob, model_name):
    """Compute AUROC for each subgroup."""
    y_true = (df_test[TARGET_COL] == POSITIVE_LABEL).astype(int).values
    results = {}

    for sg_name, sg_def in SUBGROUPS.items():
        col = sg_def["col"]
        sg_results = {}

        if "bins" in sg_def:
            # Binned subgroup (age)
            groups = pd.cut(
                df_test[col], bins=sg_def["bins"], labels=sg_def["labels"], right=False
            )
            for label in sg_def["labels"]:
                mask = (groups == label).values
                if mask.sum() > 0 and y_true[mask].sum() > 0:
                    sg_results[label] = float(roc_auc_score(y_true[mask], y_prob[mask]))
                else:
                    sg_results[label] = None
        else:
            # Mapped subgroup (binary)
            for val, label in sg_def["map"].items():
                mask = (df_test[col] == val).values
                if mask.sum() > 0 and y_true[mask].sum() > 0:
                    sg_results[label] = float(roc_auc_score(y_true[mask], y_prob[mask]))
                else:
                    sg_results[label] = None

        results[sg_name] = sg_results

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_comparison(all_curves, all_metrics, prevalence, output_dir):
    """ROC and PR curves for all models side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    # ROC curves
    for i, (model_key, curves) in enumerate(all_curves.items()):
        fpr, tpr = curves["roc"]
        auc = all_metrics[model_key]["auroc"]
        name = MODEL_CONFIGS[model_key]["name"]
        ax1.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{name}\n(AUROC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.2)

    # PR curves
    for i, (model_key, curves) in enumerate(all_curves.items()):
        rec, prec = curves["pr"]
        ap = all_metrics[model_key]["auprc"]
        name = MODEL_CONFIGS[model_key]["name"]
        ax2.plot(rec, prec, color=colors[i], lw=2,
                 label=f"{name}\n(AUPRC={ap:.3f})")
    ax2.axhline(y=prevalence, color="gray", linestyle="--", lw=1,
                label=f"Random (prevalence={prevalence:.3f})")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_calibration(all_curves, output_dir):
    """Calibration plots for all models."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    for i, (model_key, curves) in enumerate(all_curves.items()):
        ax = axes[i]
        cal_prob, cal_frac = curves["calibration"]
        name = MODEL_CONFIGS[model_key]["name"]

        ax.plot(cal_prob, cal_frac, "s-", color=colors[i], lw=2, markersize=6)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives", fontsize=10)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_plots.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_subgroup_fairness(all_subgroups, output_dir):
    """Grouped bar charts for subgroup AUROC."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sg_names = list(SUBGROUPS.keys())

    for ax_i, sg_name in enumerate(sg_names):
        ax = axes[ax_i]
        model_names = []
        group_labels = []
        values = []

        # Collect all group labels for this subgroup
        for model_key in MODEL_CONFIGS:
            if model_key in all_subgroups and sg_name in all_subgroups[model_key]:
                sg_data = all_subgroups[model_key][sg_name]
                if not group_labels:
                    group_labels = list(sg_data.keys())
                model_names.append(MODEL_CONFIGS[model_key]["name"].split("(")[0].strip())
                values.append([sg_data.get(g) for g in group_labels])

        if not values:
            continue

        x = np.arange(len(group_labels))
        width = 0.18
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

        for j, (mname, vals) in enumerate(zip(model_names, values)):
            bar_vals = [v if v is not None else 0 for v in vals]
            ax.bar(x + j * width, bar_vals, width, label=mname, color=colors[j], alpha=0.85)

        ax.set_xlabel(sg_name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(f"Subgroup: {sg_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(group_labels, fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim([0.4, 1.0])
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "subgroup_fairness.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_cols, model_name, output_dir):
    """Top 30 feature importances from an XGBoost model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:30]

    fig, ax = plt.subplots(figsize=(10, 8))
    names = [feature_cols[i] for i in indices]
    vals = importances[indices]

    # Color by feature group
    colors = []
    for name in names:
        if name.startswith("z_ecg_"):
            colors.append("#3498db")  # blue for embeddings
        elif name in CLINICAL_COLS:
            colors.append("#e74c3c")  # red for clinical
        else:
            colors.append("#2ecc71")  # green for ECG features

    ax.barh(range(len(names)), vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_title(f"Top 30 Features — {model_name}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="VAE Embedding"),
        Patch(facecolor="#2ecc71", label="ECG Feature"),
        Patch(facecolor="#e74c3c", label="Clinical"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_model(model, scaler, model_key, output_dir):
    """Save trained model to disk."""
    with open(output_dir / f"{model_key}.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Phase H: Baseline Predictive Models")
    print(f"  Master data: {MASTER_DIR}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"  Target:      {TARGET_COL} == '{POSITIVE_LABEL}' (binary)")
    print(f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    splits = load_data()
    df_train = splits["train"]
    df_val = splits["validation"]
    df_test = splits["test"]

    n_pos_train = (df_train[TARGET_COL] == POSITIVE_LABEL).sum()
    n_neg_train = len(df_train) - n_pos_train
    prevalence = n_pos_train / len(df_train)
    print(f"  Train prevalence: {prevalence:.4f} ({n_pos_train:,} pos / {n_neg_train:,} neg)")
    print(f"  scale_pos_weight: {n_neg_train / n_pos_train:.2f}")

    # -----------------------------------------------------------------------
    # 2. Verify feature columns exist
    # -----------------------------------------------------------------------
    print("\n[2/5] Verifying feature columns...")
    all_features = set()
    for cfg in MODEL_CONFIGS.values():
        all_features.update(cfg["feature_cols"])

    missing = all_features - set(df_train.columns)
    if missing:
        print(f"  ERROR: Missing columns in master parquet: {missing}")
        return
    print(f"  All {len(all_features)} unique feature columns verified.")

    # -----------------------------------------------------------------------
    # 3. Train all models
    # -----------------------------------------------------------------------
    print("\n[3/5] Training models...")
    trained_models = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        feature_cols = cfg["feature_cols"]
        print(f"\n  --- {cfg['name']} ({len(feature_cols)} features) ---")

        X_train, y_train = prepare_xy(df_train, feature_cols)
        X_val, y_val = prepare_xy(df_val, feature_cols)

        if cfg["model_type"] == "logreg":
            model, scaler = train_logreg(X_train, y_train, X_val, y_val)
        else:
            model, scaler = train_xgboost(X_train, y_train, X_val, y_val)

        trained_models[model_key] = (model, scaler, feature_cols)
        save_model(model, scaler, model_key, OUTPUT_DIR)
        print(f"    Model saved: {model_key}")

    # -----------------------------------------------------------------------
    # 4. Evaluate all models on test set
    # -----------------------------------------------------------------------
    print("\n[4/5] Evaluating on test set...")
    all_metrics = {}
    all_curves = {}
    all_subgroups = {}

    for model_key, (model, scaler, feature_cols) in trained_models.items():
        print(f"\n  --- {MODEL_CONFIGS[model_key]['name']} ---")

        X_test, y_test = prepare_xy(df_test, feature_cols)
        metrics, curves, y_prob = evaluate_model(model, scaler, X_test, y_test, feature_cols)

        all_metrics[model_key] = metrics
        all_curves[model_key] = curves

        print(f"    AUROC:       {metrics['auroc']:.4f}")
        print(f"    AUPRC:       {metrics['auprc']:.4f}")
        print(f"    Brier Score: {metrics['brier_score']:.4f}")
        print(f"    ECE:         {metrics['ece']:.4f}")
        print(f"    Sens@90%:    {metrics['sensitivity_at_90']:.4f}")
        print(f"    Spec@90%:    {metrics['specificity_at_90']:.4f}")

        # Subgroup analysis
        sg = subgroup_auroc(df_test, y_prob, model_key)
        all_subgroups[model_key] = sg
        for sg_name, sg_data in sg.items():
            vals = [f"{k}={v:.3f}" if v else f"{k}=N/A" for k, v in sg_data.items()]
            print(f"    {sg_name}: {', '.join(vals)}")

    # -----------------------------------------------------------------------
    # 5. Generate plots and save results
    # -----------------------------------------------------------------------
    print("\n[5/5] Generating plots and report...")

    # a) ROC + PR curves
    plot_model_comparison(all_curves, all_metrics, prevalence, OUTPUT_DIR)
    print("  Saved: model_comparison.png")

    # b) Calibration plots
    plot_calibration(all_curves, OUTPUT_DIR)
    print("  Saved: calibration_plots.png")

    # c) Subgroup fairness
    plot_subgroup_fairness(all_subgroups, OUTPUT_DIR)
    print("  Saved: subgroup_fairness.png")

    # d) Feature importance (best XGBoost model)
    best_xgb_key = max(
        [k for k in all_metrics if k.startswith("xgb_")],
        key=lambda k: all_metrics[k]["auroc"]
    )
    best_xgb_model, _, best_xgb_feats = trained_models[best_xgb_key]
    plot_feature_importance(
        best_xgb_model, best_xgb_feats,
        MODEL_CONFIGS[best_xgb_key]["name"], OUTPUT_DIR
    )
    print("  Saved: feature_importance.png")

    # e) Save all results as JSON
    results = {
        "timestamp": datetime.now().isoformat(),
        "target": f"{TARGET_COL} == '{POSITIVE_LABEL}'",
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
        "prevalence_train": float(prevalence),
        "models": {},
    }
    for model_key in MODEL_CONFIGS:
        results["models"][model_key] = {
            "name": MODEL_CONFIGS[model_key]["name"],
            "n_features": len(MODEL_CONFIGS[model_key]["feature_cols"]),
            "metrics": all_metrics[model_key],
            "subgroup_auroc": all_subgroups.get(model_key, {}),
        }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved: results.json")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE MODEL COMPARISON (Test Set)")
    print("=" * 70)
    print(f"  {'Model':<45s} {'AUROC':>7s} {'AUPRC':>7s} {'Brier':>7s} {'ECE':>7s}")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for model_key in MODEL_CONFIGS:
        m = all_metrics[model_key]
        name = MODEL_CONFIGS[model_key]["name"]
        print(f"  {name:<45s} {m['auroc']:>7.4f} {m['auprc']:>7.4f} {m['brier_score']:>7.4f} {m['ece']:>7.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
