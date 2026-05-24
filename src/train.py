"""
train.py — RiskRadar
--------------------
Trains an XGBoost classifier on the Victorian Road Crash dataset.

Key design decision: we train a BINARY classifier (high vs moderate risk)
since the dataset has only 3 "low" risk samples — not enough to learn from.
The model predicts: 0 = moderate risk, 1 = high risk.

At prediction time, "low" is assigned when conditions are clearly safe
(daytime, dry, low speed zone, no adverse weather).

Usage:
    python src/train.py
    python src/train.py --data data/victorian_road_crash_data.csv
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_features, FEATURE_COLS, RISK_LABELS

DEFAULT_DATA   = "data/victorian_road_crash_data.csv"
DEFAULT_OUTPUT = "models/riskradar_model.joblib"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42


def train(data_path: str, output_path: str) -> None:

    print("=" * 55)
    print("  RiskRadar — Model Training")
    print("=" * 55)

    # ── 1. Load features ───────────────────────────────────────────────────────
    df = build_features(data_path, melbourne_only=True, verbose=True)

    # ── 2. Binary classification: drop the 3 "low" samples, high vs moderate ──
    # "low" has only 3 samples — impossible to learn. We train binary:
    #   0 = moderate risk (Other injury accident)
    #   1 = high risk (Fatal or Serious injury)
    df = df[df["risk_label"] != 0].copy()
    df["risk_label"] = (df["risk_label"] == 2).astype(int)  # 1=high, 0=moderate

    X = df[FEATURE_COLS]
    y = df["risk_label"]

    print(f"\nBinary classification: high vs moderate risk")
    print(f"  High risk   : {(y==1).sum():,} ({100*(y==1).mean():.1f}%)")
    print(f"  Moderate    : {(y==0).sum():,} ({100*(y==0).mean():.1f}%)")
    print(f"  Features    : {len(FEATURE_COLS)}")

    # ── 3. Train / test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain set : {len(X_train):,} samples")
    print(f"Test set  : {len(X_test):,} samples")

    # ── 4. Class weight (high risk is minority) ────────────────────────────────
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight: {scale_pos_weight:.2f} (balances class imbalance)")

    # ── 5. Train XGBoost ───────────────────────────────────────────────────────
    print("\nTraining XGBoost classifier...")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=2,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print("Training complete.")

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_pred_prob)

    print(f"\n{'=' * 55}")
    print(f"  Accuracy  : {accuracy * 100:.1f}%")
    print(f"  ROC-AUC   : {roc_auc:.3f}  ← key metric for imbalanced data")
    print(f"{'=' * 55}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["moderate", "high"]))

    # Cross-validation for robustness check
    print("Cross-validation (5-fold AUC)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── 7. Plots ───────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    importances.plot(kind="barh", ax=axes[0], color="#4a90d9")
    axes[0].set_title("Feature importance")
    axes[0].set_xlabel("Score")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["moderate", "high"])
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title(f"Confusion matrix — AUC: {roc_auc:.3f}")

    plt.tight_layout()
    plot_path = "models/training_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlots saved → {plot_path}")

    # ── 8. Save model ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    joblib.dump({
        "model":        model,
        "feature_cols": FEATURE_COLS,
        "risk_labels":  {0: "moderate", 1: "high"},
        "model_type":   "binary",   # high vs moderate
        "accuracy":     round(accuracy, 4),
        "roc_auc":      round(roc_auc, 4),
        "cv_auc_mean":  round(cv_scores.mean(), 4),
        "trained_on":   pd.Timestamp.now().isoformat(),
    }, output_path)

    print(f"Model saved → {output_path}")
    print(f"\nDone! Accuracy: {accuracy*100:.1f}%  |  ROC-AUC: {roc_auc:.3f}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RiskRadar XGBoost model")
    parser.add_argument("--data",   default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train(data_path=args.data, output_path=args.output)