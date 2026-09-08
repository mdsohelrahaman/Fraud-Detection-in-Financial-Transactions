"""Imbalance-aware supervised models and evaluation."""
from typing import Dict, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, ...]:
    return train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )


def oversample_smote(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE only to the training data."""
    return SMOTE(random_state=42).fit_resample(X_train, y_train)


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """Train Logistic Regression and Random Forest with imbalance awareness."""
    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_binary_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Return fraud-focused metrics."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "positive_rate": float(predictions.mean()),
    }


def find_recall_threshold(y_true: np.ndarray, scores: np.ndarray, target_recall: float = 0.90) -> float:
    """Select the highest threshold meeting the requested recall where possible."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    candidates = thresholds[recall[:-1] >= target_recall]
    if len(candidates) == 0:
        return 0.5
    return float(candidates.max())


def print_report(model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> None:
    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= threshold).astype(int)
    print(classification_report(y_test, predictions, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, scores):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, scores):.4f}")
