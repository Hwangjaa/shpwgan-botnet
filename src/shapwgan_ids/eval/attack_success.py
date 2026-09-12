"""Attack-success / evasion metrics against an IDS oracle."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from shapwgan_ids.models.surrogate import _as_numpy


def evasion_success_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of malicious samples reclassified as benign (lower detection rate = higher ESR)."""
    malicious = y_true == 1
    if not malicious.any():
        return 0.0
    return float((y_pred[malicious] == 0).mean())


def evaluate_oracle(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute clean-test metrics for an IDS oracle."""
    x_test = _as_numpy(x_test)
    y_test = _as_numpy(y_test)

    probs = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(x_test)
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division="warn")),
        "recall": float(recall_score(y_test, preds, zero_division="warn")),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0,
        "detection_rate": float((preds[y_test == 1] == 1).mean()),
        "evasion_success_rate": evasion_success_rate(y_test, preds),
    }
