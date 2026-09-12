"""XGBoost IDS oracle (used as target detector, not the surrogate)."""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from .surrogate import _as_numpy


def train_ids_xgb(x_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any] | None = None) -> XGBClassifier:
    """Train the XGBoost IDS oracle on clean or loop-augmented data."""
    x_train = _as_numpy(x_train)
    y_train = _as_numpy(y_train)
    defaults = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)
    model = XGBClassifier(**defaults)
    model.fit(x_train, y_train)
    return model
