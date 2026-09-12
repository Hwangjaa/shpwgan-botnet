"""Surrogate classifier used as black-box oracle approximation (Equation 2.11).

The surrogate must expose ``predict_proba`` so that:

* S(X_adv) = P(y = malicious | X_adv)   (Equation 2.11)
* SHAP explainers can build a global feature ranking from its decisions.

Supported back-ends:
* ``xgboost`` -- fast, deterministic, TreeSHAP-compatible (default).
* ``mlp``     -- small feed-forward network for sensitivity studies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

log = logging.getLogger(__name__)


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class SurrogateModel:
    """Wrapper that exposes ``predict_proba`` for XGBoost and sklearn MLP."""

    def __init__(self, backend: str, model: Any, device: torch.device | None = None) -> None:
        self.backend = backend
        self.model = model
        self.device = device or torch.device("cpu")

    def predict_proba(self, x: Any) -> np.ndarray:
        """Return P(y=malicious | x) for every row (Equation 2.11)."""
        arr = _as_numpy(x)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(arr)
        if self.backend == "xgboost":
            return self.model.predict_proba(arr)[:, 1]
        if self.backend == "mlp":
            return self.model.predict_proba(arr)[:, 1]
        raise ValueError(f"unknown backend {self.backend!r}")

    def predict(self, x: Any) -> np.ndarray:
        arr = _as_numpy(x)
        if hasattr(self.model, "predict"):
            return self.model.predict(arr)
        if self.backend in {"xgboost", "mlp"}:
            return self.model.predict(arr)
        raise ValueError(f"unknown backend {self.backend!r}")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"backend": self.backend, "model": self.model}, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: torch.device | None = None) -> SurrogateModel:
        payload = joblib.load(Path(path))
        return cls(backend=payload["backend"], model=payload["model"], device=device)


def train_surrogate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    backend: str = "xgboost",
    params: dict[str, Any] | None = None,
    random_state: int = 42,
) -> SurrogateModel:
    """Fit a surrogate on labelled (benign/malicious) traffic.

    ``params`` is passed straight to the underlying estimator; keys not supported
    by the chosen backend are ignored.
    """
    x_train = _as_numpy(x_train)
    y_train = _as_numpy(y_train)

    if backend == "xgboost":
        defaults = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
        }
        if params:
            defaults.update(params)
        model = XGBClassifier(**defaults)
        model.fit(x_train, y_train)
        return SurrogateModel(backend="xgboost", model=model)

    if backend == "mlp":
        defaults = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "batch_size": 256,
            "learning_rate_init": 1e-3,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "random_state": random_state,
        }
        if params:
            defaults.update(params)
        model = MLPClassifier(**defaults)
        model.fit(x_train, y_train)
        return SurrogateModel(backend="mlp", model=model)

    raise ValueError(f"unsupported surrogate backend {backend!r}")
