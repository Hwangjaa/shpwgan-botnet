"""Feature importance ranking from SHAP values."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


def aggregate_shap(values: object) -> np.ndarray:
    """Reduce SHAP output to a per-feature mean(|SHAP|) importance vector.

    Accepts what the different explainers return:
    ``(n_samples, n_features)`` for binary/regression, ``(n_samples, n_features, n_classes)``
    or a list of per-class arrays for multiclass. Explainer quirks (raw lists, torch
    tensors) are normalised here so callers only ever see a 1-D float array.
    """
    if isinstance(values, (list, tuple)):
        values = np.mean([aggregate_shap(v) for v in values], axis=0)
    array = np.asarray(values, dtype=np.float64)

    if array.ndim == 3:  # (samples, features, classes) -> average over classes
        array = np.abs(array).mean(axis=(0, 2))
    elif array.ndim == 2:
        array = np.abs(array).mean(axis=0)
    elif array.ndim == 1:
        array = np.abs(array)
    else:
        raise ValueError(f"unsupported SHAP array shape {array.shape}")

    if not np.isfinite(array).all():
        raise ValueError("SHAP importance contains NaN/inf -- check the explainer call")
    return array


def rank_features(importance: Sequence[float], feature_names: Sequence[str]) -> list[tuple[str, float]]:
    """Descending (feature, importance) ranking; ties broken by name for determinism."""
    if len(importance) != len(feature_names):
        raise ValueError(f"importance ({len(importance)}) and names ({len(feature_names)}) length mismatch")
    pairs = list(zip(feature_names, (float(v) for v in importance)))
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs


def top_k_names(importance: Sequence[float], feature_names: Sequence[str], k: int) -> list[str]:
    if k < 0:
        raise ValueError("k must be >= 0")
    return [name for name, _ in rank_features(importance, feature_names)[: min(k, len(feature_names))]]
