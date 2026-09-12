"""Feature importance ranking from SHAP values."""

from __future__ import annotations

import logging
from collections.abc import Sequence

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


def rank_features(importance: Sequence[float] | np.ndarray, feature_names: Sequence[str]) -> list[tuple[str, float]]:
    """Descending (feature, importance) ranking; ties broken by name for determinism."""
    values = np.asarray(importance)
    if values.shape[0] != len(feature_names):
        raise ValueError(f"importance ({values.shape[0]}) and names ({len(feature_names)}) length mismatch")
    pairs = list(zip(feature_names, (float(v) for v in values), strict=True))
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs


def cumulative_importance(importance: Sequence[float] | np.ndarray) -> tuple[list[float], list[int]]:
    """Cumulative SHAP importance ratio C_k (Equation 2.9).

    Returns (ratios, sorted_indices) where ratios[k] is C_{k+1} after sorting
    features by descending absolute importance.
    """
    values = np.asarray(importance, dtype=np.float64)
    if values.size == 0:
        return [], []
    total = float(values.sum())
    if total <= 0:
        raise ValueError("total importance must be positive for cumulative ratio")
    sorted_idx = np.argsort(-values)
    cumulative = np.cumsum(values[sorted_idx])
    ratios = (cumulative / total).tolist()
    return ratios, sorted_idx.tolist()


def top_k_by_cumulative(
    importance: Sequence[float] | np.ndarray,
    coverage: float = 0.9,
    min_k: int = 1,
    max_k: int | None = None,
) -> int:
    """Return smallest k such that C_k >= coverage (Equation 2.9 threshold variant)."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    values = np.asarray(importance, dtype=np.float64)
    max_k = max_k if max_k is not None else int(values.size)
    ratios, _ = cumulative_importance(values)
    for k, ratio in enumerate(ratios, start=1):
        if ratio >= coverage and k >= min_k:
            return min(k, max_k)
    return min(int(values.size), max_k)


def top_k_names(importance: Sequence[float] | np.ndarray, feature_names: Sequence[str], k: int) -> list[str]:
    if k < 0:
        raise ValueError("k must be >= 0")
    return [name for name, _ in rank_features(importance, feature_names)[: min(k, len(feature_names))]]
