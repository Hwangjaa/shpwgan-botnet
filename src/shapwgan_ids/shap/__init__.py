"""SHAP-based feature importance and the mutation mask constraint."""

from __future__ import annotations

from .mask import MODES, FeatureMask, build_mask
from .ranking import aggregate_shap, rank_features, top_k_names

__all__ = [
    "MODES",
    "FeatureMask",
    "aggregate_shap",
    "build_mask",
    "rank_features",
    "top_k_names",
]
