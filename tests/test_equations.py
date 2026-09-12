"""Numerical tests that the implementation matches the thesis equations.

2.8  I_j = (1/N) sum_i |phi_ij|
2.9  C_k = (sum_{j=1}^k I_j) / (sum_{j=1}^d I_j)
2.10 M_j in {0,1}; M_j = 0 for top-k immutable, 1 otherwise
2.16 X_adv = X_orig + (epsilon * M)
"""

from __future__ import annotations

import numpy as np

from shapwgan_ids.shap import build_mask
from shapwgan_ids.shap.ranking import aggregate_shap, cumulative_importance

NAMES = [f"f{i}" for i in range(6)]


def test_equation_2_8_mean_absolute_shap():
    rng = np.random.default_rng(0)
    # 10 samples x 6 features
    shap_values = rng.normal(size=(10, 6))
    got = aggregate_shap(shap_values)
    want = np.mean(np.abs(shap_values), axis=0)
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_equation_2_9_cumulative_importance_is_sorted_ratio():
    importance = np.array([1.0, 2.0, 3.0, 4.0])
    ratios, sorted_idx = cumulative_importance(importance)
    # descending order: 4,3,2,1 -> indices 3,2,1,0
    assert sorted_idx == [3, 2, 1, 0]
    total = 10.0
    expected = [4 / total, 7 / total, 9 / total, 10 / total]
    np.testing.assert_allclose(ratios, expected, rtol=1e-10)


def test_equation_2_9_cumulative_coverage_selects_k():
    importance = np.array([5.0, 3.0, 2.0])
    # C_1 = 0.5, C_2 = 0.8, C_3 = 1.0
    mask = build_mask(importance, NAMES[:3], top_k=3, coverage=0.75, mode="immutable_topk")
    # need C_k >= 0.75 -> k=2 (0.8)
    assert mask.top_k == 2
    assert set(np.array(NAMES[:3])[mask.immutable_idx]) == {"f0", "f1"}


def test_equation_2_10_mask_is_binary_and_top_k():
    importance = np.arange(6.0)
    mask = build_mask(importance, NAMES, top_k=2, mode="immutable_topk")
    assert set(np.unique(mask.immutable)) == {True, False}
    assert mask.n_immutable == 2
    # top 2 most important are f5, f4
    assert set(np.array(NAMES)[mask.immutable_idx]) == {"f4", "f5"}


def test_equation_2_16_additive_mask_only_changes_mutable_features():
    importance = np.arange(6.0)
    mask = build_mask(importance, NAMES, top_k=2, mode="immutable_topk")
    rng = np.random.default_rng(1)
    x_orig = rng.normal(size=(8, 6)).astype(np.float32)
    perturbation = rng.normal(size=(8, 6)).astype(np.float32)

    x_adv = mask.apply_additive(x_orig, perturbation)

    # immutable features unchanged
    np.testing.assert_array_equal(x_adv[:, mask.immutable_idx], x_orig[:, mask.immutable_idx])
    # mutable features: x_orig + perturbation
    np.testing.assert_array_equal(
        x_adv[:, mask.mutable_idx],
        x_orig[:, mask.mutable_idx] + perturbation[:, mask.mutable_idx],
    )
