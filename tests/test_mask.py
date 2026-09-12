"""The mask contract: immutable features must survive generation untouched."""

from __future__ import annotations

import numpy as np
import pytest

from shapwgan_ids.shap import build_mask
from shapwgan_ids.shap.mask import FeatureMask

NAMES = [f"f{i}" for i in range(10)]
IMPORTANCE = list(range(10))  # f9 most important, f0 least


def test_immutable_topk_freezes_the_highest_ranked_features():
    mask = build_mask(IMPORTANCE, NAMES, top_k=3, mode="immutable_topk")
    assert mask.n_immutable == 3
    assert set(np.array(NAMES)[mask.immutable_idx]) == {"f7", "f8", "f9"}
    assert mask.ranked_names[0] == "f9"
    assert mask.n_mutable == 7


def test_mutable_topk_is_the_inverse():
    mask = build_mask(IMPORTANCE, NAMES, top_k=3, mode="mutable_topk")
    assert set(np.array(NAMES)[mask.mutable_idx]) == {"f7", "f8", "f9"}


def test_mode_none_freezes_nothing():
    mask = build_mask(IMPORTANCE, NAMES, mode="none")
    assert mask.n_immutable == 0
    assert mask.n_mutable == len(NAMES)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_mask(IMPORTANCE, NAMES, mode="whatever")


def test_apply_keeps_immutable_features_at_original_values():
    mask = build_mask(IMPORTANCE, NAMES, top_k=4, mode="immutable_topk")
    rng = np.random.default_rng(0)
    original = rng.normal(size=(32, len(NAMES))).astype(np.float32)
    generated = rng.normal(size=(32, len(NAMES))).astype(np.float32)

    blended = mask.apply(original, generated)
    np.testing.assert_array_equal(blended[:, mask.immutable_idx], original[:, mask.immutable_idx])
    np.testing.assert_array_equal(blended[:, mask.mutable_idx], generated[:, mask.mutable_idx])
    # inputs untouched, output is a new array
    assert not np.shares_memory(blended, generated)


def test_apply_works_on_a_single_vector():
    mask = build_mask(IMPORTANCE, NAMES, top_k=2, mode="immutable_topk")
    original = np.arange(10, dtype=np.float32)
    generated = np.full(10, -1.0, dtype=np.float32)
    out = mask.apply(original, generated)
    assert out.shape == (10,)
    assert out[8] == original[8] and out[9] == original[9]
    assert out[0] == -1.0


def test_apply_rejects_shape_mismatch():
    mask = build_mask(IMPORTANCE, NAMES, top_k=2)
    with pytest.raises(ValueError):
        mask.apply(np.zeros((4, 10), dtype=np.float32), np.zeros((4, 9), dtype=np.float32))


def test_roundtrip_serialisation(tmp_path):
    mask = build_mask(IMPORTANCE, NAMES, top_k=5, mode="immutable_topk")
    path = mask.save(tmp_path / "artifacts" / "mask.json")
    assert path.is_file()
    restored = FeatureMask.load(path)
    assert restored.names == mask.names
    assert restored.top_k == mask.top_k
    assert np.array_equal(restored.immutable, mask.immutable)


def test_top_k_larger_than_feature_count_is_clamped():
    mask = build_mask(IMPORTANCE, NAMES, top_k=99)
    assert mask.n_immutable == len(NAMES)
