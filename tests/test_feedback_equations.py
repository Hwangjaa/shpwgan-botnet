"""Numerical checks for Equations 2.11-2.13 and 2.17."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from shapwgan_ids.loop.feedback import (
    attack_success_rate,
    combined_generator_loss,
    evasion_loss,
    surrogate_confidence,
)
from shapwgan_ids.models.surrogate import SurrogateModel


class _FakeSurrogate:
    """Deterministic surrogate that always predicts the requested probability."""

    def __init__(self, probs: np.ndarray) -> None:
        self.probs = np.asarray(probs, dtype=np.float32)

    def predict_proba(self, x: Any) -> np.ndarray:
        n = x.shape[0] if hasattr(x, "shape") else len(x)
        if len(self.probs) == 1:
            return np.repeat(self.probs, n)
        return self.probs[:n]


def test_equation_2_11_confidence_is_malicious_probability():
    probs = np.array([0.9, 0.7, 0.3, 0.1])
    surrogate = SurrogateModel(backend="dummy", model=_FakeSurrogate(probs))
    x_adv = torch.randn(4, 4)
    got = surrogate_confidence(x_adv, surrogate)
    np.testing.assert_allclose(got.numpy(), probs, atol=1e-6)


def test_equation_2_12_evasion_loss_is_mean_confidence():
    probs = np.array([0.9, 0.7, 0.3, 0.1])
    surrogate = SurrogateModel(backend="dummy", model=_FakeSurrogate(probs))
    x_adv = torch.randn(4, 4)
    loss = evasion_loss(x_adv, surrogate)
    assert pytest.approx(float(loss), abs=1e-5) == probs.mean()


def test_equation_2_13_combined_loss_matches_formula():
    probs = np.array([0.5, 0.5, 0.5])
    surrogate = SurrogateModel(backend="dummy", model=_FakeSurrogate(probs))
    x_adv = torch.randn(3, 4)
    w_loss = torch.tensor(1.2)
    alpha = 0.7
    loss, metrics = combined_generator_loss(w_loss, x_adv, surrogate, alpha=alpha)
    expected = 1.2 + 0.7 * 0.5
    assert pytest.approx(float(loss), abs=1e-5) == expected
    assert metrics["evasion_loss"] == pytest.approx(0.5, abs=1e-5)


def test_equation_2_17_stopping_threshold():
    probs = np.array([0.9, 0.4, 0.2, 0.6])
    surrogate = SurrogateModel(backend="dummy", model=_FakeSurrogate(probs))
    x_adv = torch.randn(4, 4)
    asr = attack_success_rate(x_adv, surrogate, threshold=0.5)
    # samples below 0.5: 0.4 and 0.2 -> 2/4 = 0.5
    assert asr == 0.5
