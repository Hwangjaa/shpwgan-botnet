"""Numerical checks that WGAN-GP losses match Equations 2.3-2.5, 2.15-2.16."""

from __future__ import annotations

import pytest
import torch

from shapwgan_ids.models.wgan import WGAN_GP, Critic, Generator


def _make_trainer(n_features: int = 4, latent_dim: int = 2, device: str = "cpu") -> WGAN_GP:
    gen = Generator(n_features=n_features, latent_dim=latent_dim, hidden_dims=(8, 8), output_activation="tanh")
    crt = Critic(n_features=n_features, hidden_dims=(8, 8))
    return WGAN_GP(
        generator=gen,
        critic=crt,
        device=torch.device(device),
        latent_dim=latent_dim,
        gp_lambda=10.0,
        n_critic=1,
        lr=1e-3,
    )


def test_equation_2_15_perturbation_is_generated():
    trainer = _make_trainer()
    x = torch.randn(8, 4)
    eps = trainer.generate_perturbation(x)
    assert eps.shape == x.shape
    # output activation tanh keeps perturbation in (-1, 1) before budget scaling
    assert eps.abs().max() <= 1.0 + 1e-6


def test_equation_2_16_additive_mask():
    trainer = _make_trainer()
    x = torch.randn(4, 4)
    eps = torch.randn(4, 4)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    x_adv = trainer.apply_mask_additive(x, eps, mask)

    # immutable positions keep original value
    torch.testing.assert_close(x_adv[:, 1], x[:, 1])
    torch.testing.assert_close(x_adv[:, 3], x[:, 3])
    # mutable positions are x + epsilon
    torch.testing.assert_close(x_adv[:, 0], x[:, 0] + eps[:, 0])
    torch.testing.assert_close(x_adv[:, 2], x[:, 2] + eps[:, 2])


def test_equation_2_3_critic_loss_components():
    trainer = _make_trainer()
    x_benign = torch.randn(16, 4)
    x_adv = torch.randn(16, 4)

    loss, metrics = trainer.critic_loss(x_benign, x_adv)

    # Manual check: loss = E[D(fake)] - E[D(real)] + lambda*GP
    with torch.no_grad():
        real_score = trainer.critic(x_benign).mean()
        fake_score = trainer.critic(x_adv).mean()
        gp = metrics["gradient_penalty"]
        expected = fake_score - real_score + trainer.gp_lambda * gp
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)
    assert "wasserstein_distance" in metrics


def test_equation_2_4_gradient_penalty_non_negative():
    trainer = _make_trainer()
    x_benign = torch.randn(16, 4)
    x_adv = torch.randn(16, 4)
    _, metrics = trainer.critic_loss(x_benign, x_adv)
    assert metrics["gradient_penalty"] >= 0.0


def test_equation_2_5_generator_loss_is_negative_fake_score():
    trainer = _make_trainer()
    x_adv = torch.randn(16, 4)
    loss, metrics = trainer.generator_loss(x_adv)
    with torch.no_grad():
        expected = -trainer.critic(x_adv).mean()
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)
    assert metrics["generator_loss"] == pytest.approx(float(loss), abs=1e-5)
