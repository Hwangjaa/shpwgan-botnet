"""Confidence feedback and combined generator loss (Equations 2.11-2.13).

Equation 2.11:  S(X_adv) = P(y = malicious | X_adv)
Equation 2.12:  L_evasion = E[ S(X_adv) ]
Equation 2.13:  L_G = L_WGAN + alpha * L_evasion
"""

from __future__ import annotations

import logging

import torch

from shapwgan_ids.models.surrogate import SurrogateModel

log = logging.getLogger(__name__)


def surrogate_confidence(x_adv: torch.Tensor, surrogate: SurrogateModel) -> torch.Tensor:
    """Equation 2.11: S(X_adv) = P(y = malicious | X_adv).

    Returns a tensor of shape ``(batch,)`` with probabilities in [0, 1].
    """
    with torch.no_grad():
        probs = surrogate.predict_proba(x_adv)
    return torch.as_tensor(probs, dtype=torch.float32, device=x_adv.device)


def evasion_loss(x_adv: torch.Tensor, surrogate: SurrogateModel) -> torch.Tensor:
    """Equation 2.12: L_evasion = E[ S(X_adv) ]."""
    confidence = surrogate_confidence(x_adv, surrogate)
    return confidence.mean()


def combined_generator_loss(
    wasserstein_loss: torch.Tensor,
    x_adv: torch.Tensor,
    surrogate: SurrogateModel,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Equation 2.13: L_G = L_WGAN + alpha * L_evasion.

    The evasion term is detached from the surrogate (black-box), so gradients flow
    only through ``x_adv`` and into the generator parameters.
    """
    l_evasion = evasion_loss(x_adv, surrogate)
    l_total = wasserstein_loss + alpha * l_evasion
    metrics = {
        "generator_total_loss": float(l_total.detach().cpu()),
        "wasserstein_loss": float(wasserstein_loss.detach().cpu()),
        "evasion_loss": float(l_evasion.detach().cpu()),
    }
    return l_total, metrics


def attack_success_rate(x_adv: torch.Tensor, surrogate: SurrogateModel, threshold: float = 0.5) -> float:
    """Fraction of samples whose malicious confidence is below ``threshold``.

    This is the operational interpretation of Equation 2.17: S(X_adv) < delta.
    """
    confidence = surrogate_confidence(x_adv, surrogate)
    return float((confidence < threshold).float().mean().cpu())
