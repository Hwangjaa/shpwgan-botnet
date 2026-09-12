"""Perturbation budgets enforced on top of the SHAP mask (thesis 3.6.3)."""

from __future__ import annotations

import torch


def relative_l2_project(x_orig: torch.Tensor, x_adv: torch.Tensor, ratio: float) -> torch.Tensor:
    """Project perturbation so ||delta||_2 <= ratio * ||x_orig||_2 per sample."""
    delta = x_adv - x_orig
    orig_norm = x_orig.norm(dim=1, keepdim=True)
    max_norm = ratio * orig_norm
    delta_norm = delta.norm(dim=1, keepdim=True)
    scale = torch.clamp(delta_norm / max_norm, min=1.0)
    return x_orig + delta / scale


def relative_linf_project(x_orig: torch.Tensor, x_adv: torch.Tensor, ratio: float) -> torch.Tensor:
    """Project perturbation so |delta_j| <= ratio * |x_orig_j| per feature."""
    delta = x_adv - x_orig
    max_delta = ratio * x_orig.abs()
    return x_orig + torch.clamp(delta, -max_delta, max_delta)


def apply_budgets(
    x_orig: torch.Tensor,
    x_adv: torch.Tensor,
    l2_ratio: float | None = None,
    linf_ratio: float | None = None,
) -> torch.Tensor:
    """Apply L2 and/or L-inf relative budgets sequentially."""
    if l2_ratio is not None:
        x_adv = relative_l2_project(x_orig, x_adv, l2_ratio)
    if linf_ratio is not None:
        x_adv = relative_linf_project(x_orig, x_adv, linf_ratio)
    return x_adv
