"""One closed-loop co-evolution cycle (Equations 2.14, 2.17).

A cycle executes:

1. Train/update surrogate on current data.
2. Compute SHAP ranking on the surrogate and build/update the feature mask.
3. Train the WGAN-GP generator for a number of inner steps.
4. Generate adversarial traffic X_adv = X_orig + (epsilon * M) (Eq. 2.16).
5. Evaluate S(X_adv) = P(y=malicious | X_adv) (Eq. 2.11).
6. Stop inner loop when S(X_adv) < delta (Eq. 2.17) or max iterations reached.

Equation 2.14 (parameter update) is handled by PyTorch optimizers inside
``models.wgan.WGAN_GP``; Adam is the concrete instantiation mentioned in the thesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch

from shapwgan_ids.loop import budget as budget_mod
from shapwgan_ids.loop.feedback import (
    attack_success_rate,
    combined_generator_loss,
    evasion_loss,
)
from shapwgan_ids.models.surrogate import SurrogateModel, train_surrogate
from shapwgan_ids.models.wgan import WGAN_GP
from shapwgan_ids.shap.mask import FeatureMask
from shapwgan_ids.shap.ranking import aggregate_shap

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    cycle: int
    surrogate_path: Path | None
    mask_path: Path | None
    generator_path: Path | None
    mean_evasion_loss: float
    attack_success_rate: float
    mean_confidence: float
    stopped_by: str
    metrics_history: list[dict[str, float]] = field(default_factory=list)


class LoopCycle:
    """Single cycle of the adaptive SHAP-WGAN closed-loop feedback."""

    def __init__(
        self,
        wgan: WGAN_GP,
        surrogate_backend: str = "xgboost",
        surrogate_params: dict[str, Any] | None = None,
        shap_explainer: str = "tree",
        background_samples: int = 200,
        alpha: float = 1.0,
        confidence_threshold: float = 0.5,
        max_inner_steps: int = 1000,
        l2_budget_ratio: float | None = None,
        linf_budget_ratio: float | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.wgan = wgan
        self.surrogate_backend = surrogate_backend
        self.surrogate_params = surrogate_params or {}
        self.shap_explainer = shap_explainer
        self.background_samples = background_samples
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.max_inner_steps = max_inner_steps
        self.l2_budget_ratio = l2_budget_ratio
        self.linf_budget_ratio = linf_budget_ratio
        self.device = device or wgan.device
        self.surrogate: SurrogateModel | None = None
        self.mask: FeatureMask | None = None

    def update_surrogate(self, x_train: np.ndarray, y_train: np.ndarray) -> SurrogateModel:
        """Step 1: retrain or fit the black-box surrogate."""
        self.surrogate = train_surrogate(
            x_train,
            y_train,
            backend=self.surrogate_backend,
            params=self.surrogate_params,
        )
        return self.surrogate

    def update_mask(self, x_background: np.ndarray, feature_names: list[str] | tuple[str, ...]) -> FeatureMask:
        """Step 2: recompute SHAP ranking and rebuild the binary feature mask."""
        if self.surrogate is None:
            raise RuntimeError("update_mask called before update_surrogate")

        # Subsample background if needed
        if x_background.shape[0] > self.background_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(x_background.shape[0], size=self.background_samples, replace=False)
            bg = x_background[idx]
        else:
            bg = x_background

        explainer = shap.TreeExplainer(self.surrogate.model) if self.shap_explainer == "tree" else shap.KernelExplainer(
            self.surrogate.predict_proba, bg
        )
        shap_values = explainer.shap_values(bg)
        importance = aggregate_shap(shap_values)

        from shapwgan_ids.shap.mask import build_mask

        self.mask = build_mask(importance, feature_names, top_k=30, mode="immutable_topk")
        return self.mask

    def run_inner_loop(
        self,
        x_benign: torch.Tensor,
        x_malicious: torch.Tensor,
    ) -> dict[str, Any]:
        """Step 3-6: train generator until stopping criterion."""
        if self.surrogate is None:
            raise RuntimeError("run_inner_loop called before update_surrogate")

        mask_tensor: torch.Tensor | None = None
        if self.mask is not None:
            mask_tensor = torch.tensor(~self.mask.immutable, dtype=torch.float32, device=self.device)

        history: list[dict[str, float]] = []
        stopped_by = "max_steps"
        mean_conf = 1.0
        asr = 0.0

        for step in range(self.max_inner_steps):
            metrics = self.wgan.train_step(x_benign, x_malicious, mask=mask_tensor)

            # Generate and evaluate (Equation 2.16 + 2.11)
            self.wgan.generator.eval()
            with torch.no_grad():
                eps = self.wgan.generate_perturbation(x_malicious)
                x_adv = self.wgan.apply_mask_additive(x_malicious, eps, mask_tensor) if mask_tensor is not None else x_malicious + eps
                x_adv = budget_mod.apply_budgets(x_malicious, x_adv, self.l2_budget_ratio, self.linf_budget_ratio)

                # Combined loss for monitoring (Equation 2.13)
                fake_score = self.wgan.critic(x_adv).mean()
                w_loss = -fake_score
                _total_loss, fb_metrics = combined_generator_loss(w_loss, x_adv, self.surrogate, alpha=self.alpha)
                mean_conf = float(evasion_loss(x_adv, self.surrogate).cpu())
                asr = attack_success_rate(x_adv, self.surrogate, threshold=self.confidence_threshold)

            metrics.update(fb_metrics)
            metrics["mean_confidence"] = mean_conf
            metrics["attack_success_rate"] = asr
            metrics["step"] = step
            history.append(metrics)

            # Stopping criterion (Equation 2.17): confidence below threshold
            if mean_conf < self.confidence_threshold:
                stopped_by = "confidence_threshold"
                break

            # Additional convergence guard: no improvement in ASR for many steps
            if step >= 20:
                recent = [h["attack_success_rate"] for h in history[-20:]]
                if max(recent) - min(recent) < 0.01:
                    stopped_by = "convergence"
                    break

        return {
            "history": history,
            "stopped_by": stopped_by,
            "mean_evasion_loss": mean_conf,
            "attack_success_rate": asr,
            "mean_confidence": mean_conf,
        }

    def run(
        self,
        cycle_id: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_benign: torch.Tensor,
        x_malicious: torch.Tensor,
        feature_names: list[str] | tuple[str, ...],
        artifact_dir: Path | None = None,
    ) -> CycleResult:
        """Execute a full closed-loop cycle."""
        self.update_surrogate(x_train, y_train)
        self.update_mask(x_train, feature_names)

        inner = self.run_inner_loop(x_benign, x_malicious)

        result = CycleResult(
            cycle=cycle_id,
            surrogate_path=None,
            mask_path=None,
            generator_path=None,
            mean_evasion_loss=inner["mean_evasion_loss"],
            attack_success_rate=inner["attack_success_rate"],
            mean_confidence=inner["mean_confidence"],
            stopped_by=inner["stopped_by"],
            metrics_history=inner["history"],
        )

        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            if self.surrogate is not None:
                result.surrogate_path = self.surrogate.save(artifact_dir / f"surrogate_cycle{cycle_id}.joblib")
            if self.mask is not None:
                result.mask_path = self.mask.save(artifact_dir / f"mask_cycle{cycle_id}.json")
            result.generator_path = self.wgan.save(artifact_dir / f"generator_cycle{cycle_id}.pt")

        return result
