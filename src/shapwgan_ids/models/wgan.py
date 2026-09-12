"""WGAN-GP perturbation generator and critic (Equations 2.1-2.5, 2.15-2.16).

The generator is *conditional-additive*: it receives a malicious traffic sample
``x`` and a latent noise vector ``z``, and produces a perturbation ``epsilon``.
The adversarial sample is then constructed as ``x_adv = x + epsilon * M`` where
``M`` is the binary SHAP feature mask (Equation 2.16).  This is the construction
used in the thesis.

The critic (WGAN terminology; also called Discriminator in standard GANs) scores
samples.  Training follows the WGAN-GP formulation (Gulrajani et al., 2017):

* Critic loss (Equation 2.3): L_C = E[D(x_adv)] - E[D(x_benign)] + lambda_gp * GP
* Gradient penalty (Equation 2.4): GP = E[(||grad D(x_hat)||_2 - 1)^2]
* Generator Wasserstein loss (Equation 2.5): L_WGAN = -E[D(x_adv)]

Equation 2.1 (original GAN minimax) and 2.2 (Wasserstein distance) are
implemented conceptually by the WGAN-GP losses above.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class _MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm: str | None = "layernorm",
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm: nn.Module | None = None
        if norm == "layernorm":
            self.norm = nn.LayerNorm(out_dim)
        elif norm == "batchnorm":
            self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        x = nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    """Conditional perturbation generator: epsilon = G(x, z).

    Inputs are concatenated as ``[x, z]``.  The output has the same dimension as
    ``x`` and represents the additive perturbation ``epsilon`` in Equation 2.15.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        output_activation: str = "tanh",
        norm: str | None = "layernorm",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim

        layers: list[nn.Module] = []
        prev = n_features + latent_dim
        for h in hidden_dims:
            layers.append(_MLPBlock(prev, h, norm=norm))
            prev = h
        layers.append(nn.Linear(prev, n_features))
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation is not None and output_activation != "none":
            raise ValueError(f"unknown output_activation {output_activation!r}")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=-1))


class Critic(nn.Module):
    """WGAN Critic (Discriminator): scores whether a sample is real or generated."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int] = (256, 256, 256),
        norm: str | None = "layernorm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers.append(_MLPBlock(prev, h, norm=norm, dropout=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class WGAN_GP:
    """Training wrapper for the conditional-additive WGAN-GP."""

    generator: Generator
    critic: Critic
    device: torch.device
    latent_dim: int
    gp_lambda: float = 10.0
    n_critic: int = 5
    lr: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.9)
    use_amp: bool = False

    def __post_init__(self) -> None:
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=self.betas)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == "cuda" else None

    @classmethod
    def from_config(cls, n_features: int, cfg: dict[str, Any], device: str | torch.device = "auto") -> WGAN_GP:
        g_cfg = cfg["generator"]
        c_cfg = cfg["critic"]
        t_cfg = cfg.get("training", {})
        gen = Generator(
            n_features=n_features,
            latent_dim=int(g_cfg["latent_dim"]),
            hidden_dims=tuple(g_cfg["hidden_dims"]),
            output_activation=g_cfg.get("output_activation", "tanh"),
            norm=g_cfg.get("norm", "layernorm"),
        )
        crt = Critic(
            n_features=n_features,
            hidden_dims=tuple(c_cfg["hidden_dims"]),
            norm=c_cfg.get("norm", "layernorm"),
            dropout=float(c_cfg.get("dropout", 0.0)),
        )
        dev = _device(device)
        return cls(
            generator=gen.to(dev),
            critic=crt.to(dev),
            device=dev,
            latent_dim=int(g_cfg["latent_dim"]),
            gp_lambda=float(t_cfg.get("gp_lambda", 10.0)),
            n_critic=int(t_cfg.get("n_critic", 5)),
            lr=float(t_cfg.get("lr", 1e-4)),
            betas=tuple(t_cfg.get("betas", [0.5, 0.9])),
            use_amp=bool(t_cfg.get("amp", False)),
        )

    def sample_latent(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def generate_perturbation(self, x: torch.Tensor) -> torch.Tensor:
        """Produce epsilon = G(x, z) with fresh noise (Equation 2.15)."""
        self.generator.eval()
        with torch.no_grad():
            z = self.sample_latent(x.shape[0])
            return self.generator(x, z)

    def apply_mask_additive(
        self,
        x: torch.Tensor,
        epsilon: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Equation 2.16: X_adv = X_orig + (epsilon * M) for a binary mask M."""
        return x + epsilon * mask

    def critic_loss(
        self,
        x_benign: torch.Tensor,
        x_adv: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Equation 2.3 + 2.4.

        Returns the scalar loss and a dict of detached components for logging.
        """
        self.opt_c.zero_grad()

        real_score = self.critic(x_benign).mean()
        fake_score = self.critic(x_adv).mean()

        # Interpolate for gradient penalty (Equation 2.4)
        alpha = torch.rand(x_benign.size(0), 1, device=self.device)
        interpolates = alpha * x_benign + (1 - alpha) * x_adv
        interpolates.requires_grad_(True)

        interpolate_score = self.critic(interpolates)
        grads = torch.autograd.grad(
            outputs=interpolate_score,
            inputs=interpolates,
            grad_outputs=torch.ones_like(interpolate_score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = grads.view(grads.size(0), -1).norm(2, dim=1)
        gp = ((grad_norm - 1.0) ** 2).mean()

        loss = fake_score - real_score + self.gp_lambda * gp
        metrics = {
            "critic_loss": float(loss.detach().cpu()),
            "wasserstein_distance": float((real_score - fake_score).detach().cpu()),
            "gradient_penalty": float(gp.detach().cpu()),
            "real_score": float(real_score.detach().cpu()),
            "fake_score": float(fake_score.detach().cpu()),
        }
        return loss, metrics

    def generator_loss(self, x_adv: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Equation 2.5: L_WGAN = -E[D(X_adv)]."""
        self.opt_g.zero_grad()
        score = self.critic(x_adv).mean()
        loss = -score
        return loss, {"generator_loss": float(loss.detach().cpu()), "fake_score": float(score.detach().cpu())}

    def train_step(
        self,
        x_benign: torch.Tensor,
        x_malicious: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """One generator step (preceded by ``n_critic`` critic steps).

        ``mask`` is a binary tensor of shape ``(n_features,)`` with 1 on mutable
        positions and 0 on immutable positions.  If None, all features are mutable.
        """
        x_benign = x_benign.to(self.device)
        x_malicious = x_malicious.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        metrics: dict[str, float] = {}

        # Critic updates
        for _ in range(self.n_critic):
            z = self.sample_latent(x_malicious.size(0))
            self.generator.train()
            epsilon = self.generator(x_malicious, z)
            x_adv = self.apply_mask_additive(x_malicious, epsilon, mask) if mask is not None else x_malicious + epsilon

            c_loss, c_metrics = self.critic_loss(x_benign, x_adv)
            c_loss.backward()
            self.opt_c.step()
            # Average the last critic step into metrics
            metrics = c_metrics

        # Generator update
        z = self.sample_latent(x_malicious.size(0))
        self.generator.train()
        epsilon = self.generator(x_malicious, z)
        x_adv = self.apply_mask_additive(x_malicious, epsilon, mask) if mask is not None else x_malicious + epsilon

        g_loss, g_metrics = self.generator_loss(x_adv)
        g_loss.backward()
        self.opt_g.step()
        metrics.update(g_metrics)
        return metrics

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "critic": self.critic.state_dict(),
                "latent_dim": self.latent_dim,
                "gp_lambda": self.gp_lambda,
                "n_critic": self.n_critic,
                "lr": self.lr,
                "betas": self.betas,
            },
            path,
        )
        return path

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.generator.load_state_dict(checkpoint["generator"])
        self.critic.load_state_dict(checkpoint["critic"])
