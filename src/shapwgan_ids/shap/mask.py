"""Feature mutation mask derived from SHAP ranking (thesis 3.6.2).

The mask is the core constraint of the whole method: the Top-K most attack-defining
features are declared *immutable*, so the generator may only perturb the remaining
mutable features. That is what separates this work from unconstrained adversarial
generation -- a perturbation that rewrites the defining statistics of, say, a SYN flood
no longer produces a plausible SYN flood.

Contract enforced here (and unit-tested in tests/test_mask.py):

* :meth:`FeatureMask.apply` never changes an immutable feature, whatever the generator says.
* Mutable features take the generated value as-is; the caller handles budgets/clipping.
* The mask is serialisable so a loop cycle can persist exactly which features were frozen.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ranking import rank_features, top_k_by_cumulative, top_k_names

MODES = ("immutable_topk", "mutable_topk", "none")


@dataclass(frozen=True)
class FeatureMask:
    """Boolean mask over feature positions; ``immutable[i] is True`` means frozen."""

    names: tuple[str, ...]
    immutable: np.ndarray
    mode: str
    top_k: int
    ranked_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.immutable.shape != (len(self.names),):
            raise ValueError(f"immutable mask shape {self.immutable.shape} does not match {len(self.names)} names")
        if self.mode not in MODES:
            raise ValueError(f"unknown mask mode {self.mode!r} (expected one of {MODES})")

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def immutable_idx(self) -> np.ndarray:
        return np.flatnonzero(self.immutable)

    @property
    def mutable_idx(self) -> np.ndarray:
        return np.flatnonzero(~self.immutable)

    @property
    def n_immutable(self) -> int:
        return int(self.immutable.sum())

    @property
    def n_mutable(self) -> int:
        return int((~self.immutable).sum())

    def apply(self, original: np.ndarray, generated: np.ndarray) -> np.ndarray:
        """Blend: generated values on mutable features, original values on immutable ones.

        Useful for ablations; the thesis construction is :meth:`apply_additive`.
        Both inputs are ``(n_samples, n_features)``; a single 1-D vector is accepted and
        returned as 1-D. The result is a new array -- inputs are never mutated in place.
        """
        a = np.asarray(original, dtype=np.float32)
        b = np.asarray(generated, dtype=np.float32)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: original {a.shape} vs generated {b.shape}")
        if a.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {a.shape[-1]}")

        out = b.copy()
        out[..., self.immutable_idx] = a[..., self.immutable_idx]
        return out

    def apply_additive(self, original: np.ndarray, perturbation: np.ndarray) -> np.ndarray:
        """Apply perturbation only on mutable features (Equation 2.16).

        X_adv = X_orig + (epsilon * M) where M_j = 1 for mutable, 0 for immutable.
        Immutable features stay exactly equal to ``original``.
        """
        x_orig = np.asarray(original, dtype=np.float32)
        delta = np.asarray(perturbation, dtype=np.float32)
        if x_orig.shape != delta.shape:
            raise ValueError(f"shape mismatch: original {x_orig.shape} vs perturbation {delta.shape}")
        if x_orig.shape[-1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x_orig.shape[-1]}")

        out = x_orig.copy()
        out[..., self.mutable_idx] = x_orig[..., self.mutable_idx] + delta[..., self.mutable_idx]
        return out

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "names": list(self.names),
            "immutable": [bool(v) for v in self.immutable],
            "ranked_names": list(self.ranked_names),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, payload: dict) -> FeatureMask:
        return cls(
            names=tuple(payload["names"]),
            immutable=np.asarray(payload["immutable"], dtype=bool),
            mode=payload["mode"],
            top_k=int(payload["top_k"]),
            ranked_names=tuple(payload.get("ranked_names", ())),
        )

    @classmethod
    def load(cls, path: Path) -> FeatureMask:
        return cls.from_dict(json.loads(Path(path).read_text()))


def build_mask(
    importance: Sequence[float] | np.ndarray,
    feature_names: Sequence[str],
    top_k: int | None = 30,
    coverage: float | None = None,
    mode: str = "immutable_topk",
) -> FeatureMask:
    """Build a mask from a per-feature importance vector.

    * ``immutable_topk`` -- Top-K features frozen (method default).
    * ``mutable_topk``   -- only Top-K features may move (ablation, inverted budget).
    * ``none``           -- everything mutable (unconstrained baseline).

    If ``coverage`` is given, K is chosen so that C_k >= coverage (Equation 2.9)
    and ``top_k`` becomes a hard ceiling. Both cannot be omitted.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mask mode {mode!r} (expected one of {MODES})")
    importance = np.asarray(importance, dtype=np.float64)
    names = tuple(feature_names)
    if mode == "none":
        return FeatureMask(names=names, immutable=np.zeros(len(names), dtype=bool), mode=mode, top_k=0)

    if coverage is not None:
        k = top_k_by_cumulative(importance, coverage=coverage, max_k=top_k)
    elif top_k is not None:
        k = top_k
    else:
        raise ValueError("build_mask needs either top_k or coverage")

    top = top_k_names(importance, names, k)
    ranked = tuple(name for name, _ in rank_features(importance, names))
    top_set = set(top)
    select = np.array([name in top_set for name in names], dtype=bool)
    immutable = select if mode == "immutable_topk" else ~select
    return FeatureMask(names=names, immutable=immutable, mode=mode, top_k=len(top), ranked_names=ranked)
