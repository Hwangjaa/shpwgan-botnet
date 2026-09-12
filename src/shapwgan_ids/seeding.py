"""Reproducibility helpers. Call :func:`set_seed` once at the start of every entry point."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed python/numpy/torch and (optionally) force deterministic kernels.

    Deterministic mode disables cudnn autotuning and can slow GPU training down;
    keep it on for anything whose numbers end up in the thesis.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional for data-only commands
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def resolve_device(preference: str = "auto") -> str:
    """Map a config value ('auto' | 'cpu' | 'cuda') to a usable torch device string."""
    try:
        import torch
    except ImportError:
        return "cpu"
    want_cuda = preference in ("auto", "cuda", "gpu", "cuda:0")
    if want_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"
