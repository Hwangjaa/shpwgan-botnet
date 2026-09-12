"""Evaluation metrics (thesis 3.6.6 + scenario definition).

* ``realism.py``       -- distributional plausibility of generated flows.
* ``attack_success.py``-- evasion success rate / detection-rate metrics.
* ``robustness.py``    -- per-cycle robustness curves and comparison tables.
"""

from __future__ import annotations

from .attack_success import evaluate_oracle, evasion_success_rate

__all__ = ["evaluate_oracle", "evasion_success_rate"]
