"""Closed-loop co-evolution driver (Equations 2.11-2.17).

* ``feedback.py`` -- turns detector outputs into the confidence/detection-rate
  signal the generator optimises against (Equations 2.12-2.13).
* ``cycle.py``   -- one cycle: generate -> query detector -> retrain detector ->
  recompute SHAP ranking -> adapt generator (Equations 2.14, 2.17).
* ``budget.py``  -- perturbation budgets (L2/linf ratios) enforced on top of the mask.
"""

from __future__ import annotations

__all__: list[str] = []
