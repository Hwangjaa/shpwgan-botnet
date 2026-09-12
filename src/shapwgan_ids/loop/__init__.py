"""Closed-loop co-evolution driver (roadmap step 5).

Planned modules:

* ``cycle.py``   -- one cycle: generate -> query detector -> retrain detector -> recompute
  SHAP ranking -> adapt generator. Writes per-cycle JSON to ``artifacts/loop/cycles.json``.
* ``feedback.py``-- turns detector outputs into the confidence/detection-rate signal the
  generator optimises against (see ``loop.feedback`` in configs/loop.yaml).
* ``budget.py``  -- perturbation budgets (L2/linf ratios) enforced on top of the mask.
"""

from __future__ import annotations

__all__: list[str] = []
