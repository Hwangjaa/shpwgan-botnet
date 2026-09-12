"""Evaluation metrics (roadmap step 6).

Planned modules:

* ``realism.py``       -- distributional plausibility of generated flows (feature stats,
  per-family distances) so "realism" is measured, not asserted.
* ``attack_success.py``-- attack success rate / evasion rate against each oracle.
* ``robustness.py``    -- per-cycle robustness curves, clean-vs-generated accuracy drop,
  leave-one-attack-out generalisation tables.

Output of this package feeds the thesis tables/figures; nothing here may invent numbers.
"""

from __future__ import annotations

__all__: list[str] = []
