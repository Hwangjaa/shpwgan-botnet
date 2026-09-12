"""IDS oracles and the WGAN-GP generator.

Planned modules (roadmap step 3-4), kept out of the repo until they are real:

* ``surrogate.py``  -- black-box approximation of the IDS that provides probabilities
  both for SHAP ranking and for the closed-loop confidence feedback.
* ``ids_xgb.py``    -- XGBoost oracle, trained on clean or loop-cycle data.
* ``ids_cnn.py``    -- 1D-CNN oracle over the 115-dim flow vector.
* ``wgan.py``       -- WGAN-GP generator whose output is projected through
  :class:`~shapwgan_ids.shap.mask.FeatureMask` before being handed to a detector.
"""

from __future__ import annotations

__all__: list[str] = []
