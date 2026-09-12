"""IDS oracles, surrogate classifier, and the WGAN-GP generator.

* ``surrogate.py``  -- black-box approximation of the IDS that provides probabilities
  both for SHAP ranking and for the closed-loop confidence feedback.
* ``ids_xgb.py``    -- XGBoost oracle, trained on clean or loop-cycle data.
* ``ids_cnn.py``    -- 1D-CNN oracle over the 115-dim flow vector.
* ``wgan.py``       -- WGAN-GP generator whose output is projected through
  :class:`~shapwgan_ids.shap.mask.FeatureMask` before being handed to a detector.
"""

from __future__ import annotations

from .ids_cnn import CNNIDS, train_ids_cnn
from .ids_xgb import train_ids_xgb
from .surrogate import SurrogateModel, train_surrogate
from .wgan import WGAN_GP, Critic, Generator

__all__ = [
    "CNNIDS",
    "WGAN_GP",
    "Critic",
    "Generator",
    "SurrogateModel",
    "train_ids_cnn",
    "train_ids_xgb",
    "train_surrogate",
]
