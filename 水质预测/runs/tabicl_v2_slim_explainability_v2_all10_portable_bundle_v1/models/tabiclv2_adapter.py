"""Small adapter for the isolated TabICLv2 pilot.

The adapter intentionally exposes only fit/predict and keeps the official
TabICLv2 defaults. It is not imported by the formal full_station pipeline.
"""

from __future__ import annotations

from typing import Any
import os

import numpy as np


class TabICLv2Regressor:
    """Thin wrapper around the official TabICL regression estimator."""

    def __init__(self, **kwargs: Any) -> None:
        from tabicl import TabICLRegressor

        defaults = {
            "device": "cuda",
            "use_amp": "auto",
            "offload_mode": "auto",
            "random_state": 42,
            "verbose": False,
        }
        checkpoint = os.environ.get("TABICL_PRETRAINED_CHECKPOINT")
        if checkpoint:
            defaults["model_path"] = checkpoint
            defaults["allow_auto_download"] = False
        defaults.update(kwargs)
        self.model = TabICLRegressor(**defaults)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TabICLv2Regressor":
        self.model.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train, dtype=np.float32))
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(np.asarray(x_test, dtype=np.float32))
        return np.asarray(prediction, dtype=float).reshape(-1)
