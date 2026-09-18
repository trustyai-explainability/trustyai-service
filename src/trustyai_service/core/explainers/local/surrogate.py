"""Explicit, opt-in sklearn surrogate construction."""

from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_surrogate(
    inputs: np.ndarray,
    outputs: np.ndarray,
    task: Literal["CLASSIFICATION", "REGRESSION"],
) -> RandomForestClassifier | RandomForestRegressor:
    """Fit a surrogate only when the caller explicitly selected SURROGATE."""
    if task == "CLASSIFICATION":
        estimator: RandomForestClassifier | RandomForestRegressor = (
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
    elif task == "REGRESSION":
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        message = "Surrogate task must be CLASSIFICATION or REGRESSION"
        raise ValueError(message)
    return estimator.fit(inputs, outputs)
