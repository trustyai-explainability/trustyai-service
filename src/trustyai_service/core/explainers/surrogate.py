"""Shared surrogate model training for explainers.

Pure computation module — no FastAPI, no storage, no Pydantic.
All parameters are primitive types (numpy arrays, strings, etc).
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_surrogate(inputs: np.ndarray, outputs: np.ndarray) -> BaseEstimator:
    """Train a surrogate model on stored organic observations.

    Detects output type: floating-point dtype → RandomForestRegressor,
    otherwise → RandomForestClassifier.

    Args:
        inputs: 2-D training data (rows: observations, columns: features).
        outputs: 1-D target values.

    Returns:
        Fitted sklearn estimator (RandomForestRegressor or RandomForestClassifier).

    """
    if np.issubdtype(outputs.dtype, np.floating):
        surrogate: BaseEstimator = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
    else:
        surrogate = RandomForestClassifier(n_estimators=100, random_state=42)
    return surrogate.fit(inputs, outputs)
