"""Shared surrogate model training for explainers.

Pure computation module — no FastAPI, no storage, no Pydantic.
All parameters are primitive types (numpy arrays, strings, etc).
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_surrogate(
    inputs: np.ndarray,
    outputs: np.ndarray,
    *,
    mode: str | None = None,
) -> BaseEstimator:
    """Train a surrogate model on stored organic observations.

    Args:
        inputs: 2-D training data (rows: observations, columns: features).
        outputs: 1-D target values.
        mode: Explicit task mode ("regression" or "classification").
            If None, infers from output dtype: floating-point → regression,
            otherwise → classification. Always use explicit mode for
            float-coded classification or integer-valued regression.

    Returns:
        Fitted sklearn estimator (RandomForestRegressor or RandomForestClassifier).

    """
    if mode is None:
        # Fallback inference: use dtype
        mode = (
            "regression"
            if np.issubdtype(outputs.dtype, np.floating)
            else "classification"
        )

    if mode == "regression":
        surrogate: BaseEstimator = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
    else:
        surrogate = RandomForestClassifier(n_estimators=100, random_state=42)
    return surrogate.fit(inputs, outputs)
