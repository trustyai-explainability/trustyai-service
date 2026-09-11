"""Unit tests for surrogate model training."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from trustyai_service.core.explainers.surrogate import build_surrogate


class TestBuildSurrogate:
    """Tests for build_surrogate() function."""

    def test_regression_detection(self) -> None:
        """Regression output (float dtype) → RandomForestRegressor."""
        x_train = np.random.RandomState(42).randn(100, 5).astype(np.float64)
        y_train = np.random.RandomState(42).randn(100).astype(np.float64)

        surrogate = build_surrogate(x_train, y_train)

        assert isinstance(surrogate, RandomForestRegressor)
        assert surrogate.n_estimators == 100

    def test_classification_detection(self) -> None:
        """Non-float output → RandomForestClassifier."""
        x_train = np.random.RandomState(42).randn(100, 5).astype(np.float64)
        y_train = np.array([0, 1, 0, 1] * 25, dtype=int)

        surrogate = build_surrogate(x_train, y_train)

        assert isinstance(surrogate, RandomForestClassifier)
        assert surrogate.n_estimators == 100

    def test_surrogate_is_fitted(self) -> None:
        """Surrogate returns fitted estimator."""
        x_train = np.random.RandomState(42).randn(50, 3).astype(np.float64)
        y_train = np.random.RandomState(42).randn(50).astype(np.float64)

        surrogate = build_surrogate(x_train, y_train)

        # Verify it can make predictions
        predictions = surrogate.predict(x_train[:1])
        assert len(predictions) == 1
        assert isinstance(predictions[0], (float, np.floating))
