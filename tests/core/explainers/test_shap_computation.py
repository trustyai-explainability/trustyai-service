"""Unit tests for core KernelSHAP computation functions.

Tests ``compute_shap_values`` and ``compute_confidence_intervals`` against a
synthetic linear dataset where Shapley properties are analytically verifiable.
"""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

pytest.importorskip("shap", reason="shap extra not installed")

from trustyai_service.core.explainers.shap import (
    compute_confidence_intervals,
    compute_shap_values,
)

_LINK = "identity"
_L1_REG = "bic"
_N_SAMPLES = 50


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Return 50 rows of a 5-feature linear dataset.

    Feature 0: y = 3*x0; features 1-4 carry zero weight.
    """
    rng = np.random.default_rng(0)
    n, d = 50, 5
    X = rng.standard_normal((n, d))
    y = (3.0 * X[:, 0]).astype(np.float64)
    return X.astype(np.float64), y


@pytest.fixture(scope="module")
def ridge_surrogate(synthetic_data: tuple[np.ndarray, np.ndarray]) -> Ridge:
    """Fit a Ridge surrogate on the synthetic data."""
    X, y = synthetic_data
    model = Ridge(alpha=0.01)
    model.fit(X, y)
    return model


class TestComputeSHAPValues:
    """Tests for ``compute_shap_values``."""

    def test_local_accuracy(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """SHAP values sum to f(x) - E[f(X)] within tolerance."""
        X, _ = synthetic_data
        instance = X[0]

        shap_vals = compute_shap_values(
            instance,
            X,
            ridge_surrogate,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
        )

        expected_diff = ridge_surrogate.predict(instance.reshape(1, -1))[0] - float(
            np.mean(ridge_surrogate.predict(X))
        )
        assert abs(float(np.sum(shap_vals)) - expected_diff) < 0.05

    def test_zero_contribution_feature_near_zero(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """Features with no model contribution receive near-zero SHAP values.

        Features 1-4 have zero weight in y = 3*x0; their SHAP values should
        be small relative to feature 0.
        """
        X, _ = synthetic_data
        instance = X[0]

        shap_vals = compute_shap_values(
            instance,
            X,
            ridge_surrogate,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
        )

        dominant = abs(float(shap_vals[0]))
        for i in range(1, 5):
            assert abs(float(shap_vals[i])) < dominant

    def test_returns_one_value_per_feature(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """Output length matches number of input features."""
        X, _ = synthetic_data

        shap_vals = compute_shap_values(
            X[0],
            X,
            ridge_surrogate,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
        )

        assert len(shap_vals) == X.shape[1]


class TestComputeConfidenceIntervals:
    """Tests for ``compute_confidence_intervals``."""

    def test_confidence_one_returns_none(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """confidence=1.0 sentinel disables CI computation and returns (None, None)."""
        X, _ = synthetic_data

        lower, upper = compute_confidence_intervals(
            X[0],
            X,
            ridge_surrogate,
            confidence=1.0,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
            n_bootstrap=5,
        )

        assert lower is None
        assert upper is None

    def test_confidence_just_below_one_computes_intervals(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """confidence=0.9999 (just below sentinel) still computes non-None intervals."""
        X, _ = synthetic_data

        lower, upper = compute_confidence_intervals(
            X[0],
            X,
            ridge_surrogate,
            confidence=0.9999,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
            n_bootstrap=5,
            seed=0,
        )

        assert lower is not None
        assert upper is not None

    def test_lower_le_upper(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """Lower bound does not exceed upper bound for any feature."""
        X, _ = synthetic_data

        lower, upper = compute_confidence_intervals(
            X[0],
            X,
            ridge_surrogate,
            confidence=0.95,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
            n_bootstrap=10,
            seed=1,
        )

        assert lower is not None
        assert upper is not None
        assert np.all(lower <= upper)

    def test_bounds_bracket_point_estimate(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """Point SHAP estimate falls within bootstrap CI for the dominant feature."""
        X, _ = synthetic_data
        instance = X[0]

        shap_vals = compute_shap_values(
            instance,
            X,
            ridge_surrogate,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
        )
        lower, upper = compute_confidence_intervals(
            instance,
            X,
            ridge_surrogate,
            confidence=0.80,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
            n_bootstrap=20,
            seed=2,
        )

        assert lower is not None
        assert upper is not None
        assert float(lower[0]) <= float(shap_vals[0]) <= float(upper[0])

    def test_output_shape_matches_features(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """CI arrays have the same length as the number of input features."""
        X, _ = synthetic_data

        lower, upper = compute_confidence_intervals(
            X[0],
            X,
            ridge_surrogate,
            confidence=0.95,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
            n_bootstrap=5,
            seed=3,
        )

        assert lower is not None
        assert upper is not None
        assert lower.shape == (X.shape[1],)
        assert upper.shape == (X.shape[1],)

    def test_degenerate_bootstrap_sample_skipped(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray],
        ridge_surrogate: Ridge,
    ) -> None:
        """Bootstrap iterations with degenerate samples are skipped gracefully."""
        X, _ = synthetic_data
        instance = X[0]
        tiny_background = X[:2]

        lower, upper = compute_confidence_intervals(
            instance,
            tiny_background,
            ridge_surrogate,
            confidence=0.95,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=_L1_REG,
            n_bootstrap=3,
            seed=42,
        )

        assert lower is not None
        assert upper is not None
