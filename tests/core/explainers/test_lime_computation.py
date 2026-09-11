"""Unit tests for LIME core computation functions."""

import numpy as np
import pytest

from trustyai_service.core.explainers.lime import (
    _LIME_AVAILABLE,
    compute_lime_confidence_intervals,
    compute_lime_explanation,
    create_lime_explainer,
)
from trustyai_service.core.explainers.surrogate import build_surrogate

pytestmark = pytest.mark.skipif(not _LIME_AVAILABLE, reason="lime not installed")


class TestCreateLimeExplainer:
    """Tests for create_lime_explainer() function."""

    def test_create_regression_explainer(self) -> None:
        """Create a LIME explainer for regression."""
        training_data = np.random.RandomState(42).randn(100, 5)
        feature_names = [f"feature_{i}" for i in range(5)]

        explainer = create_lime_explainer(
            training_data, feature_names, mode="regression", seed=42
        )

        assert explainer is not None
        assert explainer.mode == "regression"
        assert len(explainer.feature_names) == 5

    def test_create_classification_explainer(self) -> None:
        """Create a LIME explainer for classification."""
        training_data = np.random.RandomState(42).randn(100, 5)
        feature_names = [f"feature_{i}" for i in range(5)]

        explainer = create_lime_explainer(
            training_data, feature_names, mode="classification", seed=42
        )

        assert explainer is not None
        assert explainer.mode == "classification"


class TestComputeLimeExplanation:
    """Tests for compute_lime_explanation() function."""

    def test_explanation_structure(self) -> None:
        """Explanation returns tuple of (weights, r2, local_pred, intercept)."""
        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((100, 5))
        # Linear relationship: y = 3*x0 + noise
        y_train = 3 * x_train[:, 0] + rng.standard_normal(100) * 0.1

        surrogate = build_surrogate(x_train, y_train)
        explainer = create_lime_explainer(
            x_train,
            [f"feature_{i}" for i in range(5)],
            mode="regression",
            seed=42,
        )

        instance = x_train[0]
        weights, r2, local_pred, intercept = compute_lime_explanation(
            explainer,
            instance,
            surrogate.predict,
            num_samples=1000,
            num_features=5,
        )

        assert isinstance(weights, list)
        assert all(isinstance(w, tuple) and len(w) == 2 for w in weights)
        assert isinstance(r2, float)
        assert 0 <= r2 <= 1
        assert isinstance(local_pred, float)
        assert isinstance(intercept, float)

    def test_dominant_feature_ranking(self) -> None:
        """Dominant feature (x0) gets highest absolute weight."""
        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((100, 5))
        y_train = 3 * x_train[:, 0] + rng.standard_normal(100) * 0.1

        surrogate = build_surrogate(x_train, y_train)
        explainer = create_lime_explainer(
            x_train,
            [f"feature_{i}" for i in range(5)],
            mode="regression",
            seed=42,
        )

        instance = x_train[0]
        weights, _, _, _ = compute_lime_explanation(
            explainer,
            instance,
            surrogate.predict,
            num_samples=1000,
            num_features=5,
        )

        # LIME discretizes features, so names become ranges like "-0.01 < feature_0 <= 0.52"
        # Check that feature_0 appears in top weights (due to y=3*x0 relationship)
        top_features = [name for name, _ in weights[:2]]
        assert any("feature_0" in name for name in top_features)


class TestComputeLimeExplanationClassification:
    """Tests for classification mode (exercises predict_proba path)."""

    def test_classification_explanation(self) -> None:
        """Classification mode works with integer targets."""
        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((100, 5))
        # Binary classification
        y_train = np.array([0, 1] * 50, dtype=int)

        surrogate = build_surrogate(x_train, y_train)
        explainer = create_lime_explainer(
            x_train,
            [f"feature_{i}" for i in range(5)],
            mode="classification",
            seed=42,
        )

        instance = x_train[0]
        weights, r2, local_pred, intercept = compute_lime_explanation(
            explainer,
            instance,
            surrogate.predict_proba,  # Classification requires predict_proba
            num_samples=1000,
            num_features=5,
        )

        assert isinstance(weights, list)
        assert len(weights) > 0
        assert 0 <= r2 <= 1
        assert isinstance(local_pred, float)
        assert isinstance(intercept, float)

    def test_classification_confidence_intervals(self) -> None:
        """Classification CI works with stable feature names."""
        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((100, 5))
        y_train = np.array([0, 1] * 50, dtype=int)

        surrogate = build_surrogate(x_train, y_train)

        instance = x_train[0]
        lower, upper = compute_lime_confidence_intervals(
            x_train,
            [f"feature_{i}" for i in range(5)],
            "classification",
            instance,
            surrogate.predict_proba,
            confidence=0.9,
            num_samples=1000,
            num_features=5,
            kernel_width=0.75,
            n_bootstrap=5,
            seed=42,
        )

        assert lower is not None, "Expected lower bounds dict, got None"
        assert upper is not None, "Expected upper bounds dict, got None"
        assert isinstance(lower, dict)
        assert isinstance(upper, dict)
        for feature_name in lower:
            assert lower[feature_name] <= upper[feature_name]


class TestComputeLimeConfidenceIntervals:
    """Tests for compute_lime_confidence_intervals() function."""

    def test_confidence_1_disables_ci(self) -> None:
        """Confidence=1.0 sentinel returns (None, None)."""
        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((100, 5))
        y_train = 3 * x_train[:, 0] + rng.standard_normal(100) * 0.1
        feature_names = [f"feature_{i}" for i in range(5)]

        surrogate = build_surrogate(x_train, y_train)

        instance = x_train[0]
        lower, upper = compute_lime_confidence_intervals(
            x_train,
            feature_names,
            "regression",
            instance,
            surrogate.predict,
            confidence=1.0,
            num_samples=1000,
            num_features=5,
            kernel_width=0.75,
            n_bootstrap=10,
            seed=42,
        )

        assert lower is None
        assert upper is None

    def test_confidence_returns_bounds(self) -> None:
        """Confidence < 1.0 returns lower/upper dicts with valid bounds."""
        rng = np.random.default_rng(42)
        x_train = rng.standard_normal((100, 5))
        y_train = 3 * x_train[:, 0] + rng.standard_normal(100) * 0.1
        feature_names = [f"feature_{i}" for i in range(5)]

        surrogate = build_surrogate(x_train, y_train)

        instance = x_train[0]
        lower, upper = compute_lime_confidence_intervals(
            x_train,
            feature_names,
            "regression",
            instance,
            surrogate.predict,
            confidence=0.9,
            num_samples=1000,
            num_features=5,
            kernel_width=0.75,
            n_bootstrap=10,
            seed=42,
        )

        # CI must be computed when confidence < 1.0
        assert lower is not None, "Expected lower bounds dict, got None"
        assert upper is not None, "Expected upper bounds dict, got None"
        # Bounds are dictionaries keyed by feature name
        assert isinstance(lower, dict)
        assert isinstance(upper, dict)
        # All features in lower should have lower <= upper
        for feature_name in lower:
            assert lower[feature_name] <= upper[feature_name], (
                f"{feature_name}: {lower[feature_name]} > {upper[feature_name]}"
            )
