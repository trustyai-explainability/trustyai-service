"""Unit tests for core KernelSHAP computation functions.

Tests ``compute_shap_values`` and ``compute_confidence_intervals`` against a
synthetic linear dataset where Shapley properties are analytically verifiable.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

pytest.importorskip("shap", reason="shap extra not installed")

from tests.core.explainers.local import factory as core_factory
from trustyai_service.core.explainers.local.shap import (
    _get_predict_fn,
    build_surrogate,
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


class TestBuildSurrogate:
    """Tests for ``build_surrogate`` — explicit mode overrides dtype inference."""

    def test_explicit_regression_mode_with_integer_targets(self) -> None:
        """mode='regression' selects RandomForestRegressor even for integer y_train."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3)).astype(np.float64)
        y = rng.integers(0, 10, size=50)  # integer dtype — dtype inference → classifier
        surrogate = build_surrogate(X, y, mode="regression")
        assert isinstance(surrogate, RandomForestRegressor)

    def test_explicit_classification_mode_with_float_targets(self) -> None:
        """mode='classification' selects RandomForestClassifier even for float y_train."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3)).astype(np.float64)
        y = rng.choice([0.0, 1.0], size=50)  # float dtype — dtype inference → regressor
        surrogate = build_surrogate(X, y, mode="classification")
        assert isinstance(surrogate, RandomForestClassifier)


class TestGetPredictFn:
    """Tests for ``_get_predict_fn`` — the predict-function selector."""

    def test_identity_link_returns_hard_predictions(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Identity link returns continuous surrogate predictions (not probabilities)."""
        X, y = synthetic_data
        surrogate = build_surrogate(X, y)
        fn = _get_predict_fn(surrogate, "identity")
        preds = fn(X[:3])
        # Regressor predictions are continuous, not in {0, 1}
        assert preds.shape == (3,)
        assert not all(p in {0.0, 1.0} for p in preds)

    def test_logit_link_regressor_returns_predict_output(
        self, synthetic_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Logit link with regressor returns same output as predict (caller validates range)."""
        X, y = synthetic_data
        surrogate = build_surrogate(X, y)  # float y → RandomForestRegressor
        fn = _get_predict_fn(surrogate, "logit")
        np.testing.assert_array_equal(fn(X[:3]), surrogate.predict(X[:3]))

    def test_logit_link_classifier_uses_predict_proba(self) -> None:
        """Logit link with classifier surrogate uses predict_proba[:, 1]."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        y = (X[:, 0] > 0).astype(int)  # binary labels → RandomForestClassifier
        surrogate = build_surrogate(X, y)
        assert isinstance(surrogate, RandomForestClassifier)
        fn = _get_predict_fn(surrogate, "logit")
        # fn should return probabilities in [0, 1], not hard labels {0, 1}
        preds = fn(X[:5])
        assert preds.shape == (5,)
        assert all(0.0 <= p <= 1.0 for p in preds)
        assert not all(p in {0.0, 1.0} for p in preds), (
            "Expected probabilities, not hard labels"
        )

    def test_logit_classifier_shap_values_nontrivial(self) -> None:
        """SHAP with logit+classifier returns non-zero attributions."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 3)).astype(np.float64)
        y = (X[:, 0] > 0).astype(int)
        surrogate = build_surrogate(X, y)
        vals = compute_shap_values(
            X[0], X, surrogate, n_samples=30, link="logit", l1_reg="bic"
        )
        assert len(vals) == 3
        assert np.any(np.abs(vals) > 1e-6), "Expected non-trivial SHAP values"


class TestNoneRegularizer:
    """Verify RegularizerType.NONE (l1_reg=0.0) does not zero all attributions."""

    def test_none_regularizer_returns_nontrivial_values(
        self, synthetic_data: tuple[np.ndarray, np.ndarray], ridge_surrogate: Ridge
    ) -> None:
        """l1_reg=0.0 disables L1 feature selection; should still return non-zero values."""
        X, _ = synthetic_data
        vals = compute_shap_values(
            X[0], X, ridge_surrogate, n_samples=_N_SAMPLES, link=_LINK, l1_reg=0.0
        )
        assert len(vals) == X.shape[1]
        assert np.any(np.abs(vals) > 1e-6), (
            "l1_reg=0.0 should not zero all attributions"
        )

    def test_none_regularizer_ci_returns_nontrivial_bounds(
        self, synthetic_data: tuple[np.ndarray, np.ndarray], ridge_surrogate: Ridge
    ) -> None:
        """l1_reg=0.0 with CI computation returns non-None, non-zero bounds."""
        X, _ = synthetic_data
        lower, upper = compute_confidence_intervals(
            X[0],
            X,
            ridge_surrogate,
            confidence=0.90,
            n_samples=_N_SAMPLES,
            link=_LINK,
            l1_reg=0.0,
            n_bootstrap=5,
            seed=0,
        )
        assert lower is not None
        assert upper is not None
        assert np.any(np.abs(lower) > 1e-6) or np.any(np.abs(upper) > 1e-6), (
            "l1_reg=0.0 CI should produce non-zero bounds"
        )


class TestSHAPCIContract:
    """CI contract tests generated by core factory (avoids duplication with LIME)."""

    def _build_args(self) -> tuple[tuple, dict]:
        """Return (args, kwargs) for compute_confidence_intervals."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 5)).astype(np.float64)
        y = (3.0 * X[:, 0]).astype(np.float64)
        model = Ridge(alpha=0.01)
        model.fit(X, y)
        args = (X[0], X, model)
        kwargs = {
            "confidence": 0.90,
            "n_samples": _N_SAMPLES,
            "link": _LINK,
            "l1_reg": _L1_REG,
            "n_bootstrap": 5,
            "seed": 0,
        }
        return args, kwargs

    test_ci_disabled_at_confidence_one = core_factory.make_ci_disabled_at_one_test(
        ci_fn=compute_confidence_intervals,
        build_args=lambda: TestSHAPCIContract()._build_args(),
    )

    test_ci_bounds_valid = core_factory.make_ci_bounds_valid_test(
        ci_fn=compute_confidence_intervals,
        build_args=lambda: TestSHAPCIContract()._build_args(),
        confidence=0.90,
    )
