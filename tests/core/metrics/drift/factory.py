"""
Factory functions to generate unified tests for drift metrics.

This module provides comprehensive common tests for all drift metrics using
factory functions to avoid repeating testing situations. Each metric is tested
against shared behavioral properties while maintaining metric-specific configurations.
"""

from typing import Any, Callable, Dict

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

# ============================================================================
# Factory Functions - Create test methods with proper scoping
# ============================================================================


def make_identical_distributions_test(
    metric_fn: Callable,
    params: Dict[str, Any],
    statistic_key: str,
    required_keys: list[str],
):
    """
    Create a test function for identical distributions.

    Tests that metrics correctly handle identical distributions (Type I error control).
    Each factory call creates a unique test method instance.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :param statistic_key: Key for the metric's statistic in the result dict
    :param required_keys: List of required keys in the result dict
    :return: Test function bound to the metric configuration
    """

    @given(
        n_samples=st.integers(min_value=20, max_value=200),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_impl(self, n_samples: int, seed: int) -> None:
        """Test metric on identical distributions (Type I error control)."""
        rng = np.random.RandomState(seed)
        reference = stats.norm(loc=0, scale=1).rvs(size=n_samples, random_state=rng)
        current = stats.norm(loc=0, scale=1).rvs(size=n_samples, random_state=rng)

        result = metric_fn(reference, current, **params)

        # Verify return structure
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Verify types
        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result[statistic_key], (int, float))

    return test_impl


def make_detects_large_shift_test(
    metric_fn: Callable,
    params: Dict[str, Any],
):
    """
    Create a test function for detecting large mean shifts.

    Tests that metrics can reliably detect large distribution shifts.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :return: Test function bound to the metric configuration
    """

    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        shift=st.floats(min_value=3.0, max_value=6.0),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=25, deadline=None)
    def test_impl(self, n_samples: int, shift: float, seed: int) -> None:
        """Test metric detects large mean shifts."""
        rng = np.random.RandomState(seed)
        reference = stats.norm(loc=0, scale=1).rvs(size=n_samples, random_state=rng)
        current = stats.norm(loc=shift, scale=1).rvs(size=n_samples, random_state=rng)

        result = metric_fn(reference, current, **params)

        # Large shift should be detected
        assert result["drift_detected"] is True

    return test_impl


def make_different_sample_sizes_test(
    metric_fn: Callable,
    params: Dict[str, Any],
    statistic_key: str,
):
    """
    Create a test function for different sample sizes.

    Tests that metrics handle different sample sizes correctly.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :param statistic_key: Key for the metric's statistic in the result dict
    :return: Test function bound to the metric configuration
    """

    @given(
        n_ref=st.integers(min_value=30, max_value=100),
        n_curr=st.integers(min_value=30, max_value=100),
        seed=st.integers(min_value=0, max_value=5000),
    )
    @settings(max_examples=20, deadline=None)
    def test_impl(self, n_ref: int, n_curr: int, seed: int) -> None:
        """Test metric handles different sample sizes."""
        rng = np.random.RandomState(seed)
        reference = stats.norm(loc=0, scale=1).rvs(size=n_ref, random_state=rng)
        current = stats.norm(loc=0, scale=1).rvs(size=n_curr, random_state=rng)

        result = metric_fn(reference, current, **params)

        # Should produce valid results regardless of sample size differences
        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result[statistic_key], (int, float))

    return test_impl


def make_empty_input_test(
    metric_fn: Callable,
    params: Dict[str, Any],
):
    """
    Create a test function for empty input validation.

    Tests that metrics raise appropriate errors for empty inputs.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :return: Test function bound to the metric configuration
    """

    def test_impl(self) -> None:
        """Test metric raises error for empty inputs."""
        with pytest.raises(ValueError, match="cannot be empty|empty"):
            metric_fn(np.array([]), np.array([1, 2, 3]), **params)

        with pytest.raises(ValueError, match="cannot be empty|empty"):
            metric_fn(np.array([1, 2, 3]), np.array([]), **params)

    return test_impl


# ============================================================================
# Multivariate-specific factory functions
# ============================================================================


def make_multivariate_no_drift_test(
    metric_fn: Callable,
    params: Dict[str, Any],
    statistic_key: str,
):
    """
    Create a test function for multivariate data without drift.

    Tests that metrics handle multi-dimensional data correctly.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :param statistic_key: Key for the metric's statistic in the result dict
    :return: Test function bound to the metric configuration
    """

    @given(
        n_samples=st.integers(min_value=50, max_value=150),
        n_features=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=5000),
    )
    @settings(max_examples=20, deadline=None)
    def test_impl(self, n_samples: int, n_features: int, seed: int) -> None:
        """Test metric handles multi-dimensional data."""
        rng = np.random.RandomState(seed)
        # Sample from multivariate normal with identity covariance
        reference = stats.multivariate_normal(mean=np.zeros(n_features), cov=np.eye(n_features)).rvs(
            size=n_samples, random_state=rng
        )
        current = stats.multivariate_normal(mean=np.zeros(n_features), cov=np.eye(n_features)).rvs(
            size=n_samples, random_state=rng
        )

        result = metric_fn(reference, current, **params)

        # Verify return structure for multivariate case
        assert isinstance(result["drift_detected"], bool)
        assert isinstance(result[statistic_key], (int, float))

    return test_impl


def make_multivariate_detects_shift_test(
    metric_fn: Callable,
    params: Dict[str, Any],
):
    """
    Create a test function for detecting shifts in multivariate data.

    Tests that metrics can detect distribution shifts in multi-dimensional data.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :return: Test function bound to the metric configuration
    """

    @given(
        n_samples=st.integers(min_value=50, max_value=150),
        n_features=st.integers(min_value=2, max_value=5),
        shift=st.floats(min_value=3.0, max_value=5.0),
        seed=st.integers(min_value=0, max_value=5000),
    )
    @settings(max_examples=20, deadline=None)
    def test_impl(self, n_samples: int, n_features: int, shift: float, seed: int) -> None:
        """Test metric detects shifts in multi-dimensional data."""
        rng = np.random.RandomState(seed)
        # Sample from multivariate normal with identity covariance
        reference = stats.multivariate_normal(mean=np.zeros(n_features), cov=np.eye(n_features)).rvs(
            size=n_samples, random_state=rng
        )
        # Shifted distribution (mean shift in all dimensions)
        current = stats.multivariate_normal(mean=np.full(n_features, shift), cov=np.eye(n_features)).rvs(
            size=n_samples, random_state=rng
        )

        result = metric_fn(reference, current, **params)

        # Large shift should be detected
        assert result["drift_detected"] is True

    return test_impl


# ============================================================================
# Symmetric metric factory functions
# ============================================================================


def make_symmetry_test(
    metric_fn: Callable,
    params: Dict[str, Any],
    statistic_key: str,
):
    """
    Create a test function for metric symmetry.

    Tests that symmetric metrics satisfy M(A, B) = M(B, A).
    This applies to symmetric divergence measures like Jensen-Shannon,
    Hellinger distance, etc.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :param statistic_key: Key for the metric's statistic in the result dict
    :return: Test function bound to the metric configuration
    """

    @given(
        n_samples=st.integers(min_value=50, max_value=150),
        loc_shift=st.floats(min_value=0.5, max_value=2.0),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_impl(self, n_samples: int, loc_shift: float, seed: int) -> None:
        """Test metric is symmetric: M(A, B) = M(B, A)."""
        rng = np.random.RandomState(seed)
        data_a = stats.norm(loc=0, scale=1).rvs(size=n_samples, random_state=rng)
        data_b = stats.norm(loc=loc_shift, scale=1).rvs(size=n_samples, random_state=rng)

        # Compute metric in both directions
        result_ab = metric_fn(data_a, data_b, **params)
        result_ba = metric_fn(data_b, data_a, **params)

        # Metric should be symmetric
        assert pytest.approx(result_ab[statistic_key], abs=1e-10) == result_ba[statistic_key]
        assert result_ab["drift_detected"] == result_ba["drift_detected"]

    return test_impl


def make_threshold_independence_test(
    metric_fn: Callable,
    params: Dict[str, Any],
    statistic_key: str,
    threshold_param: str = "threshold",
):
    """
    Create a test function for threshold independence.

    Tests that changing the threshold parameter affects drift detection
    but not the computed metric value itself.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :param statistic_key: Key for the metric's statistic in the result dict
    :param threshold_param: Name of the threshold parameter (default: "threshold")
    :return: Test function bound to the metric configuration
    """

    @given(
        n_samples=st.integers(min_value=50, max_value=150),
        loc_shift=st.floats(min_value=0.3, max_value=1.5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_impl(self, n_samples: int, loc_shift: float, seed: int) -> None:
        """Test that threshold affects drift_detected but not the metric value."""
        rng = np.random.RandomState(seed)
        reference = stats.norm(loc=0, scale=1).rvs(size=n_samples, random_state=rng)
        current = stats.norm(loc=loc_shift, scale=1).rvs(size=n_samples, random_state=rng)

        # Get base threshold from params
        base_params = params.copy()
        base_threshold = base_params.get(threshold_param, 0.1)

        # Test with strict threshold (half of base)
        strict_params = base_params.copy()
        strict_params[threshold_param] = base_threshold * 0.5
        strict_result = metric_fn(reference, current, **strict_params)

        # Test with lenient threshold (double base)
        lenient_params = base_params.copy()
        lenient_params[threshold_param] = base_threshold * 2.0
        lenient_result = metric_fn(reference, current, **lenient_params)

        # Metric value should be identical regardless of threshold
        assert pytest.approx(strict_result[statistic_key]) == lenient_result[statistic_key]

        # Drift detection may differ based on threshold
        # If drift is detected, verify the metric exceeds the threshold
        if strict_result["drift_detected"]:
            assert strict_result[statistic_key] > strict_params[threshold_param]
        if lenient_result["drift_detected"]:
            assert lenient_result[statistic_key] > lenient_params[threshold_param]

    return test_impl


def make_alpha_independence_test(
    metric_fn: Callable,
    params: Dict[str, Any],
    statistic_key: str,
    p_value_key: str = "p_value",
    alpha_param: str = "alpha",
):
    """
    Create a test function for alpha independence (for p-value based tests).

    Tests that changing the alpha parameter affects drift detection
    but not the computed statistic or p-value itself. This is specifically
    for hypothesis tests where drift is detected when p_value < alpha.

    :param metric_fn: The drift metric function to test
    :param params: Parameters to pass to the metric function
    :param statistic_key: Key for the metric's statistic in the result dict
    :param p_value_key: Key for the p-value in the result dict (default: "p_value")
    :param alpha_param: Name of the alpha parameter (default: "alpha")
    :return: Test function bound to the metric configuration
    """

    @given(
        n_samples=st.integers(min_value=50, max_value=150),
        loc_shift=st.floats(min_value=0.3, max_value=1.5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_impl(self, n_samples: int, loc_shift: float, seed: int) -> None:
        """Test that alpha affects drift_detected but not the statistic or p-value."""
        rng = np.random.RandomState(seed)
        reference = stats.norm(loc=0, scale=1).rvs(size=n_samples, random_state=rng)
        current = stats.norm(loc=loc_shift, scale=1).rvs(size=n_samples, random_state=rng)

        # Get base alpha from params
        base_params = params.copy()
        base_alpha = base_params.get(alpha_param, 0.05)

        # Test with strict alpha (half of base)
        strict_params = base_params.copy()
        strict_params[alpha_param] = base_alpha * 0.5
        strict_result = metric_fn(reference, current, **strict_params)

        # Test with lenient alpha (double base)
        lenient_params = base_params.copy()
        lenient_params[alpha_param] = base_alpha * 2.0
        lenient_result = metric_fn(reference, current, **lenient_params)

        # Statistic and p-value should be identical regardless of alpha
        assert pytest.approx(strict_result[statistic_key]) == lenient_result[statistic_key]
        assert pytest.approx(strict_result[p_value_key]) == lenient_result[p_value_key]

        # Drift detection may differ based on alpha
        # For p-value tests: drift detected when p_value < alpha
        if strict_result["drift_detected"]:
            assert strict_result[p_value_key] < strict_params[alpha_param]
        if lenient_result["drift_detected"]:
            assert lenient_result[p_value_key] < lenient_params[alpha_param]

    return test_impl
