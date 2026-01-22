# pylint: disable=line-too-long
"""
Tests for the streaming Kolmogorov-Smirnov test implementation.

These tests verify the KolmogorovSmirnovStreaming class which implements
the Lall (2015) streaming 2-sample KS algorithm using GK sketches.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from scipy.stats import ks_2samp

from src.core.metrics.drift.kolmogorov_smirnov_streaming import KolmogorovSmirnovStreaming

from . import factory


def _streaming_ks_wrapper(reference_data: np.ndarray, current_data: np.ndarray, **kwargs) -> dict:
    """
    Wrapper function to adapt KolmogorovSmirnovStreaming to the factory interface.

    The factory expects a function signature: metric_fn(reference_data, current_data, **params)
    """
    epsilon = kwargs.pop("epsilon", 0.01)
    alpha = kwargs.pop("alpha", 0.05)

    ks = KolmogorovSmirnovStreaming(epsilon=epsilon)
    ks.insert_reference_batch(reference_data)
    ks.insert_current_batch(current_data)

    return ks.kstest(alpha=alpha)


# ==============================================================================
# Streaming Kolmogorov-Smirnov Test - Unified Tests via Factory
# ==============================================================================


class TestStreamingKSTestUnified:
    """Unified tests for streaming KS test using factory functions."""

    test_identical_distributions_no_drift = factory.make_identical_distributions_test(
        metric_fn=_streaming_ks_wrapper,
        params={"epsilon": 0.01, "alpha": 0.05},
        statistic_key="statistic",
        required_keys=["drift_detected", "statistic", "p_value", "alpha", "n_reference", "n_current", "epsilon"],
    )

    test_detects_large_shift = factory.make_detects_large_shift_test(
        metric_fn=_streaming_ks_wrapper,
        params={"epsilon": 0.01, "alpha": 0.05},
    )

    test_different_sample_sizes = factory.make_different_sample_sizes_test(
        metric_fn=_streaming_ks_wrapper,
        params={"epsilon": 0.01, "alpha": 0.05},
        statistic_key="statistic",
    )

    test_empty_input_raises_error = factory.make_empty_input_test(
        metric_fn=_streaming_ks_wrapper,
        params={"epsilon": 0.01, "alpha": 0.05},
    )


# ==============================================================================
# Streaming KS-specific Tests
# ==============================================================================


class TestKolmogorovSmirnovStreamingBasic:
    """Basic functionality tests for KolmogorovSmirnovStreaming."""

    def test_initialization(self):
        """Test basic initialization."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)
        assert ks.epsilon == 0.01
        assert ks.n_reference == 0
        assert ks.n_current == 0

    def test_initialization_invalid_epsilon(self):
        """Test that invalid epsilon values raise ValueError."""
        with pytest.raises(ValueError, match="epsilon must be in the range"):
            KolmogorovSmirnovStreaming(epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be in the range"):
            KolmogorovSmirnovStreaming(epsilon=1.0)

        with pytest.raises(ValueError, match="epsilon must be in the range"):
            KolmogorovSmirnovStreaming(epsilon=-0.1)

    def test_insert_reference(self):
        """Test inserting values into reference sketch."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        for i in range(100):
            ks.insert_reference(float(i))

        assert ks.n_reference == 100
        assert ks.n_current == 0

    def test_insert_current(self):
        """Test inserting values into current sketch."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        for i in range(100):
            ks.insert_current(float(i))

        assert ks.n_reference == 0
        assert ks.n_current == 100

    def test_insert_batch_reference(self):
        """Test batch insertion into reference sketch."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)
        values = np.arange(100, dtype=float)

        ks.insert_reference_batch(values)

        assert ks.n_reference == 100
        assert ks.n_current == 0

    def test_insert_batch_current(self):
        """Test batch insertion into current sketch."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)
        values = np.arange(100, dtype=float)

        ks.insert_current_batch(values)

        assert ks.n_reference == 0
        assert ks.n_current == 100

    def test_reset_reference(self):
        """Test resetting reference sketch."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)
        ks.insert_reference_batch(np.arange(100, dtype=float))
        assert ks.n_reference == 100

        ks.reset_reference()
        assert ks.n_reference == 0

    def test_reset_current(self):
        """Test resetting current sketch."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)
        ks.insert_current_batch(np.arange(100, dtype=float))
        assert ks.n_current == 100

        ks.reset_current()
        assert ks.n_current == 0

    def test_reset_both(self):
        """Test resetting both sketches."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)
        ks.insert_reference_batch(np.arange(100, dtype=float))
        ks.insert_current_batch(np.arange(100, dtype=float))

        ks.reset()

        assert ks.n_reference == 0
        assert ks.n_current == 0


class TestKolmogorovSmirnovStreamingStatistic:
    """Tests for KS statistic computation."""

    def test_statistic_bounded_zero_one(self):
        """Test that statistic is always in [0, 1]."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        rng = np.random.default_rng(42)
        ref_data = rng.uniform(-100, 100, 500)
        cur_data = rng.uniform(-100, 100, 500)

        ks.insert_reference_batch(ref_data)
        ks.insert_current_batch(cur_data)

        stat = ks.statistic()
        assert 0.0 <= stat <= 1.0


class TestKolmogorovSmirnovStreamingPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        epsilon=st.sampled_from([0.01, 0.05, 0.1]),
        n_samples=st.integers(min_value=100, max_value=500),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_approximate_statistic_bounded_error(self, epsilon: float, n_samples: int, seed: int):
        """Property: Approximate statistic is within bounded error of exact."""
        ks = KolmogorovSmirnovStreaming(epsilon=epsilon)

        rng = np.random.RandomState(seed)
        ref_data = stats.norm(loc=0, scale=1).rvs(size=n_samples, random_state=rng)
        cur_data = stats.norm(loc=0.5, scale=1.2).rvs(size=n_samples, random_state=rng)

        ks.insert_reference_batch(ref_data)
        ks.insert_current_batch(cur_data)

        approx_stat = ks.statistic()
        exact_stat, _ = ks_2samp(ref_data, cur_data)

        # Error should be bounded by 4*epsilon (2*epsilon per CDF)
        # Add small slack for finite sample effects
        assert abs(approx_stat - exact_stat) < 4 * epsilon + 0.1


class TestKolmogorovSmirnovStreamingRegression:
    """Regression tests with fixed data."""

    def test_fixed_example_matches_scipy(self):
        """Deterministic test comparing to scipy.stats.ks_2samp."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.001)

        ref_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        cur_data = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05])

        ks.insert_reference_batch(ref_data)
        ks.insert_current_batch(cur_data)

        approx_stat = ks.statistic()
        exact_stat, _ = ks_2samp(ref_data, cur_data)

        # With very small epsilon, should be very close
        assert abs(approx_stat - exact_stat) < 0.05

    def test_uniform_vs_normal_detects_drift(self):
        """Test that uniform vs normal distribution is detected."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        rng = np.random.default_rng(42)
        ref_data = rng.uniform(0, 1, 500)
        cur_data = rng.normal(0.5, 0.2, 500)

        ks.insert_reference_batch(ref_data)
        ks.insert_current_batch(cur_data)

        result = ks.kstest(alpha=0.05)

        assert result["drift_detected"] is True

    def test_bimodal_vs_unimodal_detects_drift(self):
        """Test that bimodal vs unimodal distribution is detected."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        rng = np.random.default_rng(42)
        ref_data = rng.normal(0, 1, 500)
        cur_data = np.concatenate([rng.normal(-2, 0.5, 250), rng.normal(2, 0.5, 250)])

        ks.insert_reference_batch(ref_data)
        ks.insert_current_batch(cur_data)

        result = ks.kstest(alpha=0.05)

        assert result["drift_detected"] is True

    def test_statistic_with_equal_values(self):
        """Test statistic computation when reference and current have equal values."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        # Add some common values to both distributions
        for i in [1.0, 2.0, 3.0, 4.0, 5.0]:
            ks.insert_reference(i)
            ks.insert_current(i)

        # Statistic should be very small (close to 0) since distributions are identical
        stat = ks.statistic()
        assert 0.0 <= stat <= 0.2  # Allow small error due to sketch approximation

    def test_kstest_empty_sketches(self):
        """Test that kstest raises ValueError when sketches are empty."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        # No data inserted - both sketches empty
        with pytest.raises(ValueError, match="Reference sketch is empty"):
            ks.kstest()

        # Only reference has data
        ks.insert_reference(1.0)
        with pytest.raises(ValueError, match="Current sketch is empty"):
            ks.kstest()

    def test_p_value_empty_sketches(self):
        """Test that p_value raises ValueError when sketches are empty (line 186 coverage)."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        # Test with both sketches empty
        with pytest.raises(ValueError, match="Both sketches must be non-empty"):
            ks.p_value()

        # Test with only reference having data
        ks.insert_reference(1.0)
        with pytest.raises(ValueError, match="Both sketches must be non-empty"):
            ks.p_value()

        # Test with only current having data
        ks2 = KolmogorovSmirnovStreaming(epsilon=0.01)
        ks2.insert_current(1.0)
        with pytest.raises(ValueError, match="Both sketches must be non-empty"):
            ks2.p_value()

    def test_property_accessors(self):
        """Test that property accessors return the correct sketch objects."""
        ks = KolmogorovSmirnovStreaming(epsilon=0.01)

        ks.insert_reference(1.0)
        ks.insert_current(2.0)

        # Access properties
        ref_sketch = ks.reference_sketch
        cur_sketch = ks.current_sketch

        assert ref_sketch.n == 1
        assert cur_sketch.n == 1

        # Verify they're the actual sketch objects
        assert len(ref_sketch) == 1
        assert len(cur_sketch) == 1
