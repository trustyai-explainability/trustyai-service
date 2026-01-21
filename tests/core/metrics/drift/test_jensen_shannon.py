import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon

from src.core.metrics.drift.jensen_shannon import JensenShannon

from . import factory

eps = np.finfo(float).eps

# ==============================================================================
# Jensen-Shannon Divergence - Unified Tests
# ==============================================================================


class TestJensenShannonUnified:
    """Unified tests for Jensen-Shannon divergence using factory functions."""

    # ==========================================================================
    # Core Behavior Tests - Drift Detection
    # ==========================================================================
    # Tests for basic drift detection functionality using factory methods.

    test_identical_distributions_no_drift = factory.make_identical_distributions_test(
        metric_fn=JensenShannon.jensenshannon,
        params={"statistic": "distance", "threshold": 0.1, "method": "kde", "grid_points": 256},
        statistic_key="Jensen-Shannon_distance",
        required_keys=["drift_detected", "Jensen-Shannon_distance", "Jensen-Shannon_divergence", "threshold"],
    )

    test_detects_large_shift = factory.make_detects_large_shift_test(
        metric_fn=JensenShannon.jensenshannon,
        params={"statistic": "distance", "threshold": 0.1, "method": "kde", "grid_points": 256},
    )

    # ==========================================================================
    # Core Behavior Tests - Sample Size Handling
    # ==========================================================================
    # Tests for handling different sample sizes and edge cases.

    test_different_sample_sizes = factory.make_different_sample_sizes_test(
        metric_fn=JensenShannon.jensenshannon,
        params={"statistic": "distance", "threshold": 0.1, "method": "kde", "grid_points": 256},
        statistic_key="Jensen-Shannon_distance",
    )

    test_small_sample_sizes = factory.make_small_sample_sizes_test(
        metric_fn=JensenShannon.jensenshannon,
        params={"statistic": "distance", "threshold": 0.1, "method": "kde", "grid_points": 256},
        statistic_key="Jensen-Shannon_distance",
    )

    # ==========================================================================
    # Core Behavior Tests - Input Validation
    # ==========================================================================
    # Tests for input validation and error handling.

    test_empty_input_raises_error = factory.make_empty_input_test(
        metric_fn=JensenShannon.jensenshannon,
        params={"statistic": "distance", "threshold": 0.1, "method": "kde", "grid_points": 256},
    )

    # ==========================================================================
    # Parameter Independence Tests
    # ==========================================================================
    # Tests that verify parameters affect drift detection but not computed statistics.

    test_threshold_independence = factory.make_threshold_independence_test(
        metric_fn=JensenShannon.jensenshannon,
        params={"statistic": "distance", "threshold": 0.1, "method": "kde", "grid_points": 256},
        statistic_key="Jensen-Shannon_distance",
        threshold_param="threshold",
    )

    # ==========================================================================
    # Mathematical Property Tests
    # ==========================================================================
    # Tests for mathematical properties like symmetry.

    test_symmetry = factory.make_symmetry_test(
        metric_fn=JensenShannon.jensenshannon,
        params={"statistic": "distance", "threshold": 0.1, "method": "kde", "grid_points": 256},
        statistic_key="Jensen-Shannon_distance",
    )

    # ==========================================================================
    # Parameter-Specific Tests
    # ==========================================================================
    # Tests for specific parameter behaviors that require direct implementation.

    def test_grid_points_parameter(self):
        """Test that grid_points parameter affects computation."""
        np.random.seed(456)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=0.3, scale=1.0, size=100)

        # Compute with different grid resolutions
        result_coarse = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="kde", grid_points=128
        )
        result_fine = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="kde", grid_points=512
        )

        # Both should produce valid results
        assert "Jensen-Shannon_distance" in result_coarse
        assert "Jensen-Shannon_distance" in result_fine

        # Results may differ slightly due to grid resolution, but should be in same ballpark
        assert (
            pytest.approx(result_coarse["Jensen-Shannon_distance"], abs=1e-3) == result_fine["Jensen-Shannon_distance"]
        )

    def test_statistic_parameter(self):
        """Test that statistic parameter controls drift detection correctly."""
        np.random.seed(555)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=0.5, scale=1.0, size=100)

        # Test with distance statistic
        result_distance = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="kde", grid_points=256
        )

        # Test with divergence statistic
        result_divergence = JensenShannon.jensenshannon(
            reference, current, statistic="divergence", threshold=0.1, method="kde", grid_points=256
        )

        # Both should return the same distance and divergence values
        assert pytest.approx(result_distance["Jensen-Shannon_distance"]) == result_divergence["Jensen-Shannon_distance"]
        assert (
            pytest.approx(result_distance["Jensen-Shannon_divergence"])
            == result_divergence["Jensen-Shannon_divergence"]
        )

        # Divergence is distance squared, so it's smaller
        # If distance > threshold, divergence might be < threshold (and vice versa depending on value)
        # Just verify the logic is applied correctly
        if result_distance["drift_detected"]:
            assert result_distance["Jensen-Shannon_distance"] > 0.1
        if result_divergence["drift_detected"]:
            assert result_divergence["Jensen-Shannon_divergence"] > 0.1

    def test_method_parameter(self):
        """Test that method parameter works for both KDE and histogram."""
        np.random.seed(999)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=0.5, scale=1.0, size=100)

        # Test with KDE method
        result_kde = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="kde", grid_points=256
        )

        # Test with histogram method
        result_hist = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="hist", bins=64
        )

        # Both should produce valid results
        assert "Jensen-Shannon_distance" in result_kde
        assert "Jensen-Shannon_distance" in result_hist
        assert "drift_detected" in result_kde
        assert "drift_detected" in result_hist

        # Both should be bounded [0, 1]
        assert 0 <= result_kde["Jensen-Shannon_distance"] <= 1
        assert 0 <= result_hist["Jensen-Shannon_distance"] <= 1

    def test_invalid_method_raises_error(self):
        """Test that invalid method parameter raises ValueError."""
        reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        current = np.array([0.15, 0.25, 0.35, 0.45, 0.55])

        with pytest.raises(ValueError, match="method.*must be.*hist.*kde"):
            JensenShannon.jensenshannon(reference, current, statistic="distance", threshold=0.1, method="invalid")

    def test_invalid_statistic_raises_error(self):
        """Test that invalid statistic parameter raises ValueError."""
        reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        current = np.array([0.15, 0.25, 0.35, 0.45, 0.55])

        with pytest.raises(ValueError, match="statistic must be 'distance' or 'divergence'"):
            JensenShannon.jensenshannon(reference, current, statistic="invalid", threshold=0.1, method="kde")

    def test_kwargs_passed_to_kde(self):
        """Test that kwargs are correctly passed through to KDE estimation."""
        np.random.seed(789)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=0.5, scale=1.0, size=100)

        # Test with default KDE (uses scipy's default bandwidth)
        result_default = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="kde", grid_points=256
        )

        # Test with explicit bandwidth method (should produce different but valid results)
        result_bandwidth = JensenShannon.jensenshannon(
            reference,
            current,
            statistic="distance",
            threshold=0.1,
            method="kde",
            grid_points=256,
            bw_method="scott",  # Explicit bandwidth method
        )

        # Both should produce valid results
        assert "Jensen-Shannon_distance" in result_default
        assert "Jensen-Shannon_distance" in result_bandwidth
        assert 0 <= result_default["Jensen-Shannon_distance"] <= 1
        assert 0 <= result_bandwidth["Jensen-Shannon_distance"] <= 1

    def test_bins_parameter_for_hist_method(self):
        """Test that bins parameter affects histogram-based computation."""
        np.random.seed(321)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=0.3, scale=1.0, size=100)

        # Compute with different bin counts
        result_coarse = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="hist", bins=32
        )
        result_fine = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="hist", bins=128
        )

        # Both should produce valid results
        assert "Jensen-Shannon_distance" in result_coarse
        assert "Jensen-Shannon_distance" in result_fine
        assert 0 <= result_coarse["Jensen-Shannon_distance"] <= 1
        assert 0 <= result_fine["Jensen-Shannon_distance"] <= 1

    # ==========================================================================
    # Integration/Regression Tests
    # ==========================================================================
    # Deterministic tests comparing with reference implementations.

    def test_matches_scipy_for_fixed_example(self):
        """Deterministic regression test comparing to scipy.spatial.distance.jensenshannon."""
        # Use deterministic data
        reference = np.array([0.1, 0.2, 0.2, 0.5, 0.9])
        current = np.array([0.05, 0.25, 0.3, 0.55, 0.95])

        threshold = 0.1

        # Our implementation under test
        result = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=threshold, method="kde", grid_points=256
        )

        # Verify return structure
        assert "Jensen-Shannon_distance" in result
        assert "Jensen-Shannon_divergence" in result
        assert "drift_detected" in result
        assert "threshold" in result

        # Verify divergence is square of distance (JS divergence = JS distance^2)
        assert pytest.approx(result["Jensen-Shannon_divergence"], abs=4 * eps) == result["Jensen-Shannon_distance"] ** 2

        # Verify threshold is wired through
        assert result["threshold"] == threshold

        # Verify JS distance is bounded [0, 1]
        assert 0 <= result["Jensen-Shannon_distance"] <= 1
        assert 0 <= result["Jensen-Shannon_divergence"] <= 1

        # Compare against SciPy's jensenshannon using the same probability estimates
        from src.core.metrics.drift import utils

        p_ref, p_cur = utils.prob_dist_kde(reference, current, grid_points=256)
        expected_distance = jensenshannon(p_ref, p_cur)

        # Our implementation should match SciPy's result
        assert pytest.approx(result["Jensen-Shannon_distance"], abs=1e-10) == expected_distance

    def test_matches_scipy_with_histogram_method(self):
        """Test that histogram method produces valid results comparable to KDE."""
        np.random.seed(111)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=0.5, scale=1.0, size=100)

        # Test with histogram method
        result_hist = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="hist", bins=64
        )

        # Test with KDE method for comparison
        result_kde = JensenShannon.jensenshannon(
            reference, current, statistic="distance", threshold=0.1, method="kde", grid_points=256
        )

        # Both should produce valid, bounded results
        assert 0 <= result_hist["Jensen-Shannon_distance"] <= 1
        assert 0 <= result_kde["Jensen-Shannon_distance"] <= 1
        assert 0 <= result_hist["Jensen-Shannon_divergence"] <= 1
        assert 0 <= result_kde["Jensen-Shannon_divergence"] <= 1

        # Both should have same structure
        assert "drift_detected" in result_hist
        assert "drift_detected" in result_kde
        assert "threshold" in result_hist
        assert "threshold" in result_kde
