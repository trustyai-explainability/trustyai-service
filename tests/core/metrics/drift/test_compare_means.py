import numpy as np
import pytest
from scipy.stats import ttest_ind

from src.core.metrics.drift.compare_means import CompareMeans

from . import factory

# ==============================================================================
# Independent T-Test - Unified Tests
# ==============================================================================


class TestTTestUnified:
    """Unified tests for independent two-sample t-test using factory functions."""

    # ==========================================================================
    # Core Behavior Tests - Drift Detection
    # ==========================================================================
    # Tests for basic drift detection functionality using factory methods.

    test_no_drift_identical = factory.make_no_drift_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
        drift_detected_key="drift_detected",
        p_value_key="p_value",
        alpha_param="alpha",
    )

    test_detects_large_shift = factory.make_detects_large_shift_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
    )

    # ==========================================================================
    # Core Behavior Tests - Sample Size Handling
    # ==========================================================================
    # Tests for handling different sample sizes and edge cases.

    test_different_sample_sizes = factory.make_different_sample_sizes_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
        statistic_key="statistic",
    )

    test_small_sample_sizes = factory.make_small_sample_sizes_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
        statistic_key="statistic",
    )

    # ==========================================================================
    # Core Behavior Tests - Variance Handling
    # ==========================================================================
    # Tests for handling different variance scenarios.

    test_different_variances = factory.make_different_variances_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05, "equal_var": False},
        statistic_key="statistic",
    )

    # ==========================================================================
    # Core Behavior Tests - Input Validation
    # ==========================================================================
    # Tests for input validation and error handling.

    test_empty_input_raises_error = factory.make_empty_input_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
    )

    # ==========================================================================
    # Parameter Independence Tests
    # ==========================================================================
    # Tests that verify parameters affect drift detection but not computed statistics.

    test_alpha_independence = factory.make_alpha_independence_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
        statistic_key="statistic",
        p_value_key="p_value",
        alpha_param="alpha",
    )

    # ==========================================================================
    # Parameter-Specific Tests
    # ==========================================================================
    # Tests for specific parameter behaviors that require direct implementation.

    def test_equal_var_parameter(self):
        """Test that equal_var parameter is correctly passed through."""
        np.random.seed(456)
        reference = np.random.normal(loc=0.0, scale=1.0, size=50)
        current = np.random.normal(loc=0.5, scale=1.0, size=50)

        # Test with equal_var=False (Welch's t-test, default)
        result_welch = CompareMeans.ttest_ind(reference, current, alpha=0.05, equal_var=False)

        # Test with equal_var=True (Student's t-test)
        result_student = CompareMeans.ttest_ind(reference, current, alpha=0.05, equal_var=True)

        # Both should return valid results
        assert "statistic" in result_welch
        assert "statistic" in result_student

        # Compare with scipy
        scipy_welch_stat, scipy_welch_p = ttest_ind(reference, current, equal_var=False)
        scipy_student_stat, scipy_student_p = ttest_ind(reference, current, equal_var=True)

        assert pytest.approx(result_welch["statistic"]) == scipy_welch_stat
        assert pytest.approx(result_welch["p_value"]) == scipy_welch_p
        assert pytest.approx(result_student["statistic"]) == scipy_student_stat
        assert pytest.approx(result_student["p_value"]) == scipy_student_p

    def test_nan_policy_omit(self):
        """Test that NaN values are handled correctly with nan_policy='omit'."""
        reference = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        current = np.array([1.5, 2.5, 3.5, np.nan, 5.5])

        alpha = 0.05

        result = CompareMeans.ttest_ind(reference, current, alpha=alpha, nan_policy="omit")

        # Should produce valid results by omitting NaN values
        assert "statistic" in result
        assert "p_value" in result
        assert not np.isnan(result["statistic"])
        assert not np.isnan(result["p_value"])

    # ==========================================================================
    # Integration/Regression Tests
    # ==========================================================================
    # Deterministic tests comparing with reference implementations.

    def test_matches_scipy_for_fixed_example(self):
        """Deterministic regression test comparing to scipy.stats.ttest_ind."""
        reference = np.array([0.1, 0.2, 0.2, 0.5, 0.9])
        current = np.array([0.05, 0.25, 0.3, 0.55, 0.95])

        alpha = 0.05

        # Our implementation under test
        result = CompareMeans.ttest_ind(reference, current, alpha=alpha)

        # Ground truth from SciPy (Welch's t-test by default)
        scipy_statistic, scipy_p_value = ttest_ind(reference, current, equal_var=False)

        # Check wiring: statistic and p-value should match SciPy
        assert pytest.approx(result["statistic"]) == scipy_statistic
        assert pytest.approx(result["p_value"]) == scipy_p_value

        # Also verify alpha is wired through unchanged
        assert result["alpha"] == alpha
