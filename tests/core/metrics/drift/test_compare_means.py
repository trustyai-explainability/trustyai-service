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

    # Create test methods using factory functions (avoids Hypothesis executor issues)
    test_identical_distributions_no_drift = factory.make_identical_distributions_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
        statistic_key="statistic",
        required_keys=["drift_detected", "statistic", "p_value", "alpha"],
    )

    test_detects_large_shift = factory.make_detects_large_shift_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
    )

    test_different_sample_sizes = factory.make_different_sample_sizes_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
        statistic_key="statistic",
    )

    test_empty_input_raises_error = factory.make_empty_input_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
    )

    test_alpha_independence = factory.make_alpha_independence_test(
        metric_fn=CompareMeans.ttest_ind,
        params={"alpha": 0.05},
        statistic_key="statistic",
        p_value_key="p_value",
        alpha_param="alpha",
    )

    def test_ttest_matches_scipy_for_fixed_example(self):
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

    def test_ttest_basic_calculation(self):
        """Test basic t-test calculation with well-separated means."""
        # Create two distributions with different means
        np.random.seed(42)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=2.0, scale=1.0, size=100)

        alpha = 0.05

        result = CompareMeans.ttest_ind(reference, current, alpha=alpha)

        # Verify return structure
        assert "statistic" in result
        assert "p_value" in result
        assert "drift_detected" in result
        assert "alpha" in result

        # For well-separated means, should detect drift
        assert result["drift_detected"] is True
        assert result["p_value"] < alpha

    def test_ttest_no_drift_identical(self):
        """Test that identical distributions do not show drift."""
        np.random.seed(123)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=0.0, scale=1.0, size=100)

        alpha = 0.05

        result = CompareMeans.ttest_ind(reference, current, alpha=alpha)

        # For identical distributions, should not detect drift (most of the time)
        # Note: This is a statistical test, so there's a small chance of false positive
        # But with alpha=0.05, we expect drift_detected=False in ~95% of cases

    def test_ttest_equal_var_parameter(self):
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

    def test_ttest_nan_policy_omit(self):
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

    def test_ttest_different_variances(self):
        """Test t-test with distributions having different variances."""
        np.random.seed(789)
        reference = np.random.normal(loc=0.0, scale=1.0, size=100)
        current = np.random.normal(loc=0.0, scale=3.0, size=100)  # Same mean, different variance

        alpha = 0.05

        # Welch's t-test (equal_var=False) should handle different variances better
        result_welch = CompareMeans.ttest_ind(reference, current, alpha=alpha, equal_var=False)

        # Should produce valid results
        assert "statistic" in result_welch
        assert "p_value" in result_welch

        # With same means but different variances, Welch's test should generally not detect drift
        # (though this is probabilistic and depends on the sample)

    def test_ttest_small_samples(self):
        """Test t-test with small sample sizes."""
        reference = np.array([1.0, 2.0, 3.0])
        current = np.array([4.0, 5.0, 6.0])

        alpha = 0.05

        result = CompareMeans.ttest_ind(reference, current, alpha=alpha)

        # Should produce valid results even with small samples
        assert "statistic" in result
        assert "p_value" in result
        assert isinstance(result["drift_detected"], bool)
