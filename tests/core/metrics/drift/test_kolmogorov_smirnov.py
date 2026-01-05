import numpy as np
import pytest
from scipy.stats import ks_2samp

from src.core.metrics.drift.kolmogorov_smirnov import KolmogorovSmirnov

from . import factory

# ==============================================================================
# Kolmogorov-Smirnov Test - Unified Tests
# ==============================================================================


class TestKSTestUnified:
    """Unified tests for Kolmogorov-Smirnov test using factory functions."""

    # Create test methods using factory functions (avoids Hypothesis executor issues)
    test_identical_distributions_no_drift = factory.make_identical_distributions_test(
        metric_fn=KolmogorovSmirnov.kstest,
        params={"alpha": 0.05},
        statistic_key="statistic",
        required_keys=["drift_detected", "statistic", "p_value", "alpha"],
    )

    test_detects_large_shift = factory.make_detects_large_shift_test(
        metric_fn=KolmogorovSmirnov.kstest,
        params={"alpha": 0.05},
    )

    test_different_sample_sizes = factory.make_different_sample_sizes_test(
        metric_fn=KolmogorovSmirnov.kstest,
        params={"alpha": 0.05},
        statistic_key="statistic",
    )

    test_empty_input_raises_error = factory.make_empty_input_test(
        metric_fn=KolmogorovSmirnov.kstest,
        params={"alpha": 0.05},
    )

    def test_kstest_matches_scipy_for_fixed_example(self):
        """Deterministic regression test comparing to scipy.stats.ks_2samp."""

        reference = np.array([0.1, 0.2, 0.2, 0.5, 0.9])
        current = np.array([0.05, 0.25, 0.3, 0.55, 0.95])

        alpha = 0.05

        # Our implementation under test
        result = KolmogorovSmirnov.kstest(reference, current, alpha=alpha)

        # Ground truth from SciPy
        scipy_statistic, scipy_p_value = ks_2samp(reference, current)

        # Check wiring: statistic and p-value should match SciPy
        assert pytest.approx(result["statistic"]) == scipy_statistic
        assert pytest.approx(result["p_value"]) == scipy_p_value

        # Also verify alpha is wired through unchanged
        assert result["alpha"] == alpha
