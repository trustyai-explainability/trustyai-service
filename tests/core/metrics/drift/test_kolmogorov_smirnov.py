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
