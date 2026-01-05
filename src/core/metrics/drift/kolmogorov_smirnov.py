"""
Kolmogorov-Smirnov test for detecting distribution drift.

The KS test is a non-parametric test that compares two distributions
by measuring the maximum distance between their cumulative distribution functions.

This module provides the exact KS test. For approximate/streaming versions,
see the approx_ks_test module.
"""

from typing import Dict

import numpy as np
from scipy.stats import ks_2samp


class KolmogorovSmirnov:
    """
    Kolmogorov-Smirnov test for detecting distribution drift.

    The KS test is a non-parametric test that compares two distributions
    by measuring the maximum distance between their cumulative distribution functions.
    This class provides the exact two-sample KS test using `scipy.stats.ks_2samp`.
    """

    @staticmethod
    def kstest(reference_data: np.ndarray, current_data: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
        """
        Calculate exact KS test statistic and p-value for drift detection.

        Uses `scipy.stats.ks_2samp` for exact two-sample Kolmogorov-Smirnov test.

        :param reference_data: Reference distribution data (baseline)
        :param current_data: Current distribution data to compare
        :param alpha: Significance level for hypothesis testing (default: 0.05)
        :return: Dictionary containing statistic, p_value, and drift_detected boolean
        """
        if len(reference_data) == 0 or len(current_data) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Perform two-sample KS test
        statistic, p_value = ks_2samp(reference_data, current_data)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < alpha),
            "alpha": alpha,
        }
