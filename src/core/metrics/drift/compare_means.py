"""
Mean shift detection for distribution drift.

Detects drift by comparing means between reference and current distributions
using independent two-sample t-tests (Student's t-test or Welch's t-test).
"""

from typing import Literal

import numpy as np
from scipy import stats

# Type alias for validated nan_policy values
NanPolicy = Literal["propagate", "raise", "omit"]

# Default parameter values for t-test
DEFAULT_ALPHA = 0.05  # Default significance level for t-test
DEFAULT_EQUAL_VAR = False  # Use Welch's t-test by default (does not assume equal variances)
DEFAULT_NAN_POLICY: NanPolicy = "omit"  # Omit NaN values by default


class CompareMeans:
    """
    Detect if the mean of the distribution has changed.

    Detects drift by comparing means between reference and current distributions
    using independent two-sample t-tests. By default, uses Welch's t-test
    (equal_var=False), which does not assume equal population variances.
    """

    @staticmethod
    def ttest_ind(
        reference_data: np.ndarray,
        current_data: np.ndarray,
        alpha: float = DEFAULT_ALPHA,
        *,
        equal_var: bool = DEFAULT_EQUAL_VAR,
        nan_policy: NanPolicy = DEFAULT_NAN_POLICY,
    ) -> dict[str, float | bool]:
        """
        Perform a t-test statistic and p-value for drift detection.

        Uses `scipy.stats.ttest_ind` for independent two-sample t-test.
        Supports both regular numpy arrays and masked arrays.

        :param reference_data: Reference distribution data (baseline) - numpy array
        :param current_data: Current distribution data to compare - numpy array
        :param alpha: Significance level for hypothesis testing (default: DEFAULT_ALPHA)
        :param equal_var: If True, use Student's t-test (assumes equal variances).
            If False (default), use Welch's t-test.
        :param nan_policy: How to handle NaN values: "propagate", "raise", or "omit" (default)
        :return: Dictionary containing statistic, p_value, and drift_detected boolean
        """
        if len(reference_data) == 0 or len(current_data) == 0:
            raise ValueError("Input arrays cannot be empty")

        statistic, p_value = stats.ttest_ind(reference_data, current_data, equal_var=equal_var, nan_policy=nan_policy)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": alpha,
            "drift_detected": bool(p_value < alpha),
        }
