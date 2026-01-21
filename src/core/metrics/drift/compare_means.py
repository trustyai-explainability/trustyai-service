"""
Mean shift detection for distribution drift.

Detects drift by comparing means between reference and current distributions
using independent two-sample t-tests (Student's t-test or Welch's t-test).
"""

import numpy as np
from scipy import stats

# Default parameter values for t-test
DEFAULT_ALPHA = 0.05  # Default significance level for t-test
DEFAULT_EQUAL_VAR = False  # Use Welch's t-test by default (does not assume equal variances)
DEFAULT_NAN_POLICY = "omit"  # Omit NaN values by default


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
        **kwargs,
    ) -> dict[str, float | bool]:
        """
        Perform a t-test statistic and p-value for drift detection.

        Uses `scipy.stats.ttest_ind` for independent two-sample t-test.
        Supports both regular numpy arrays and masked arrays.

        :param reference_data: Reference distribution data (baseline) - numpy array
        :param current_data: Current distribution data to compare - numpy array
        :param alpha: Significance level for hypothesis testing (default: DEFAULT_ALPHA)
        :param kwargs: Additional keyword arguments to pass to scipy.stats.ttest_ind
                      (e.g., equal_var=False for Welch's t-test, nan_policy='omit')
        :return: Dictionary containing statistic, p_value, and drift_detected boolean
        """
        if len(reference_data) == 0 or len(current_data) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Set default arguments if not provided
        if "equal_var" not in kwargs:
            kwargs["equal_var"] = DEFAULT_EQUAL_VAR
        if "nan_policy" not in kwargs:
            kwargs["nan_policy"] = DEFAULT_NAN_POLICY

        # Perform independent two-sample t-test
        # scipy.stats.ttest_ind handles both regular arrays and masked arrays
        # Masked elements are automatically ignored
        statistic, p_value = stats.ttest_ind(reference_data, current_data, **kwargs)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": alpha,
            "drift_detected": bool(p_value < alpha),
        }
