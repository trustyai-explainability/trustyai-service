# pylint: disable=line-too-long
"""
Streaming two-sample Kolmogorov–Smirnov test for detecting distribution drift.

This module implements the streaming 2-sample KS algorithm from:
Lall, A., 2015. Data streaming algorithms for the Kolmogorov–Smirnov test.
IEEE International Conference on Big Data (Big Data), pp. 95-104.
https://doi.org/10.1109/BigData.2015.7363746

The algorithm uses Greenwald–Khanna quantile sketches to maintain approximate
CDFs for both samples, enabling computation of an approximate KS statistic
with bounded error using sublinear space.
"""

import numpy as np
from scipy.stats import kstwo

from .greenwald_khanna_quantile_sketch import GreenwaldKhannaSketch


class KolmogorovSmirnovStreaming:
    """
    Streaming two-sample Kolmogorov–Smirnov test using Greenwald–Khanna sketches.

    This class maintains two GK sketches (reference and current) and computes
    an approximate KS statistic by evaluating the CDF difference at all
    "important points" (values stored in both summaries).

    The approximation error is bounded by 2*epsilon for each CDF estimate,
    giving a total error bound of 4*epsilon for the KS statistic.

    Example usage:
        >>> ks = KolmogorovSmirnovStreaming(epsilon=0.01)
        >>> # Add reference data
        >>> for x in reference_stream:
        ...     ks.insert_reference(x)
        >>> # Add current data
        >>> for x in current_stream:
        ...     ks.insert_current(x)
        >>> # Compute test result
        >>> result = ks.kstest(alpha=0.05)
    """

    def __init__(self, epsilon: float = 0.01):
        """
        Initialize a streaming KS test with two GK sketches.

        :param epsilon: Error parameter for the GK sketches.
                        The KS statistic approximation error is bounded by 4*epsilon.
                        Smaller epsilon requires more space but provides better accuracy.
        :raises ValueError: If epsilon is not in (0, 1)
        """
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in the range (0, 1)")

        self.epsilon = epsilon
        self._reference_sketch = GreenwaldKhannaSketch(epsilon=epsilon)
        self._current_sketch = GreenwaldKhannaSketch(epsilon=epsilon)

    def insert_reference(self, value: float) -> None:
        """
        Insert a value into the reference distribution sketch.

        :param value: The value to insert
        """
        self._reference_sketch.insert(value)

    def insert_current(self, value: float) -> None:
        """
        Insert a value into the current distribution sketch.

        :param value: The value to insert
        """
        self._current_sketch.insert(value)

    def insert_reference_batch(self, values: list[float] | np.ndarray) -> None:
        """
        Insert multiple values into the reference distribution sketch.

        :param values: Array-like of values to insert
        """
        for v in values:
            self._reference_sketch.insert(float(v))

    def insert_current_batch(self, values: list[float] | np.ndarray) -> None:
        """
        Insert multiple values into the current distribution sketch.

        :param values: Array-like of values to insert
        """
        for v in values:
            self._current_sketch.insert(float(v))

    def statistic(self) -> float:
        """
        Compute the approximate KS statistic.

        The statistic is D = max_x |F_ref(x) - F_cur(x)|, evaluated at all
        "important points" (values stored in both sketch summaries).

        Uses an O(m + n) merge-scan algorithm instead of O((m+n) log(m+n))
        sorting, where m and n are the summary sizes.

        :return: Approximate KS statistic in [0, 1]
        :raises ValueError: If either sketch is empty
        """
        if len(self._reference_sketch) == 0:
            raise ValueError("Reference sketch is empty")
        if len(self._current_sketch) == 0:
            raise ValueError("Current sketch is empty")

        ref_summary = self._reference_sketch.summary
        cur_summary = self._current_sketch.summary
        n_ref = float(self._reference_sketch.n)
        n_cur = float(self._current_sketch.n)

        # Running r_max values (used for CDF estimation)
        ref_r_min, ref_r_max = 0, 0
        cur_r_min, cur_r_max = 0, 0

        i, j = 0, 0
        max_diff = 0.0

        while i < len(ref_summary) or j < len(cur_summary):
            # Determine which summary has the next smallest value
            if j >= len(cur_summary):
                # Only ref points left
                _, g, delta = ref_summary[i]
                ref_r_min += g
                ref_r_max = ref_r_min + delta
                i += 1
            elif i >= len(ref_summary):
                # Only cur points left
                _, g, delta = cur_summary[j]
                cur_r_min += g
                cur_r_max = cur_r_min + delta
                j += 1
            elif ref_summary[i][0] < cur_summary[j][0]:
                # Ref point comes first
                _, g, delta = ref_summary[i]
                ref_r_min += g
                ref_r_max = ref_r_min + delta
                i += 1
            elif cur_summary[j][0] < ref_summary[i][0]:
                # Cur point comes first
                _, g, delta = cur_summary[j]
                cur_r_min += g
                cur_r_max = cur_r_min + delta
                j += 1
            else:
                # Equal values - advance both
                _, g_ref, delta_ref = ref_summary[i]
                _, g_cur, delta_cur = cur_summary[j]
                ref_r_min += g_ref
                ref_r_max = ref_r_min + delta_ref
                cur_r_min += g_cur
                cur_r_max = cur_r_min + delta_cur
                i += 1
                j += 1

            # Compute CDF difference at this point
            f_ref = ref_r_max / n_ref
            f_cur = cur_r_max / n_cur
            diff = abs(f_ref - f_cur)
            if diff > max_diff:
                max_diff = diff

        return max_diff

    def p_value(self) -> float:
        """
        Compute the approximate p-value for the two-sample KS test.

        Uses scipy's kstwo distribution (exact two-sample KS distribution)
        with the effective sample size for the asymptotic approximation.

        :return: Approximate p-value
        :raises ValueError: If either sketch is empty
        """
        n1 = len(self._reference_sketch)
        n2 = len(self._current_sketch)

        if n1 == 0 or n2 == 0:
            raise ValueError("Both sketches must be non-empty")

        d_stat = self.statistic()

        # Effective sample size for asymptotic distribution
        # This is the same formula used by scipy.stats.ks_2samp
        n_eff = (n1 * n2) / (n1 + n2)

        # Use scipy's kstwo survival function
        # kstwo.sf(d, n) computes P(D_n > d) for the two-sample KS statistic
        p_val = kstwo.sf(d_stat, round(n_eff))

        return float(p_val)

    def kstest(self, alpha: float = 0.05) -> dict[str, float | bool]:
        """
        Perform the streaming two-sample KS test.

        :param alpha: Significance level for hypothesis testing (default: 0.05)
        :return: Dictionary containing:
                 - statistic: Approximate KS statistic
                 - p_value: Approximate p-value
                 - alpha: Significance level used
                 - drift_detected: True if p_value < alpha
                 - n_reference: Number of reference samples
                 - n_current: Number of current samples
                 - epsilon: Error parameter used
        :raises ValueError: If either sketch is empty
        """
        stat = self.statistic()
        p_val = self.p_value()

        return {
            "statistic": stat,
            "p_value": p_val,
            "alpha": alpha,
            "drift_detected": bool(p_val < alpha),
            "n_reference": len(self._reference_sketch),
            "n_current": len(self._current_sketch),
            "epsilon": self.epsilon,
        }

    def reset_reference(self) -> None:
        """Reset the reference sketch to empty."""
        self._reference_sketch = GreenwaldKhannaSketch(epsilon=self.epsilon)

    def reset_current(self) -> None:
        """Reset the current sketch to empty."""
        self._current_sketch = GreenwaldKhannaSketch(epsilon=self.epsilon)

    def reset(self) -> None:
        """Reset both sketches to empty."""
        self.reset_reference()
        self.reset_current()

    @property
    def n_reference(self) -> int:
        """Number of elements in the reference sketch."""
        return len(self._reference_sketch)

    @property
    def n_current(self) -> int:
        """Number of elements in the current sketch."""
        return len(self._current_sketch)

    @property
    def reference_sketch(self) -> GreenwaldKhannaSketch:
        """Access the reference GK sketch (read-only recommended)."""
        return self._reference_sketch

    @property
    def current_sketch(self) -> GreenwaldKhannaSketch:
        """Access the current GK sketch (read-only recommended)."""
        return self._current_sketch

    def to_dict(self) -> dict[str, float | dict]:
        """
        Serialize the streaming KS test to a dictionary.

        :return: Dictionary containing all state for serialization
        """
        return {
            "epsilon": self.epsilon,
            "reference_sketch": self._reference_sketch.to_dict(),
            "current_sketch": self._current_sketch.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | dict]) -> "KolmogorovSmirnovStreaming":
        """
        Deserialize a streaming KS test from a dictionary.

        :param data: Dictionary containing state (from to_dict())
        :return: Reconstructed KolmogorovSmirnovStreaming instance
        :raises ValueError: If the data format is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        required_keys = {"epsilon", "reference_sketch", "current_sketch"}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"Missing required keys: {required_keys - data.keys()}")

        ks = cls(epsilon=float(data["epsilon"]))
        ks._reference_sketch = GreenwaldKhannaSketch.from_dict(data["reference_sketch"])
        ks._current_sketch = GreenwaldKhannaSketch.from_dict(data["current_sketch"])

        return ks
