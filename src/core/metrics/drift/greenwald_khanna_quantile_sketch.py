# pylint: disable=line-too-long
"""
Greenwald–Khanna quantile sketch for streaming quantile estimation.

Implementation of the space-efficient streaming quantile summary algorithm from:
M. Greenwald and S. Khanna. Space-efficient online computation of quantile summaries.
In SIGMOD, pages 58–66, 2001.

This algorithm maintains an ε-approximate quantile summary of a data stream,
allowing quantile queries with bounded error guarantees.

Note that ranks are 1-based indexed, consistent with the GK paper.
"""

import bisect
from math import ceil, floor
from typing import List, Tuple


class GreenwaldKhannaSketch:
    """
    Greenwald–Khanna quantile sketch for streaming quantile estimation.

    This data structure maintains a compact summary of observed values from a stream,
    supporting approximate quantile queries with rank error bounded by 2εn.

    The summary consists of tuples (v, g, Δ) where:
    - v: observed value
    - g: difference in rank from previous tuple (g_i = r_min(v_i) - r_min(v_{i-1}))
    - Δ: maximum error in rank (Δ_i = r_max(v_i) - r_min(v_i))

    Invariant: For every tuple, g_i + Δ_i ≤ ⌊2εn⌋
    """

    def __init__(self, epsilon: float = 0.01):
        """
        Initialize a Greenwald–Khanna quantile sketch.

        :param epsilon: Error parameter ε ∈ (0, 1).
                        The sketch guarantees that quantile estimates
                        are within ε of the true quantile.
                        Smaller ε requires more space but provides better accuracy.
        :raises ValueError: If epsilon is not in the range (0, 1)
        """
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in the range (0, 1)")

        self.epsilon = epsilon
        self.n = 0
        self.summary: List[Tuple[float, int, int]] = []  # (value, g, delta)
        self._cumulative_r_max: List[int] = []
        self._cumulative_cache_valid = False

    def insert(self, value: float) -> None:
        """
        Insert a new value into the sketch.

        :param value: The value to insert into the sketch
        """
        self.n += 1

        if not self.summary:
            self.summary.append((value, 1, 0))
            return

        pos = bisect.bisect(self.summary, value, key=lambda x: x[0])
        compress_period = floor(1 / (2 * self.epsilon))

        if pos == 0 or pos == len(self.summary):
            g_i, delta_i = 1, 0
        elif self.n <= compress_period:
            # Early elements use Δ = 0 for stability
            g_i, delta_i = 1, 0
        else:
            # Middle insertions: Δ = ⌊2εn⌋ - 1 (corrected from paper's ⌊2εn⌋ to maintain invariant)
            # See: https://www.stevenengelhardt.com/2018/03/07/calculating-percentiles-on-streaming-data-part-2-notes-on-implementing-greenwald-khanna/
            g_i = 1
            delta_i = max(0, floor(2 * self.epsilon * self.n) - 1)

        self.summary.insert(pos, (value, g_i, delta_i))
        self._cumulative_cache_valid = False

        if self.n % compress_period == 0:
            self._compress()

    def delete(self, value: float) -> None:
        """
        Delete a value from the sketch. No action if value not found.

        :param value: The value to delete from the sketch
        :raises ValueError: If sketch is empty
        """
        if not self.summary or self.n == 0:
            raise ValueError("Cannot delete from empty sketch")

        pos = bisect.bisect_left(self.summary, value, key=lambda x: x[0])

        if pos >= len(self.summary) or self.summary[pos][0] != value:
            return

        self.n -= 1
        self._cumulative_cache_valid = False

        # Special case: if sketch becomes empty after delete
        if self.n == 0:
            self.summary = []
            return

        v_pos, g_pos, delta_pos = self.summary[pos]

        if pos == 0:
            if len(self.summary) > 1:
                v_next, g_next, delta_next = self.summary[1]
                self.summary[1] = (v_next, g_next + g_pos - 1, delta_next)
            self.summary.pop(0)
        elif pos == len(self.summary) - 1:
            self.summary.pop()
        else:
            # Transfer g-1 to next tuple (removing one element from the rank count)
            if pos < len(self.summary) - 1:
                v_next, g_next, delta_next = self.summary[pos + 1]
                self.summary.pop(pos)
                if pos < len(self.summary):
                    self.summary[pos] = (v_next, g_next + g_pos - 1, delta_next)

        compress_period = floor(1 / (2 * self.epsilon))
        if self.n > 0 and self.n % compress_period == 0:
            self._compress()

    def _ensure_cumulative_cache(self) -> None:
        """
        Build cumulative r_max cache if not already valid.

        The cache stores cumulative r_max values for each tuple, enabling
        O(log n) quantile and rank queries via binary search.
        """
        if not self._cumulative_cache_valid:
            self._cumulative_r_max = []
            r_min = 0
            for _, g, delta in self.summary:
                r_min += g
                self._cumulative_r_max.append(r_min + delta)
            self._cumulative_cache_valid = True

    def quantile(self, phi: float) -> float:
        """
        Query for the φ-quantile (approximate).

        :param phi: Quantile to query, must be in [0, 1]
        :return: Approximate φ-quantile value with rank error bounded by 2εn
        :raises ValueError: If phi is not in [0, 1] or sketch is empty
        """
        if not 0 <= phi <= 1:
            raise ValueError("phi must be in the range [0, 1]")

        if not self.summary or self.n == 0:
            raise ValueError("Cannot query quantile from empty sketch")

        target_rank = ceil(phi * self.n)
        self._ensure_cumulative_cache()

        idx = bisect.bisect_left(self._cumulative_r_max, target_rank)

        # bisect_left returns the leftmost position where cumulative_r_max[idx] >= target_rank
        # If idx >= len, target_rank exceeds all values, so return maximum
        if idx >= len(self._cumulative_r_max):
            return self.summary[-1][0]

        # Otherwise, cumulative_r_max[idx] >= target_rank by definition of bisect_left
        return self.summary[idx][0]

    def _compress(self) -> None:
        """
        Compress the summary by merging adjacent tuples where possible.

        This is the COMPRESS operation from the GK paper (Section 3.2).
        A tuple t_i can be deleted if BOTH conditions hold:
        1. Band condition: BAND(Δ_i, 2εn) ≤ BAND(Δ_{i+1}, 2εn)
        2. Invariant condition: g_i + g_{i+1} + Δ_{i+1} ≤ ⌊2εn⌋
        """
        if len(self.summary) <= 2:
            return

        threshold = floor(2 * self.epsilon * self.n)
        compressed = []
        i = 0

        while i < len(self.summary):
            v_i, g_i, delta_i = self.summary[i]

            if i < len(self.summary) - 1:
                v_next, g_next, delta_next = self.summary[i + 1]

                if threshold > 0:
                    band_condition = (delta_i // threshold) <= (delta_next // threshold)
                else:
                    band_condition = True

                invariant_condition = g_i + g_next + delta_next <= threshold

                # Never merge first tuple (exact min) or if next is last (exact max)
                can_merge = i > 0 and i < len(self.summary) - 2 and band_condition and invariant_condition

                if can_merge:
                    self.summary[i + 1] = (v_next, g_i + g_next, delta_next)
                    i += 1
                    continue

            compressed.append((v_i, g_i, delta_i))
            i += 1

        self.summary = compressed
        self._cumulative_cache_valid = False

    def rank(self, value: float) -> float:
        """
        Estimate the rank of a value in the stream.

        Uses the same cumulative r_max cache as quantile() for efficiency.
        For the streaming KS test per Lall (2015).

        :param value: The value to estimate the rank for
        :return: Estimated rank (0 to n)
        :raises ValueError: If sketch is empty
        """
        if not self.summary or self.n == 0:
            raise ValueError("Cannot estimate rank from empty sketch")

        if value < self.summary[0][0]:
            return 0.0
        if value >= self.summary[-1][0]:
            return float(self.n)

        self._ensure_cumulative_cache()

        # Find rightmost tuple with value <= query value
        pos = bisect.bisect_right(self.summary, value, key=lambda x: x[0]) - 1
        return float(self._cumulative_r_max[pos]) if pos >= 0 else 0.0

    def cdf(self, value: float) -> float:
        """
        Estimate the CDF at a given value: F(x) = P(X <= x).

        :param value: The value to estimate the CDF at
        :return: Estimated CDF value in [0, 1]
        """
        if self.n == 0:
            raise ValueError("Cannot estimate CDF from empty sketch")
        return self.rank(value) / self.n

    def min(self) -> float:
        """Return the minimum value observed."""
        if not self.summary:
            raise ValueError("Cannot get min from empty sketch")
        return self.summary[0][0]

    def max(self) -> float:
        """Return the maximum value observed."""
        if not self.summary:
            raise ValueError("Cannot get max from empty sketch")
        return self.summary[-1][0]

    def __len__(self) -> int:
        """Return the number of elements observed."""
        return self.n

    def size(self) -> int:
        """Return the number of tuples in the summary. Space is O(1/ε × log(εn))."""
        return len(self.summary)

    def merge(self, other: "GreenwaldKhannaSketch") -> "GreenwaldKhannaSketch":
        """
        Merge two sketches into a new sketch.

        :param other: Another GK sketch to merge with this one
        :return: A new merged sketch
        :raises ValueError: If the sketches have different epsilon values
        """
        if self.epsilon != other.epsilon:
            raise ValueError("Cannot merge sketches with different epsilon values")

        merged = GreenwaldKhannaSketch(self.epsilon)
        merged.n = self.n + other.n

        if not self.summary and not other.summary:
            return merged
        if not self.summary:
            merged.summary = other.summary.copy()
            return merged
        if not other.summary:
            merged.summary = self.summary.copy()
            return merged

        # Merge two sorted summaries
        i, j = 0, 0
        while i < len(self.summary) or j < len(other.summary):
            if i >= len(self.summary):
                merged.summary.append(other.summary[j])
                j += 1
            elif j >= len(other.summary):
                merged.summary.append(self.summary[i])
                i += 1
            elif self.summary[i][0] <= other.summary[j][0]:
                merged.summary.append(self.summary[i])
                i += 1
            else:
                merged.summary.append(other.summary[j])
                j += 1

        merged._compress()
        return merged
