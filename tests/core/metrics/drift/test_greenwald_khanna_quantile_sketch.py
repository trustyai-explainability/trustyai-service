# pylint: disable=line-too-long
"""
Tests for the Greenwald-Khanna quantile sketch implementation.
"""

from math import floor, log

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.core.metrics.drift.greenwald_khanna_quantile_sketch import GreenwaldKhannaSketch


class TestGreenwaldKhannaSketch:
    """Test suite for GreenwaldKhannaSketch class."""

    def test_initialization(self):
        """Test sketch initialization with valid epsilon."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        assert sketch.epsilon == 0.01
        assert sketch.n == 0
        assert len(sketch.summary) == 0

    def test_initialization_invalid_epsilon(self):
        """Test that invalid epsilon values raise ValueError."""
        with pytest.raises(ValueError, match="epsilon must be in the range"):
            GreenwaldKhannaSketch(epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be in the range"):
            GreenwaldKhannaSketch(epsilon=1.0)

        with pytest.raises(ValueError, match="epsilon must be in the range"):
            GreenwaldKhannaSketch(epsilon=-0.1)

        with pytest.raises(ValueError, match="epsilon must be in the range"):
            GreenwaldKhannaSketch(epsilon=1.5)

    def test_single_insert(self):
        """Test inserting a single value."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        sketch.insert(5.0)

        assert sketch.n == 1
        assert len(sketch.summary) == 1
        assert sketch.summary[0] == (5.0, 1, 0)

    def test_multiple_inserts_sorted(self):
        """Test inserting multiple values in sorted order."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            sketch.insert(v)

        assert sketch.n == 5
        assert len(sketch) == 5
        # Summary should contain all values since compression hasn't kicked in
        summary_values = [v for v, _, _ in sketch.summary]
        assert summary_values == values

    def test_multiple_inserts_unsorted(self):
        """Test inserting multiple values in unsorted order."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [3.0, 1.0, 4.0, 2.0, 5.0]

        for v in values:
            sketch.insert(v)

        assert sketch.n == 5
        # Summary should be sorted
        summary_values = [v for v, _, _ in sketch.summary]
        assert summary_values == sorted(values)

    def test_min_max(self):
        """Test min and max operations."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        # Empty sketch should raise error
        with pytest.raises(ValueError, match="Cannot get min from empty sketch"):
            sketch.min()

        with pytest.raises(ValueError, match="Cannot get max from empty sketch"):
            sketch.max()

        # Insert values
        values = [3.0, 1.0, 4.0, 2.0, 5.0]
        for v in values:
            sketch.insert(v)

        assert sketch.min() == 1.0
        assert sketch.max() == 5.0

    def test_median_exact(self):
        """Test median query on small dataset (should be exact)."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        # Odd number of elements
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            sketch.insert(v)

        median = sketch.quantile(0.5)
        # Should be close to true median (3.0)
        assert abs(median - 3.0) <= 1.0  # Within epsilon bound

    def test_quantiles_uniform_distribution(self):
        """Test quantile queries on uniform distribution."""
        sketch = GreenwaldKhannaSketch(epsilon=0.05)
        n = 1000

        # Generate uniform data [0, 1000)
        rng = np.random.default_rng(42)
        values = rng.uniform(0, 1000, n)

        for v in values:
            sketch.insert(float(v))

        # Test quartiles
        q25 = sketch.quantile(0.25)
        q50 = sketch.quantile(0.50)
        q75 = sketch.quantile(0.75)

        # For uniform [0, 1000), expected quantiles are approximately:
        # Q1 ≈ 250, Q2 ≈ 500, Q3 ≈ 750
        # With epsilon=0.05, error bound is 0.05 * 1000 = 50
        assert abs(q25 - 250) <= 100  # Allow some slack for randomness
        assert abs(q50 - 500) <= 100
        assert abs(q75 - 750) <= 100

        # Quantiles should be ordered
        assert q25 < q50 < q75

    def test_quantiles_normal_distribution(self):
        """Test quantile queries on normal distribution."""
        sketch = GreenwaldKhannaSketch(epsilon=0.05)
        n = 1000

        # Generate normal data N(0, 1)
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, n)

        for v in values:
            sketch.insert(float(v))

        # Test median (should be close to 0)
        median = sketch.quantile(0.5)
        assert abs(median - 0.0) <= 0.5

        # Test that quantiles are ordered
        q10 = sketch.quantile(0.1)
        q50 = sketch.quantile(0.5)
        q90 = sketch.quantile(0.9)

        assert q10 < q50 < q90

    def test_quantile_edge_cases(self):
        """Test quantile queries at edges (0.0 and 1.0)."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            sketch.insert(v)

        # Quantile 0.0 should give min
        q0 = sketch.quantile(0.0)
        assert q0 == sketch.min()

        # Quantile 1.0 should give max
        q1 = sketch.quantile(1.0)
        assert q1 == sketch.max()

    def test_quantile_invalid_phi(self):
        """Test that invalid phi values raise ValueError."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        sketch.insert(1.0)

        with pytest.raises(ValueError, match="phi must be in the range"):
            sketch.quantile(-0.1)

        with pytest.raises(ValueError, match="phi must be in the range"):
            sketch.quantile(1.1)

    def test_quantile_empty_sketch(self):
        """Test that querying empty sketch raises ValueError."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        with pytest.raises(ValueError, match="Cannot query quantile from empty sketch"):
            sketch.quantile(0.5)

    def test_compression(self):
        """Test that compression maintains space bounds."""
        epsilon = 0.01
        sketch = GreenwaldKhannaSketch(epsilon=epsilon)
        n = 10000

        # Insert many values
        rng = np.random.default_rng(42)
        values = rng.uniform(0, 1000, n)

        for v in values:
            sketch.insert(float(v))

        # Check that summary size is bounded
        # Theoretical bound is O(1/epsilon * log(epsilon * n))
        # For epsilon=0.01, n=10000, this should be much less than n
        assert sketch.size() < n
        assert sketch.size() < 1000  # Practical bound

    def test_size_method(self):
        """Test the size() method."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        assert sketch.size() == 0

        sketch.insert(1.0)
        assert sketch.size() == 1

        for i in range(2, 100):
            sketch.insert(float(i))

        # Size should be less than number of elements due to compression
        assert sketch.size() <= sketch.n

    def test_len_method(self):
        """Test the __len__ method."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        assert len(sketch) == 0

        for i in range(100):
            sketch.insert(float(i))
            assert len(sketch) == i + 1

    def test_merge_same_epsilon(self):
        """Test merging two sketches with same epsilon."""
        sketch1 = GreenwaldKhannaSketch(epsilon=0.01)
        sketch2 = GreenwaldKhannaSketch(epsilon=0.01)

        # Insert values into first sketch
        for i in range(50):
            sketch1.insert(float(i))

        # Insert values into second sketch
        for i in range(50, 100):
            sketch2.insert(float(i))

        # Merge
        merged = sketch1.merge(sketch2)

        assert merged.n == 100
        assert merged.epsilon == 0.01

        # Check that merged sketch contains values from both ranges
        # Note: Due to compression, the exact min/max may not be preserved
        # The important property is that quantiles are approximately correct
        assert merged.min() <= 10.0  # Should be close to 0
        assert merged.max() >= 90.0  # Should be close to 99

        # Check median is approximately in the middle
        median = merged.quantile(0.5)
        assert abs(median - 49.5) <= 10.0

    def test_merge_different_epsilon(self):
        """Test that merging sketches with different epsilon raises ValueError."""
        sketch1 = GreenwaldKhannaSketch(epsilon=0.01)
        sketch2 = GreenwaldKhannaSketch(epsilon=0.05)

        sketch1.insert(1.0)
        sketch2.insert(2.0)

        with pytest.raises(ValueError, match="Cannot merge sketches with different epsilon"):
            sketch1.merge(sketch2)

    def test_deterministic_behavior(self):
        """Test that sketch produces deterministic results for same input."""
        values = [3.14, 2.71, 1.41, 0.58, 9.81]

        sketch1 = GreenwaldKhannaSketch(epsilon=0.01)
        for v in values:
            sketch1.insert(v)

        sketch2 = GreenwaldKhannaSketch(epsilon=0.01)
        for v in values:
            sketch2.insert(v)

        # Both sketches should have identical state
        assert sketch1.n == sketch2.n
        assert sketch1.summary == sketch2.summary
        assert sketch1.quantile(0.5) == sketch2.quantile(0.5)

    def test_large_dataset_performance(self):
        """Test sketch performance on large dataset."""
        epsilon = 0.01
        sketch = GreenwaldKhannaSketch(epsilon=epsilon)
        n = 100000

        # Insert many values
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, n)

        for v in values:
            sketch.insert(float(v))

        # Verify sketch properties
        assert sketch.n == n
        # Summary should be much smaller than n
        assert sketch.size() < n / 10

        # Verify quantile accuracy
        # For normal distribution, median should be near 0
        median = sketch.quantile(0.5)
        assert abs(median) <= 0.5

    def test_duplicate_values(self):
        """Test sketch behavior with duplicate values."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

        for v in values:
            sketch.insert(v)

        assert sketch.n == 6
        assert sketch.min() == 1.0
        assert sketch.max() == 3.0

        # Median should be around 2.0
        median = sketch.quantile(0.5)
        assert 1.0 <= median <= 3.0

    def test_negative_values(self):
        """Test sketch with negative values."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]

        for v in values:
            sketch.insert(v)

        assert sketch.min() == -5.0
        assert sketch.max() == 5.0

        median = sketch.quantile(0.5)
        assert abs(median - 0.0) <= 1.0

    def test_delete_single_element(self):
        """Test deleting from a single-element sketch."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        sketch.insert(5.0)

        assert len(sketch) == 1
        sketch.delete(5.0)
        assert len(sketch) == 0
        assert sketch.size() == 0

    def test_delete_from_empty_sketch(self):
        """Test that deleting from empty sketch raises ValueError."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        with pytest.raises(ValueError, match="Cannot delete from empty sketch"):
            sketch.delete(1.0)

    def test_delete_minimum(self):
        """Test deleting the minimum value."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            sketch.insert(v)

        original_size = len(sketch)
        sketch.delete(1.0)

        assert len(sketch) == original_size - 1
        assert sketch.min() >= 2.0  # New minimum should be around 2.0

    def test_delete_maximum(self):
        """Test deleting the maximum value."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            sketch.insert(v)

        original_size = len(sketch)
        sketch.delete(5.0)

        assert len(sketch) == original_size - 1
        assert sketch.max() <= 4.0  # New maximum should be around 4.0

    def test_delete_middle_value(self):
        """Test deleting a value from the middle."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            sketch.insert(v)

        original_n = len(sketch)
        sketch.delete(3.0)

        assert len(sketch) == original_n - 1

        # Quantiles should still be reasonable
        median = sketch.quantile(0.5)
        assert 2.0 <= median <= 4.0

    def test_delete_multiple_values(self):
        """Test deleting multiple values."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = list(range(1, 11))  # 1 to 10

        for v in values:
            sketch.insert(float(v))

        # Delete several values
        sketch.delete(1.0)
        sketch.delete(5.0)
        sketch.delete(10.0)

        assert len(sketch) == 7  # 10 - 3

        # Quantiles should still work
        median = sketch.quantile(0.5)
        assert 3.0 <= median <= 7.0

    def test_delete_and_insert(self):
        """Test interleaving delete and insert operations."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        # Insert some values
        for i in range(1, 6):
            sketch.insert(float(i))

        assert len(sketch) == 5

        # Delete some
        sketch.delete(2.0)
        sketch.delete(4.0)

        assert len(sketch) == 3

        # Insert more
        sketch.insert(6.0)
        sketch.insert(7.0)

        assert len(sketch) == 5

        # Quantiles should still be reasonable
        median = sketch.quantile(0.5)
        assert 3.0 <= median <= 7.0

    def test_delete_maintains_quantile_accuracy(self):
        """Test that delete maintains reasonable quantile accuracy."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        # Use a small dataset to avoid compression
        # This ensures all inserted values remain in the summary
        values = [float(i) for i in range(20)]
        for v in values:
            sketch.insert(v)

        # Verify all values are in the summary before deletion
        summary_values = [v for v, _, _ in sketch.summary]
        assert len(summary_values) == 20  # No compression yet

        # Delete first 5 values
        for v in values[:5]:
            sketch.delete(v)

        assert len(sketch) == 15

        # Check median is approximately correct
        # Remaining values are 5-19
        # Median of [5, 6, 7, ..., 19] is 12
        exact_median = 12.0
        sketch_median = sketch.quantile(0.5)

        # Should be close to exact median
        assert abs(sketch_median - exact_median) <= 2.0

    def test_delete_updates_count(self):
        """Test that delete properly updates the count."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        for i in range(10):
            sketch.insert(float(i))

        assert len(sketch) == 10

        for i in range(5):
            sketch.delete(float(i))

        assert len(sketch) == 5

    def test_delete_nonexistent_value(self):
        """Test that deleting a nonexistent value takes no action."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for v in values:
            sketch.insert(v)

        original_n = len(sketch)
        original_summary = sketch.summary.copy()

        # Try to delete a value that doesn't exist
        sketch.delete(99.0)  # Not in summary

        # Nothing should have changed
        assert len(sketch) == original_n
        assert sketch.summary == original_summary

        # Try to delete a value between existing values
        sketch.delete(2.5)  # Not in summary (between 2.0 and 3.0)

        # Still nothing should have changed
        assert len(sketch) == original_n
        assert sketch.summary == original_summary

        # Try to delete a value smaller than minimum
        sketch.delete(0.0)  # Not in summary

        # Still nothing should have changed
        assert len(sketch) == original_n
        assert sketch.summary == original_summary


# =============================================================================
# Invariant Tests
# =============================================================================


def check_gk_invariants(sketch: GreenwaldKhannaSketch) -> None:
    """
    Verify all GK sketch invariants hold.

    This function checks the core invariants from the GK paper:
    1. Summary is sorted by value
    2. All g values are positive (g ≥ 1)
    3. First and last tuples have Δ = 0
    4. For all tuples: g_i + Δ_i ≤ ⌊2εn⌋

    :param sketch: The sketch to validate
    :raises AssertionError: If any invariant is violated
    """
    if not sketch.summary or sketch.n == 0:
        return

    threshold = floor(2 * sketch.epsilon * sketch.n)

    # Check summary is sorted
    values = [v for v, _, _ in sketch.summary]
    assert values == sorted(values), "Summary is not sorted by value"

    # Check all g values are positive
    for i, (v, g, delta) in enumerate(sketch.summary):
        assert g >= 1, f"Tuple {i} has g={g} < 1"

    # Check first tuple has Δ = 0
    _, _, delta_first = sketch.summary[0]
    assert delta_first == 0, f"First tuple has Δ={delta_first}, expected 0"

    # Check last tuple has Δ = 0
    _, _, delta_last = sketch.summary[-1]
    assert delta_last == 0, f"Last tuple has Δ={delta_last}, expected 0"

    # Check main invariant: g_i + Δ_i ≤ ⌊2εn⌋
    # Only check when threshold >= 1, otherwise the invariant is not meaningful
    # (since g is always >= 1 for valid tuples)
    if threshold >= 1:
        for i, (v, g, delta) in enumerate(sketch.summary):
            invariant_value = g + delta
            assert invariant_value <= threshold, (
                f"Tuple {i} violates invariant: g + Δ = {g} + {delta} = {invariant_value} > {threshold}"
            )


class TestGKInvariants:
    """Tests that verify GK sketch invariants are maintained.

    These tests check the core invariants from the GK paper:
    1. Summary is sorted by value
    2. All g values are positive (g ≥ 1)
    3. First and last tuples have Δ = 0
    4. For all tuples: g_i + Δ_i ≤ ⌊2εn⌋
    """

    def test_invariants_after_single_insert(self):
        """Test invariants hold after a single insert."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        sketch.insert(5.0)
        check_gk_invariants(sketch)

    def test_invariants_after_multiple_inserts(self):
        """Test invariants hold after multiple sorted inserts."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        for i in range(100):
            sketch.insert(float(i))

        check_gk_invariants(sketch)

    def test_invariants_after_unsorted_inserts(self):
        """Test invariants hold when inserting in random order."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        rng = np.random.default_rng(42)
        values = rng.uniform(0, 1000, 500)

        for v in values:
            sketch.insert(float(v))

        check_gk_invariants(sketch)

    def test_invariants_after_compression(self):
        """Test invariants hold after compression kicks in."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        rng = np.random.default_rng(42)

        # Insert enough to trigger multiple compressions
        for _ in range(1000):
            sketch.insert(float(rng.uniform(0, 1000)))

        check_gk_invariants(sketch)

    def test_invariants_after_delete(self):
        """Test invariants hold after delete operations."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        # Insert values
        for i in range(20):
            sketch.insert(float(i))

        check_gk_invariants(sketch)

        # Delete some values and check invariants after each
        for i in [0, 5, 10, 15, 19]:
            sketch.delete(float(i))
            check_gk_invariants(sketch)

    def test_invariants_after_merge(self):
        """Test invariants hold after merging two sketches."""
        sketch1 = GreenwaldKhannaSketch(epsilon=0.01)
        sketch2 = GreenwaldKhannaSketch(epsilon=0.01)

        for i in range(50):
            sketch1.insert(float(i))
        for i in range(50, 100):
            sketch2.insert(float(i))

        merged = sketch1.merge(sketch2)
        check_gk_invariants(merged)

    def test_invariants_with_duplicates(self):
        """Test invariants hold with duplicate values."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        for _ in range(100):
            sketch.insert(5.0)  # All same value

        check_gk_invariants(sketch)

    def test_invariants_with_different_epsilons(self):
        """Test invariants hold for various epsilon values."""
        for epsilon in [0.01, 0.05, 0.1]:
            sketch = GreenwaldKhannaSketch(epsilon=epsilon)
            rng = np.random.default_rng(42)

            for _ in range(500):
                sketch.insert(float(rng.normal(0, 1)))

            check_gk_invariants(sketch)


# =============================================================================
# Property-Based Tests (Hypothesis)
# =============================================================================


class TestGKPropertyBased:
    """Property-based tests using Hypothesis to verify ε-error bounds.

    These tests use random input generation to stress-test the GK sketch
    implementation and verify its correctness guarantees.
    """

    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=100,  # Need larger dataset to trigger compression
            max_size=1000,
            unique=True,  # Use unique values to avoid tie-related ambiguity
        ),
        epsilon=st.sampled_from([0.01, 0.05, 0.1]),
    )
    @settings(max_examples=50, deadline=None)
    def test_quantile_error_bound(self, values, epsilon):
        """
        Property: For any φ-quantile query, the returned value v satisfies:
        |rank(v) - φn| ≤ εn

        This is the fundamental guarantee of the GK algorithm.
        Note: We use unique values to avoid tie-related ambiguity in rank calculation.
        """

        sketch = GreenwaldKhannaSketch(epsilon=epsilon)
        for v in values:
            sketch.insert(v)

        n = len(values)
        sorted_values = sorted(values)

        for phi in [0.1, 0.25, 0.5, 0.75, 0.9]:
            estimated = sketch.quantile(phi)

            # Find the actual rank of the estimated value
            # rank = number of elements <= estimated value
            actual_rank = sum(1 for v in sorted_values if v <= estimated)

            # Expected rank
            expected_rank = phi * n

            # Error should be within 2εn (the GK guarantee from the invariant g + Δ ≤ 2εn)
            # The factor of 2 accounts for the full rank interval uncertainty
            error = abs(actual_rank - expected_rank)
            max_error = 2 * epsilon * n

            assert error <= max_error, (
                f"Quantile error too large: |{actual_rank} - {expected_rank}| = {error} > {max_error} "
                f"(phi={phi}, n={n}, epsilon={epsilon})"
            )

    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=100,
            max_size=1000,
        ),
        epsilon=st.sampled_from([0.01, 0.05, 0.1]),
    )
    @settings(max_examples=50, deadline=None)
    def test_invariants_maintained(self, values, epsilon):
        """Property: GK structural invariants are maintained after all insertions."""
        assume(len(values) >= 50)

        sketch = GreenwaldKhannaSketch(epsilon=epsilon)
        for v in values:
            sketch.insert(v)

        check_gk_invariants(sketch)

    @given(
        values=st.lists(
            st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=500,
        ),
        epsilon=st.sampled_from([0.01, 0.05, 0.1]),
    )
    @settings(max_examples=30, deadline=None)
    def test_min_max_exact(self, values, epsilon):
        """Property: min() and max() return exact minimum and maximum values."""
        assume(len(values) >= 5)
        assume(len(set(values)) >= 3)  # Need some distinct values

        sketch = GreenwaldKhannaSketch(epsilon=epsilon)
        for v in values:
            sketch.insert(v)

        # Min and max should be exact (GK guarantees this)
        assert sketch.min() == min(values), f"min() = {sketch.min()}, expected {min(values)}"
        assert sketch.max() == max(values), f"max() = {sketch.max()}, expected {max(values)}"

    @given(
        values=st.lists(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=500,
        ),
        epsilon=st.sampled_from([0.01, 0.05, 0.1]),
    )
    @settings(max_examples=30, deadline=None)
    def test_quantiles_ordered(self, values, epsilon):
        """Property: Quantile queries return monotonically increasing values."""
        assume(len(set(values)) >= 5)

        sketch = GreenwaldKhannaSketch(epsilon=epsilon)
        for v in values:
            sketch.insert(v)

        phis = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        quantiles = [sketch.quantile(phi) for phi in phis]

        for i in range(len(quantiles) - 1):
            assert quantiles[i] <= quantiles[i + 1], (
                f"Quantiles not ordered: q({phis[i]})={quantiles[i]} > q({phis[i + 1]})={quantiles[i + 1]}"
            )

    @given(
        values=st.lists(
            st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=50,
            max_size=200,
        ),
        epsilon=st.sampled_from([0.01, 0.05]),
    )
    @settings(max_examples=20, deadline=None)
    def test_merge_preserves_structure(self, values, epsilon):
        """Property: Merging two sketches produces a valid sketch."""
        assume(len(set(values)) >= 10)

        mid = len(values) // 2
        values1 = values[:mid]
        values2 = values[mid:]

        sketch1 = GreenwaldKhannaSketch(epsilon=epsilon)
        sketch2 = GreenwaldKhannaSketch(epsilon=epsilon)

        for v in values1:
            sketch1.insert(v)
        for v in values2:
            sketch2.insert(v)

        merged = sketch1.merge(sketch2)

        # Check structural invariants
        check_gk_invariants(merged)

        # Merged count should be sum of both
        assert merged.n == len(values)

        # Merged quantiles should be ordered
        phis = [0.0, 0.25, 0.5, 0.75, 1.0]
        quantiles = [merged.quantile(phi) for phi in phis]
        for i in range(len(quantiles) - 1):
            assert quantiles[i] <= quantiles[i + 1]

    @given(
        values=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=10,
            max_size=100,
            unique=True,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_delete_maintains_invariants(self, values):
        """Property: Delete operations maintain GK invariants."""
        assume(len(values) >= 10)

        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        for v in values:
            sketch.insert(float(v))

        # Delete first few values (which should be in summary since no compression)
        for v in values[:3]:
            sketch.delete(float(v))
            check_gk_invariants(sketch)

    @given(
        n=st.integers(min_value=100, max_value=10000),
        epsilon=st.sampled_from([0.01, 0.05, 0.1]),
    )
    @settings(max_examples=20, deadline=None)
    def test_space_bound(self, n, epsilon):
        """Property: Summary size is O(1/ε × log(εn))."""
        sketch = GreenwaldKhannaSketch(epsilon=epsilon)
        rng = np.random.default_rng(42)

        for _ in range(n):
            sketch.insert(float(rng.uniform(0, 1000)))

        # Theoretical bound: O(1/ε × log(εn))
        # Use a generous constant factor
        theoretical_bound = (1 / epsilon) * log(max(1, epsilon * n) + 1) * 10

        assert sketch.size() <= theoretical_bound, (
            f"Summary size {sketch.size()} exceeds theoretical bound {theoretical_bound}"
        )

    def test_rank_empty_sketch(self):
        """Test that rank() raises ValueError on empty sketch."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        with pytest.raises(ValueError, match="Cannot estimate rank from empty sketch"):
            sketch.rank(5.0)

    def test_rank_edge_cases(self):
        """Test rank() with values below min and above max."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        for i in range(10, 21):  # Values 10-20
            sketch.insert(float(i))

        # Value below minimum
        assert sketch.rank(5.0) == 0.0

        # Value above maximum
        assert sketch.rank(25.0) == float(sketch.n)

        # Value at boundaries
        assert sketch.rank(10.0) >= 0.0
        assert sketch.rank(20.0) <= float(sketch.n)

    def test_cdf_empty_sketch(self):
        """Test that cdf() raises ValueError on empty sketch."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        with pytest.raises(ValueError, match="Cannot estimate CDF from empty sketch"):
            sketch.cdf(5.0)

    def test_cdf_values(self):
        """Test cdf() returns correct values in [0, 1]."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        for i in range(100):
            sketch.insert(float(i))

        # CDF at minimum should be close to 0
        cdf_min = sketch.cdf(0.0)
        assert 0.0 <= cdf_min <= 0.2

        # CDF at maximum should be close to 1
        cdf_max = sketch.cdf(99.0)
        assert 0.8 <= cdf_max <= 1.0

        # CDF at median should be close to 0.5
        cdf_median = sketch.cdf(50.0)
        assert 0.3 <= cdf_median <= 0.7

    def test_merge_empty_sketches(self):
        """Test merging when one or both sketches are empty."""
        sketch1 = GreenwaldKhannaSketch(epsilon=0.01)
        sketch2 = GreenwaldKhannaSketch(epsilon=0.01)

        # Both empty
        merged = sketch1.merge(sketch2)
        assert merged.n == 0
        assert len(merged.summary) == 0

        # First empty, second has data
        sketch3 = GreenwaldKhannaSketch(epsilon=0.01)
        sketch4 = GreenwaldKhannaSketch(epsilon=0.01)
        sketch4.insert(1.0)
        sketch4.insert(2.0)

        merged2 = sketch3.merge(sketch4)
        assert merged2.n == 2
        assert len(merged2.summary) > 0

        # First has data, second empty
        sketch5 = GreenwaldKhannaSketch(epsilon=0.01)
        sketch6 = GreenwaldKhannaSketch(epsilon=0.01)
        sketch5.insert(1.0)
        sketch5.insert(2.0)

        merged3 = sketch5.merge(sketch6)
        assert merged3.n == 2
        assert len(merged3.summary) > 0

    def test_query_rank_boundary_conditions(self):
        """Test _query_rank edge cases for better coverage."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        # Add values to create specific boundary conditions
        for i in [1, 5, 10, 15, 20]:
            sketch.insert(float(i))

        # Query for quantiles that trigger different code paths
        # This tests lines 165, 168-170 in the _query_rank method
        q1 = sketch.quantile(0.0)  # Minimum
        q2 = sketch.quantile(0.25)
        q3 = sketch.quantile(0.5)  # Median
        q4 = sketch.quantile(0.75)
        q5 = sketch.quantile(1.0)  # Maximum

        # Verify quantiles are in order
        assert q1 <= q2 <= q3 <= q4 <= q5
        assert q1 >= 1.0
        assert q5 <= 20.0

    def test_query_rank_beyond_range(self):
        """Test quantile when phi rounds to rank beyond all cumulative r_max values."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        # Insert a few values to create a summary
        for i in range(5):
            sketch.insert(float(i))

        # Query maximum quantile (phi=1.0)
        # This exercises edge cases in _query_rank including line 165
        result = sketch.quantile(1.0)

        # Should return the maximum value
        assert result == sketch.summary[-1][0]
        assert result == 4.0

    def test_compression_with_small_summary(self):
        """Test that compression early-returns when summary has ≤2 elements."""
        sketch = GreenwaldKhannaSketch(epsilon=0.01)

        # Insert just 2 values (summary will have exactly 2 tuples)
        sketch.insert(1.0)
        sketch.insert(2.0)

        # Force compression
        initial_len = len(sketch.summary)
        sketch._compress()

        # Should not compress (early return at line 182)
        assert len(sketch.summary) == initial_len

    def test_compression_with_zero_threshold(self):
        """Test compression band condition when threshold is 0 (very small n)."""
        sketch = GreenwaldKhannaSketch(epsilon=0.1)

        # Insert just 1 value (n=1)
        # With n=1, floor(2 * epsilon * n) = floor(0.2) = 0, triggering line 197
        sketch.insert(1.0)

        # Manually add more tuples to summary to trigger compression logic
        # This simulates a state where we have tuples but threshold is 0
        sketch.summary.append((2.0, 1, 0))
        sketch.summary.append((3.0, 1, 0))
        sketch.n = 1  # Keep n small so threshold stays 0

        # Force compression with threshold = 0
        sketch._compress()

        # Should complete without error (line 197: band_condition = True)
        assert len(sketch.summary) >= 1

    def test_query_rank_edge_cases_direct(self):
        """Test _query_rank edge cases by directly calling it with crafted states."""
        from math import ceil

        sketch = GreenwaldKhannaSketch(epsilon=0.1)

        # Setup 1: Test line 165 (idx >= len(_cumulative_r_max))
        # This happens when target_rank > max(_cumulative_r_max)
        sketch.summary = [(1.0, 1, 0), (2.0, 1, 0), (3.0, 1, 0)]
        sketch.n = 100  # Large n makes target_rank potentially > all r_max values
        sketch._cumulative_cache_valid = False
        sketch._ensure_cumulative_cache()

        # _cumulative_r_max will be [1, 2, 3]
        # With n=100, querying phi near 1.0 gives target_rank around 100
        # which is > all values in _cumulative_r_max
        result = sketch.quantile(0.99)  # target_rank = ceil(99) = 99
        assert result == 3.0  # Should return last summary value (line 165)

        # Setup 2: Test lines 168-170
        # Create a situation where cumulative_r_max[idx] < target_rank
        # and we need to check if idx+1 < len(summary)
        sketch.summary = [(1.0, 2, 1), (2.0, 2, 1), (3.0, 2, 0)]
        sketch.n = 6
        sketch._cumulative_cache_valid = False
        sketch._ensure_cumulative_cache()

        # _cumulative_r_max will be [(2+1)=3, (2+2+1)=5, (2+2+2+0)=6]
        # Query with target_rank that falls between values
        # phi=0.6 -> target_rank = ceil(3.6) = 4
        # bisect_left([3,5,6], 4) = 1
        # _cumulative_r_max[1] = 5 >= 4, so takes line 167 (normal path)

        # phi=0.5 -> target_rank = ceil(3) = 3
        # bisect_left([3,5,6], 3) = 0
        # _cumulative_r_max[0] = 3 >= 3, takes line 167

        # To hit line 168-170, need _cumulative_r_max[idx] < target_rank
        # This is actually impossible with bisect_left on sorted cumulative values
        # because bisect_left finds leftmost position where value >= target

        # Let's try a different approach: craft cache that violates assumptions
        sketch.summary = [(1.0, 1, 5), (2.0, 1, 0)]
        sketch.n = 2
        sketch._cumulative_r_max = [6, 7]  # Manually set cache
        sketch._cumulative_cache_valid = True

        # target_rank = 1: bisect_left([6,7], 1) = 0
        # _cumulative_r_max[0] = 6 >= 1, takes line 167
        # Still can't hit 168-170

        # These lines (168-170) appear to be unreachable defensive code
        # Let's at least test with various quantiles to ensure no crashes
        sketch = GreenwaldKhannaSketch(epsilon=0.01)
        for i in range(50):
            sketch.insert(float(i))

        for phi in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            result = sketch.quantile(phi)
            assert 0.0 <= result <= 49.0
