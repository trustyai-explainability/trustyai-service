"""Tests for MMD (Maximum Mean Discrepancy) drift detection metric."""

import numpy as np
import pytest

from src.core.metrics.drift.mmd import MMD

from . import factory


class TestMMDRFFFactory:
    """Factory-generated tests for MMD RFF multivariate drift detection."""

    test_multivariate_no_drift = factory.make_multivariate_no_drift_test(
        metric_fn=MMD.compute,
        params={"method": "rff", "alpha": 0.05, "num_permutations": 99},
        statistic_key="statistic",
    )

    test_multivariate_detects_shift = factory.make_multivariate_detects_shift_test(
        metric_fn=MMD.compute,
        params={"method": "rff", "alpha": 0.05, "num_permutations": 99},
    )


class TestMMDCTT:
    """Direct tests for MMD CTT-specific behavior."""

    def test_return_structure(self) -> None:
        """CTT returns all expected keys with correct types."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 3))
        cur = rng.standard_normal((100, 3))

        result = MMD.compute(ref, cur, method="ctt", seed=42)

        assert set(result.keys()) == {
            "statistic",
            "p_value",
            "threshold",
            "alpha",
            "drift_detected",
        }
        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)
        assert isinstance(result["threshold"], float)
        assert isinstance(result["alpha"], float)
        assert isinstance(result["drift_detected"], bool)

    def test_p_value_range(self) -> None:
        """P-value must be in (0, 1]."""
        rng = np.random.default_rng(123)
        ref = rng.standard_normal((80, 2))
        cur = rng.standard_normal((80, 2))

        result = MMD.compute(ref, cur, method="ctt", seed=123)

        assert 0 < result["p_value"] <= 1

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        rng = np.random.default_rng(99)
        ref = rng.standard_normal((100, 4))
        cur = rng.standard_normal((100, 4))

        r1 = MMD.compute(ref, cur, method="ctt", seed=7)
        r2 = MMD.compute(ref, cur, method="ctt", seed=7)

        assert r1["statistic"] == r2["statistic"]
        assert r1["p_value"] == r2["p_value"]
        assert r1["drift_detected"] == r2["drift_detected"]

    def test_drift_with_large_shift(self) -> None:
        """Large mean shift should be detected deterministically."""
        rng = np.random.default_rng(77)
        ref = rng.standard_normal((200, 3))
        cur = rng.standard_normal((200, 3)) + 5.0

        result = MMD.compute(ref, cur, method="ctt", seed=77)

        assert result["drift_detected"] is True

    def test_no_drift_identical(self) -> None:
        """Identical distributions should not flag drift."""
        rng = np.random.default_rng(55)
        data = rng.standard_normal((200, 3))
        ref = data[:100]
        cur = data[100:]

        result = MMD.compute(ref, cur, method="ctt", seed=55)

        assert result["drift_detected"] is False

    def test_custom_bandwidth(self) -> None:
        """Custom bandwidth parameter works."""
        rng = np.random.default_rng(1)
        ref = rng.standard_normal((50, 2))
        cur = rng.standard_normal((50, 2))

        result = MMD.compute(ref, cur, method="ctt", bandwidth=2.0, seed=1)

        assert "statistic" in result

    def test_different_sample_sizes(self) -> None:
        """Unequal sample sizes should work."""
        rng = np.random.default_rng(10)
        ref = rng.standard_normal((80, 2))
        cur = rng.standard_normal((120, 2))

        result = MMD.compute(ref, cur, method="ctt", seed=10)

        assert "statistic" in result


class TestMMDRFF:
    """Direct tests for MMD RFF-specific behavior."""

    def test_return_structure(self) -> None:
        """RFF returns all expected keys with correct types."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 3))
        cur = rng.standard_normal((100, 3))

        result = MMD.compute(ref, cur, method="rff", seed=42)

        assert set(result.keys()) == {
            "statistic",
            "p_value",
            "threshold",
            "alpha",
            "drift_detected",
        }

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        rng = np.random.default_rng(99)
        ref = rng.standard_normal((100, 4))
        cur = rng.standard_normal((100, 4))

        r1 = MMD.compute(ref, cur, method="rff", seed=7)
        r2 = MMD.compute(ref, cur, method="rff", seed=7)

        assert r1["statistic"] == r2["statistic"]
        assert r1["p_value"] == r2["p_value"]

    def test_drift_with_large_shift(self) -> None:
        """Large mean shift should be detected."""
        rng = np.random.default_rng(77)
        ref = rng.standard_normal((200, 3))
        cur = rng.standard_normal((200, 3)) + 5.0

        result = MMD.compute(ref, cur, method="rff", seed=77)

        assert result["drift_detected"] is True

    def test_no_drift_identical(self) -> None:
        """Identical distributions should not flag drift."""
        rng = np.random.default_rng(55)
        data = rng.standard_normal((200, 3))
        ref = data[:100]
        cur = data[100:]

        result = MMD.compute(ref, cur, method="rff", seed=55)

        assert result["drift_detected"] is False

    def test_custom_num_features(self) -> None:
        """Custom num_features parameter works."""
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((200, 3))
        cur = rng.standard_normal((200, 3))

        result = MMD.compute(ref, cur, method="rff", num_features=50, seed=0)

        assert "statistic" in result


class TestMMDACTT:
    """Direct tests for MMD ACTT (aggregated) behavior."""

    def test_return_structure(self) -> None:
        """ACTT returns all expected keys with correct types."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 3))
        cur = rng.standard_normal((100, 3))

        result = MMD.compute(
            ref,
            cur,
            method="actt",
            seed=42,
            num_permutations=39,
            num_bandwidths=5,
            b2=20,
            b3=5,
        )

        assert set(result.keys()) == {
            "statistic",
            "p_value",
            "threshold",
            "alpha",
            "drift_detected",
        }

    def test_drift_with_large_shift(self) -> None:
        """Large mean shift should be detected."""
        rng = np.random.default_rng(77)
        ref = rng.standard_normal((200, 3))
        cur = rng.standard_normal((200, 3)) + 5.0

        result = MMD.compute(
            ref,
            cur,
            method="actt",
            seed=77,
            num_permutations=299,
            num_bandwidths=5,
            b2=200,
            b3=20,
        )

        assert result["drift_detected"] is True

    def test_no_drift_identical(self) -> None:
        """Identical distributions should not flag drift."""
        rng = np.random.default_rng(55)
        data = rng.standard_normal((200, 3))
        ref = data[:100]
        cur = data[100:]

        result = MMD.compute(
            ref,
            cur,
            method="actt",
            seed=55,
            num_permutations=299,
            num_bandwidths=5,
            b2=200,
            b3=20,
        )

        assert result["drift_detected"] is False


class TestMMDDefaultMethod:
    """Tests that the default method (ctt) works without explicit method=."""

    def test_default_is_ctt(self) -> None:
        """Default method should work (CTT)."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((100, 3))
        cur = rng.standard_normal((100, 3))

        result = MMD.compute(ref, cur, seed=42)

        assert "statistic" in result


class TestMMDValidation:
    """Input validation tests for MMD."""

    def test_1d_input_raises(self) -> None:
        """1D arrays should raise ValueError."""
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([4.0, 5.0, 6.0])

        with pytest.raises(ValueError, match="2-dimensional"):
            MMD.compute(ref, cur)

    def test_mismatched_features_raises(self) -> None:
        """Different feature counts should raise ValueError."""
        ref = np.random.default_rng(0).standard_normal((50, 3))
        cur = np.random.default_rng(0).standard_normal((50, 5))

        with pytest.raises(ValueError, match="Feature dimensions must match"):
            MMD.compute(ref, cur)

    def test_empty_input_raises(self) -> None:
        """Empty arrays should raise ValueError."""
        ref = np.empty((0, 3))
        cur = np.random.default_rng(0).standard_normal((50, 3))

        with pytest.raises(ValueError, match="cannot be empty"):
            MMD.compute(ref, cur)

    def test_unknown_method_raises(self) -> None:
        """Unknown method string should raise ValueError."""
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((50, 2))
        cur = rng.standard_normal((50, 2))

        with pytest.raises(ValueError, match="Unknown method"):
            MMD.compute(ref, cur, method="nonexistent")
