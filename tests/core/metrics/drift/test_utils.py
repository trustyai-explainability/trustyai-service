import numpy as np
import pytest

from src.core.metrics.drift import utils

# ==============================================================================
# Utils Module Tests
# ==============================================================================


class TestProbDistHist:
    """Tests for histogram-based probability distribution generation."""

    def test_prob_dist_hist_basic(self):
        """Test basic histogram probability distribution generation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        p_x, p_y = utils.prob_dist_hist(x, y, bins=10)

        # Both should be probability distributions (sum to 1)
        assert pytest.approx(p_x.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y.sum(), abs=1e-10) == 1.0

        # Both should have the same length (number of bins)
        assert len(p_x) == 10
        assert len(p_y) == 10

        # All probabilities should be non-negative
        assert np.all(p_x >= 0)
        assert np.all(p_y >= 0)

    def test_prob_dist_hist_identical_distributions(self):
        """Test histogram method with identical distributions."""
        np.random.seed(42)
        x = np.random.normal(loc=0, scale=1, size=100)
        y = x.copy()  # Identical distribution

        p_x, p_y = utils.prob_dist_hist(x, y, bins=20)

        # For identical data, distributions should be identical
        assert pytest.approx(p_x, abs=1e-10) == p_y

    def test_prob_dist_hist_different_sizes(self):
        """Test histogram method with different sample sizes."""
        np.random.seed(123)
        x = np.random.normal(loc=0, scale=1, size=50)
        y = np.random.normal(loc=0, scale=1, size=150)

        p_x, p_y = utils.prob_dist_hist(x, y, bins=15)

        # Both should still be valid probability distributions
        assert pytest.approx(p_x.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y.sum(), abs=1e-10) == 1.0

        # Both should have the same number of bins
        assert len(p_x) == 15
        assert len(p_y) == 15

    def test_prob_dist_hist_empty_data_raises_error(self):
        """Test that empty first array raises ValueError."""
        empty = np.array([])
        nonempty = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="cannot be empty"):
            utils.prob_dist_hist(empty, nonempty, bins=10)
        with pytest.raises(ValueError, match="cannot be empty"):
            utils.prob_dist_hist(nonempty, empty, bins=10)
        with pytest.raises(ValueError, match="cannot be empty"):
            utils.prob_dist_hist(empty, empty, bins=10)

    def test_prob_dist_hist_nan_data_raises_error(self):
        """Test that NaN values in input arrays raise ValueError."""
        x_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        y_valid = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="cannot contain NaN"):
            utils.prob_dist_hist(x_with_nan, y_valid, bins=10)
        with pytest.raises(ValueError, match="cannot contain NaN"):
            utils.prob_dist_hist(y_valid, x_with_nan, bins=10)

    def test_prob_dist_hist_constant_inputs(self):
        """Test histogram probability distributions for constant inputs."""
        x = np.ones(100)
        y = np.ones(100)

        p_x, p_y = utils.prob_dist_hist(x, y, bins=10)

        # Distributions should be properly normalized and well-behaved for singular data
        assert pytest.approx(p_x.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y.sum(), abs=1e-10) == 1.0
        assert np.all(p_x >= 0.0)
        assert np.all(p_y >= 0.0)
        assert np.all(np.isfinite(p_x))
        assert np.all(np.isfinite(p_y))

    def test_prob_dist_hist_different_bins(self):
        """Test histogram method with different bin counts."""
        np.random.seed(456)
        x = np.random.uniform(0, 10, size=100)
        y = np.random.uniform(0, 10, size=100)

        p_x_coarse, p_y_coarse = utils.prob_dist_hist(x, y, bins=10)
        p_x_fine, p_y_fine = utils.prob_dist_hist(x, y, bins=50)

        # Different bin counts should produce different length distributions
        assert len(p_x_coarse) == 10
        assert len(p_x_fine) == 50

        # Both should still sum to 1
        assert pytest.approx(p_x_coarse.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_x_fine.sum(), abs=1e-10) == 1.0

    def test_prob_dist_hist_overlapping_ranges(self):
        """Test histogram with overlapping data ranges."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        p_x, p_y = utils.prob_dist_hist(x, y, bins=20)

        # Common range should be [0, 6]
        # Both distributions should sum to 1
        assert pytest.approx(p_x.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y.sum(), abs=1e-10) == 1.0


class TestProbDistKDE:
    """Tests for KDE-based probability distribution generation."""

    def test_prob_dist_kde_basic(self):
        """Test basic KDE probability distribution generation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        p_x, p_y = utils.prob_dist_kde(x, y, grid_points=100)

        # Both should be probability distributions (sum to 1)
        assert pytest.approx(p_x.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y.sum(), abs=1e-10) == 1.0

        # Both should have the same length (number of grid points)
        assert len(p_x) == 100
        assert len(p_y) == 100

        # All probabilities should be non-negative
        assert np.all(p_x >= 0)
        assert np.all(p_y >= 0)

    def test_prob_dist_kde_constant_inputs_raises_error(self):
        """Test that KDE raises error for constant (zero-variance) inputs."""
        x = np.ones(100)
        y = np.ones(100)

        # KDE should raise ValueError for constant inputs
        with pytest.raises(ValueError, match="zero variance"):
            utils.prob_dist_kde(x, y, grid_points=100)

    def test_prob_dist_kde_identical_distributions(self):
        """Test KDE method with identical distributions."""
        np.random.seed(42)
        x = np.random.normal(loc=0, scale=1, size=100)
        y = x.copy()  # Identical distribution

        p_x, p_y = utils.prob_dist_kde(x, y, grid_points=200)

        # For identical data, distributions should be very similar
        # (may not be exactly identical due to KDE evaluation)
        assert pytest.approx(p_x, abs=1e-5) == p_y

    def test_prob_dist_kde_different_sizes(self):
        """Test KDE method with different sample sizes."""
        np.random.seed(789)
        x = np.random.normal(loc=0, scale=1, size=50)
        y = np.random.normal(loc=0, scale=1, size=150)

        p_x, p_y = utils.prob_dist_kde(x, y, grid_points=256)

        # Both should still be valid probability distributions
        assert pytest.approx(p_x.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y.sum(), abs=1e-10) == 1.0

        # Both should have the same number of grid points
        assert len(p_x) == 256
        assert len(p_y) == 256

    def test_prob_dist_kde_empty_data_raises_error(self):
        """Test that empty first array raises ValueError."""
        empty = np.array([])
        nonempty = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="cannot be empty"):
            utils.prob_dist_kde(empty, nonempty, bins=10)
        with pytest.raises(ValueError, match="cannot be empty"):
            utils.prob_dist_kde(nonempty, empty, bins=10)
        with pytest.raises(ValueError, match="cannot be empty"):
            utils.prob_dist_kde(empty, empty, bins=10)

    def test_prob_dist_kde_nan_data_raises_error(self):
        """Test that NaN values in input arrays raise ValueError."""
        x_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        y_valid = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(ValueError, match="cannot contain NaN"):
            utils.prob_dist_kde(x_with_nan, y_valid, grid_points=100)
        with pytest.raises(ValueError, match="cannot contain NaN"):
            utils.prob_dist_kde(y_valid, x_with_nan, grid_points=100)

    def test_prob_dist_kde_small_sample_raises_error(self):
        """Test that sample size < 2 raises ValueError."""
        single_point = np.array([1.0])
        multiple_points = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="needs at least 2 points"):
            utils.prob_dist_kde(single_point, multiple_points, grid_points=100)
        with pytest.raises(ValueError, match="needs at least 2 points"):
            utils.prob_dist_kde(multiple_points, single_point, grid_points=100)

    def test_prob_dist_kde_different_grid_points(self):
        """Test KDE method with different grid resolutions."""
        np.random.seed(999)
        x = np.random.uniform(0, 10, size=100)
        y = np.random.uniform(0, 10, size=100)

        p_x_coarse, p_y_coarse = utils.prob_dist_kde(x, y, grid_points=50)
        p_x_fine, p_y_fine = utils.prob_dist_kde(x, y, grid_points=500)

        # Different grid resolutions should produce different length distributions
        assert len(p_x_coarse) == 50
        assert len(p_x_fine) == 500

        # Both should still sum to 1
        assert pytest.approx(p_x_coarse.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_x_fine.sum(), abs=1e-10) == 1.0

    def test_prob_dist_kde_overlapping_ranges(self):
        """Test KDE with overlapping data ranges."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        p_x, p_y = utils.prob_dist_kde(x, y, grid_points=256)

        # Common range should be [0, 6]
        # Both distributions should sum to 1
        assert pytest.approx(p_x.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y.sum(), abs=1e-10) == 1.0

    def test_prob_dist_kde_custom_bandwidth(self):
        """Test KDE with custom bandwidth parameter."""
        np.random.seed(555)
        x = np.random.normal(loc=0, scale=1, size=100)
        y = np.random.normal(loc=1, scale=1, size=100)

        # Test with custom bandwidth method
        p_x_scott, p_y_scott = utils.prob_dist_kde(x, y, grid_points=256, bw_method="scott")
        p_x_silverman, p_y_silverman = utils.prob_dist_kde(x, y, grid_points=256, bw_method="silverman")

        # Both should produce valid distributions
        assert pytest.approx(p_x_scott.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_x_silverman.sum(), abs=1e-10) == 1.0

        # Different bandwidth methods should produce different results
        # (not testing exact difference, just that both work)
        assert len(p_x_scott) == 256
        assert len(p_x_silverman) == 256

    def test_prob_dist_kde_well_separated_distributions(self):
        """Test KDE with well-separated distributions."""
        np.random.seed(321)
        x = np.random.normal(loc=0, scale=1, size=100)
        y = np.random.normal(loc=10, scale=1, size=100)

        p_x, p_y = utils.prob_dist_kde(x, y, grid_points=512)

        # Both should be valid probability distributions
        assert pytest.approx(p_x.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y.sum(), abs=1e-10) == 1.0

        # Distributions should be different (low correlation)
        correlation = np.corrcoef(p_x, p_y)[0, 1]
        assert correlation < 0.5, "Well-separated distributions should have low correlation"


class TestComparisonHistVsKDE:
    """Tests comparing histogram and KDE methods."""

    def test_both_methods_produce_valid_distributions(self):
        """Test that both methods produce valid probability distributions."""
        np.random.seed(42)
        x = np.random.normal(loc=0, scale=1, size=100)
        y = np.random.normal(loc=1, scale=1, size=100)

        # Histogram method
        p_x_hist, p_y_hist = utils.prob_dist_hist(x, y, bins=64)

        # KDE method
        p_x_kde, p_y_kde = utils.prob_dist_kde(x, y, grid_points=64)

        # All should be valid probability distributions
        assert pytest.approx(p_x_hist.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y_hist.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_x_kde.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_y_kde.sum(), abs=1e-10) == 1.0

    def test_both_methods_handle_small_samples(self):
        """Test that both methods handle small sample sizes."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.5, 2.5, 3.5])

        # Histogram method
        p_x_hist, p_y_hist = utils.prob_dist_hist(x, y, bins=10)

        # KDE method
        p_x_kde, p_y_kde = utils.prob_dist_kde(x, y, grid_points=10)

        # Both should produce valid distributions despite small samples
        assert pytest.approx(p_x_hist.sum(), abs=1e-10) == 1.0
        assert pytest.approx(p_x_kde.sum(), abs=1e-10) == 1.0
