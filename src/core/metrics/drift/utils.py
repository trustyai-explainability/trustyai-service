# pylint: disable=line-too-long
"""
Utility functions for drift detection metrics.

This module provides common utility functions for computing probability distributions
from data samples using different methods (histogram and kernel density estimation).
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from scipy.stats import gaussian_kde

# Default constants for probability distribution estimation
DEFAULT_BINS = 64
DEFAULT_GRID_POINTS = 256

# Type alias for gaussian_kde bandwidth selection methods.
# Supports named methods, scalar factors, None (default), and callables
# that receive a gaussian_kde instance and return a bandwidth factor.
BwMethod = Literal["scott", "silverman"] | float | Callable[[Any], float] | None


def prob_dist_hist(x: np.ndarray, y: np.ndarray, bins: int = DEFAULT_BINS) -> np.ndarray:
    """
    Generate probability distributions from two datasets using histograms.

    Creates probability mass functions by binning the data on a common grid
    that spans the range of both datasets.

    :param x: First dataset (reference distribution)
    :param y: Second dataset (current distribution)
    :param bins: Number of bins for histogram (default: 64)
    :return: List containing two normalized probability distributions [p_x, p_y]
    :raises ValueError: If either input array is empty or contains NaN values
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Input arrays cannot contain NaN values")

    # Evaluate on common grid
    x_min = min(x.min(), y.min())
    x_max = max(x.max(), y.max())
    x_range = (x_min, x_max)

    p_x, _ = np.histogram(x, bins=bins, range=x_range, density=True)
    p_y, _ = np.histogram(y, bins=bins, range=x_range, density=True)

    # Convert density to probability mass
    # Guard against division by zero if all bins are zero
    p_x_sum = p_x.sum()
    p_y_sum = p_y.sum()
    if p_x_sum == 0:
        raise ValueError("Reference distribution has zero total probability (all bins are zero)")
    if p_y_sum == 0:
        raise ValueError("Current distribution has zero total probability (all bins are zero)")
    p_x /= p_x_sum
    p_y /= p_y_sum

    return [p_x, p_y]


def prob_dist_kde(
    x: np.ndarray,
    y: np.ndarray,
    grid_points: int = DEFAULT_GRID_POINTS,
    *,
    bw_method: BwMethod = None,
) -> np.ndarray:
    """
    Generate probability distributions from two datasets using kernel density estimation.

    Estimates continuous probability density functions using Gaussian kernels and
    evaluates them on a common grid spanning the range of both datasets.

    :param x: First dataset (reference distribution)
    :param y: Second dataset (current distribution)
    :param grid_points: Number of points in the evaluation grid (default: 256)
    :param bw_method: Bandwidth selection method passed to scipy.stats.gaussian_kde
    :return: List containing two normalized probability distributions [p_x, p_y]
    :raises ValueError: If either input array is empty, contains NaN values, or has insufficient
                        sample size for KDE estimation
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("Input arrays cannot contain NaN values")

    # Validate sample size for KDE (needs at least 2 points and non-zero variance)
    if x.size < 2:
        raise ValueError(f"Reference distribution needs at least 2 points for KDE, got {x.size}")
    if y.size < 2:
        raise ValueError(f"Current distribution needs at least 2 points for KDE, got {y.size}")

    # Check for zero variance (constant values) which can cause KDE to fail
    if np.var(x) == 0:
        raise ValueError("Reference distribution has zero variance (all values are constant), cannot use KDE")
    if np.var(y) == 0:
        raise ValueError("Current distribution has zero variance (all values are constant), cannot use KDE")

    # Evaluate on common grid
    x_min = min(x.min(), y.min())
    x_max = max(x.max(), y.max())
    x_range = np.linspace(x_min, x_max, grid_points)

    # Estimate continuous distributions using kernel density estimate with Gaussian kernels
    kde_x = gaussian_kde(x, bw_method=bw_method)
    kde_y = gaussian_kde(y, bw_method=bw_method)

    # Generate probability distributions
    p_x = kde_x(x_range)
    p_y = kde_y(x_range)

    # Guard against division by zero if all densities are zero
    p_x_sum = p_x.sum()
    p_y_sum = p_y.sum()
    if p_x_sum == 0:
        raise ValueError("Reference distribution has zero total probability (all densities are zero)")
    if p_y_sum == 0:
        raise ValueError("Current distribution has zero total probability (all densities are zero)")
    p_x /= p_x_sum
    p_y /= p_y_sum

    return [p_x, p_y]
