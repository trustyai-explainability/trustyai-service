# pylint: disable=line-too-long
"""
Utility functions for drift detection metrics.

This module provides common utility functions for computing probability distributions
from data samples using different methods (histogram and kernel density estimation).
"""

import numpy as np
from scipy.stats import gaussian_kde


def prob_dist_hist(x: np.ndarray, y: np.ndarray, bins: int = 64, **kwargs) -> np.ndarray:
    """
    Generate probability distributions from two datasets using histograms.

    Creates probability mass functions by binning the data on a common grid
    that spans the range of both datasets.

    :param x: First dataset (reference distribution)
    :param y: Second dataset (current distribution)
    :param bins: Number of bins for histogram (default: 64)
    :param kwargs: Additional keyword arguments (currently unused, for future extensibility)
    :return: List containing two normalized probability distributions [p_x, p_y]
    :raises ValueError: If either input array is empty
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Evaluate on common grid
    x_min = min(x.min(), y.min())
    x_max = max(x.max(), y.max())
    x_range = (x_min, x_max)

    p_x, _ = np.histogram(x, bins=bins, range=x_range, density=True)
    p_y, _ = np.histogram(y, bins=bins, range=x_range, density=True)

    # Convert density to probability mass
    p_x /= p_x.sum()
    p_y /= p_y.sum()

    return [p_x, p_y]


def prob_dist_kde(x: np.ndarray, y: np.ndarray, grid_points: int = 256, **kwargs) -> np.ndarray:
    """
    Generate probability distributions from two datasets using kernel density estimation.

    Estimates continuous probability density functions using Gaussian kernels and
    evaluates them on a common grid spanning the range of both datasets.

    :param x: First dataset (reference distribution)
    :param y: Second dataset (current distribution)
    :param grid_points: Number of points in the evaluation grid (default: 256)
    :param kwargs: Additional keyword arguments passed to scipy.stats.gaussian_kde
                   (e.g., bw_method for bandwidth selection)
    :return: List containing two normalized probability distributions [p_x, p_y]
    :raises ValueError: If either input array is empty
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Evaluate on common grid
    x_min = min(x.min(), y.min())
    x_max = max(x.max(), y.max())
    x_range = np.linspace(x_min, x_max, grid_points)

    # Estimate continuous distributions using kernel density estimate with Gaussian kernels
    kde_x = gaussian_kde(x, **kwargs)
    kde_y = gaussian_kde(y, **kwargs)

    # Generate probability distributions
    p_x = kde_x(x_range)
    p_y = kde_y(x_range)

    p_x /= p_x.sum()
    p_y /= p_y.sum()

    return [p_x, p_y]
