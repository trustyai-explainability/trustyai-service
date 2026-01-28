# pylint: disable=line-too-long
"""
Jensen–Shannon distance and divergence for measuring distribution similarity.

JS divergence is a symmetric and smoothed version of the Kullback–Leibler divergence,
bounded between 0 and 1 (or 0 and log(2) in nats).
"""

from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import jensenshannon

from . import utils

# Default constants for Jensen-Shannon metric
DEFAULT_STATISTIC: Literal["distance", "divergence"] = "distance"
DEFAULT_THRESHOLD = 0.1
DEFAULT_METHOD: Literal["kde", "hist"] = "kde"

# Re-export utils constants for convenience
DEFAULT_BINS = utils.DEFAULT_BINS
DEFAULT_GRID_POINTS = utils.DEFAULT_GRID_POINTS


class JensenShannon:
    """
    Jensen-Shannon distance and divergence for measuring distribution similarity.

    JS divergence is a symmetric and smoothed version of KL divergence,
    bounded between 0 and 1 (or 0 and log(2) in nats).
    """

    @staticmethod
    def jensenshannon(
        data_ref: np.ndarray,
        data_cur: np.ndarray,
        statistic: Literal["distance", "divergence"] = DEFAULT_STATISTIC,
        threshold: float = DEFAULT_THRESHOLD,
        method: Literal["kde", "hist"] = DEFAULT_METHOD,
        grid_points: int = DEFAULT_GRID_POINTS,
        bins: int = DEFAULT_BINS,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Calculate Jensen-Shannon divergence between distributions using scipy.spatial.distance.jensenshannon.

        :param data_ref: Reference distribution data.
        :param data_cur: Current distribution data.
        :param threshold: Threshold for drift detection (default: 0.1) applied to the
            selected Jensen-Shannon statistic.
        :param statistic: Which Jensen-Shannon quantity to use for drift detection,
            e.g. ``"distance"`` (JS distance as returned by
            :func:`scipy.spatial.distance.jensenshannon`) or ``"divergence"``
            (typically the squared distance).
        :param method: How to estimate the underlying distributions before applying
            Jensen-Shannon; supported values are usually ``"kde"`` for kernel density
            estimation on a fixed grid and ``"hist"`` for histogram-based estimation.
        :param grid_points: Number of grid points used when ``method="kde"``.
        :param bins: Number of histogram bins used when ``method="hist"``.
        :return: Dictionary containing js_divergence and drift_detected
        """
        # Validate statistic parameter
        if statistic not in ("distance", "divergence"):
            raise ValueError(f"statistic must be 'distance' or 'divergence', got '{statistic}'")

        # Generate probability distributions from data on a common grid
        if method == "kde":
            p_ref, p_cur = utils.prob_dist_kde(data_ref, data_cur, grid_points, **kwargs)
        elif method == "hist":
            p_ref, p_cur = utils.prob_dist_hist(data_ref, data_cur, bins, **kwargs)
        else:
            raise ValueError("`method` must be `hist` or `kde`.")

        # The metric
        distance = jensenshannon(p_ref, p_cur)
        divergence = distance**2
        actual = distance if statistic == "distance" else divergence

        return {
            "Jensen-Shannon_distance": distance,
            "Jensen-Shannon_divergence": divergence,
            "drift_detected": bool(actual > threshold),
            "threshold": threshold,
        }
