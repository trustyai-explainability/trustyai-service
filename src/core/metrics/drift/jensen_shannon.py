# pylint: disable=line-too-long
"""
Jensen–Shannon distance and divergence for measuring distribution similarity.

JS divergence is a symmetric and smoothed version of the Kullback–Leibler divergence,
bounded between 0 and 1 (or 0 and log(2) in nats).
"""

from typing import Literal

import numpy as np
from scipy.spatial.distance import jensenshannon

from . import utils


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
        statistic: Literal["distance", "divergence"] = "distance",
        threshold: float = 0.1,
        method: Literal["kde", "hist"] = "kde",
        grid_points: int = 256,
        bins: int = 64,
        **kwargs,
    ) -> dict[str, float]:
        """
        Calculate Jensen-Shannon divergence between distributions using scipy.spatial.distance.jensenshannon.

        :param data_ref: Reference distribution data
        :param data_cur: Current distribution data
        :param threshold: Threshold for drift detection (default: 0.1)
        :param grid_points: Number of grid points for the kde sampling
        :param method: Whether to use distance or divergence for drift detection
        :return: Dictionary containing js_divergence and drift_detected
        """

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
            "Jensen–Shannon_distance": distance,
            "Jensen–Shannon_divergence": divergence,
            "drift_detected": bool(actual > threshold),
            "threshold": threshold,
        }
