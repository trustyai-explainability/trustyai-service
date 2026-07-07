"""Multivariate two-sample drift detection via Maximum Mean Discrepancy (MMD).

Dispatches to concrete implementations via the ``method`` parameter.
Uses the ``goodpoints`` package (Microsoft Research) as the computational
backend.

Currently supported methods:

* ``"ctt"`` — Compress Then Test: near-linear time with no approximation
  error via kernel thinning (Domingo-Enrich, Dwivedi & Mackey, AISTATS 2023).
* ``"rff"`` — Random Fourier Features: approximates the kernel via random
  feature maps and runs a permutation test (Rahimi & Recht, NeurIPS 2007).
* ``"actt"`` — Aggregated Compress Then Test: CTT with MMDAgg-style
  aggregation over multiple bandwidths (combines Domingo-Enrich et al.
  with Schrab et al., JMLR 2023).

References:
    - Domingo-Enrich, Dwivedi & Mackey.
      "Compress Then Test." AISTATS, 2023. arXiv:2301.05974.
    - Rahimi & Recht.
      "Random Features for Large-Scale Kernel Machines." NeurIPS, 2007.

"""

import logging
from typing import Any, Literal, TypedDict

import numpy as np
from goodpoints import ctt as _goodpoints_ctt

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.05
DEFAULT_NUM_PERMUTATIONS = 39
DEFAULT_NUM_FEATURES = 100
DEFAULT_BANDWIDTH = 1.0
DEFAULT_KERNEL = "gauss"

# CTT-specific defaults
DEFAULT_COMPRESSION_LEVEL = 0
DEFAULT_NUM_BINS = 16

# ACTT (aggregated) defaults
DEFAULT_ACTT_NUM_PERMUTATIONS = 299
DEFAULT_ACTT_B2 = 200
DEFAULT_ACTT_B3 = 20
DEFAULT_NUM_BANDWIDTHS = 10

Method = Literal["ctt", "rff", "actt"]


class MMDResult(TypedDict):
    """Result of an MMD two-sample test."""

    statistic: float
    p_value: float
    threshold: float
    alpha: float
    drift_detected: bool


def _build_result(result: Any, num_permutations: int, alpha: float) -> MMDResult:  # noqa: ANN401
    """Build MMDResult from a goodpoints TestResults object."""
    null_stats = result.estimator_values[:-1]
    observed = result.estimator_values[-1]
    p_value = float((np.sum(null_stats >= observed) + 1) / (num_permutations + 1))

    return {
        "statistic": float(observed),
        "p_value": p_value,
        "threshold": float(result.threshold_values),
        "alpha": alpha,
        "drift_detected": bool(result.rejects),
    }


def _mmd_rff(  # noqa: PLR0913
    reference_data: np.ndarray,
    current_data: np.ndarray,
    *,
    alpha: float,
    seed: int | None,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    num_features: int = DEFAULT_NUM_FEATURES,
    bandwidth: float = DEFAULT_BANDWIDTH,
    kernel: str = DEFAULT_KERNEL,
    **_kwargs: Any,  # noqa: ANN401
) -> MMDResult:
    """Run RFF-MMD permutation test via goodpoints."""
    if _kwargs:
        logger.warning("method='rff' ignoring unknown kwargs: %s", sorted(_kwargs))
    result = _goodpoints_ctt.rff(
        reference_data,
        current_data,
        num_features,
        B=num_permutations,
        lam=bandwidth,
        kernel=kernel,
        alpha=alpha,
        null_seed=seed,
        statistic_seed=seed,
    )

    return _build_result(result, num_permutations, alpha)


def _mmd_ctt(  # noqa: PLR0913
    reference_data: np.ndarray,
    current_data: np.ndarray,
    *,
    alpha: float,
    seed: int | None,
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
    bandwidth: float = DEFAULT_BANDWIDTH,
    kernel: str = DEFAULT_KERNEL,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    num_bins: int = DEFAULT_NUM_BINS,
    **_kwargs: Any,  # noqa: ANN401
) -> MMDResult:
    """Run Compress Then Test via goodpoints."""
    if _kwargs:
        logger.warning("method='ctt' ignoring unknown kwargs: %s", sorted(_kwargs))
    result = _goodpoints_ctt.ctt(
        reference_data,
        current_data,
        compression_level,
        B=num_permutations,
        s=num_bins,
        lam=bandwidth,
        kernel=kernel,
        alpha=alpha,
        null_seed=seed,
        statistic_seed=seed,
    )

    return _build_result(result, num_permutations, alpha)


def _mmd_actt(  # noqa: PLR0913
    reference_data: np.ndarray,
    current_data: np.ndarray,
    *,
    alpha: float,
    seed: int | None,
    num_permutations: int = DEFAULT_ACTT_NUM_PERMUTATIONS,
    bandwidth: float = DEFAULT_BANDWIDTH,
    kernel: str = DEFAULT_KERNEL,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    num_bins: int = DEFAULT_NUM_BINS,
    num_bandwidths: int = DEFAULT_NUM_BANDWIDTHS,
    b2: int = DEFAULT_ACTT_B2,
    b3: int = DEFAULT_ACTT_B3,
    **_kwargs: Any,  # noqa: ANN401
) -> MMDResult:
    """Run Aggregated CTT with MMDAgg-style bandwidth aggregation via goodpoints."""
    if _kwargs:
        logger.warning("method='actt' ignoring unknown kwargs: %s", sorted(_kwargs))
    ctt = _goodpoints_ctt

    lam = bandwidth * np.exp(np.linspace(-np.log(4.0), np.log(4.0), num_bandwidths))
    weights = np.full(num_bandwidths, 1.0 / num_bandwidths)

    result = ctt.actt(
        reference_data,
        current_data,
        compression_level,
        B=num_permutations,
        B_2=b2,
        B_3=b3,
        s=num_bins,
        lam=lam,
        weights=weights,
        kernel=kernel,
        alpha=alpha,
        null_seed=seed,
        statistic_seed=seed,
    )

    stats: dict[float, float] = result.statistic_values
    thresholds: dict[float, float] = result.threshold_values
    best_bw = max(stats, key=lambda bw: stats[bw])
    max_stat = float(stats[best_bw])

    # Permutation p-value for the best bandwidth's statistic
    estimators = result.all_estimator_values[best_bw]
    null_stats = estimators[: estimators.shape[0] - 1]
    p_value = float((np.sum(null_stats >= max_stat) + 1) / (len(null_stats) + 1))

    return {
        "statistic": max_stat,
        "p_value": p_value,
        "threshold": float(thresholds[best_bw]),
        "alpha": alpha,
        "drift_detected": bool(result.rejects),
    }


class MMD:
    """Multivariate two-sample drift test using Maximum Mean Discrepancy.

    Dispatches to a concrete algorithm via ``method``.  Method-specific
    parameters are forwarded as ``**kwargs``.
    """

    @staticmethod
    def compute(
        reference_data: np.ndarray,
        current_data: np.ndarray,
        *,
        method: Method = "ctt",
        alpha: float = DEFAULT_ALPHA,
        seed: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> MMDResult:
        """Run an MMD two-sample test.

        :param reference_data: Reference distribution, shape ``(n, d)``
        :param current_data: Current distribution, shape ``(m, d)``
        :param method: ``"ctt"`` (Compress Then Test), ``"rff"``
            (Random Fourier Features), or ``"actt"`` (Aggregated CTT)
        :param alpha: Significance level
        :param seed: Random seed for reproducibility
        :param kwargs: Forwarded to the chosen method implementation
        :return: ``MMDResult`` with ``statistic``, ``p_value``,
            ``threshold``, ``alpha``, and ``drift_detected``
        """
        reference_data = np.asarray(reference_data, dtype=np.float64)
        current_data = np.asarray(current_data, dtype=np.float64)

        if reference_data.ndim != 2 or current_data.ndim != 2:  # noqa: PLR2004
            msg = "Input arrays must be 2-dimensional (n_samples, n_features)"
            raise ValueError(msg)
        if reference_data.shape[1] != current_data.shape[1]:
            msg = (
                f"Feature dimensions must match: reference has "
                f"{reference_data.shape[1]}, current has {current_data.shape[1]}"
            )
            raise ValueError(msg)
        if reference_data.shape[0] == 0 or current_data.shape[0] == 0:
            msg = "Input arrays cannot be empty"
            raise ValueError(msg)

        if not 0 < alpha < 1:
            msg = f"alpha must be in (0, 1), got {alpha}"
            raise ValueError(msg)

        supported: set[Method] = {"ctt", "rff", "actt"}
        if method not in supported:
            msg = f"Unknown method {method!r}. Supported: {sorted(supported)}"
            raise ValueError(msg)

        if method == "rff":
            return _mmd_rff(
                reference_data, current_data, alpha=alpha, seed=seed, **kwargs
            )
        if method == "actt":
            return _mmd_actt(
                reference_data, current_data, alpha=alpha, seed=seed, **kwargs
            )
        return _mmd_ctt(reference_data, current_data, alpha=alpha, seed=seed, **kwargs)
