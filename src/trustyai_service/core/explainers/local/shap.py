"""KernelSHAP computation: surrogate training and Shapley value estimation.

Pure computation module — no FastAPI, no storage, no Pydantic.
All parameters are primitive types (str, int, float) or numpy arrays.
Enum-to-string mapping is the caller's responsibility.
"""

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

_DEFAULT_BOOTSTRAP_SAMPLES = 50
_BOOTSTRAP_CENTROID_CAP_FACTOR = 0.5


def _compute_n_centroids(background_len: int, n_samples: int) -> int:
    """Compute number of k-means centroids, capped to avoid convergence failures.

    Args:
        background_len: Size of background sample.
        n_samples: Requested number of samples/centroids.

    Returns:
        Capped centroid count: min(n_samples, max(1, half of sample size)).

    """
    return min(n_samples, max(1, int(background_len * _BOOTSTRAP_CENTROID_CAP_FACTOR)))


def build_surrogate(inputs: np.ndarray, outputs: np.ndarray) -> BaseEstimator:
    """Train a surrogate model on stored organic observations.

    Detects output type: floating-point dtype → RandomForestRegressor,
    otherwise → RandomForestClassifier.

    """
    if np.issubdtype(outputs.dtype, np.floating):
        surrogate: BaseEstimator = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
    else:
        surrogate = RandomForestClassifier(n_estimators=100, random_state=42)
    return surrogate.fit(inputs, outputs)


def _normalize_shap_output(raw: Any) -> np.ndarray:  # noqa: ANN401
    """Flatten raw KernelSHAP output to a 1-D array.

    Handles both single-output (ndarray) and multi-output (list of ndarray)
    returns from ``shap.KernelExplainer.shap_values``.

    """
    return np.asarray(raw).flatten()


def compute_shap_values(
    instance: np.ndarray,
    background: np.ndarray,
    surrogate: BaseEstimator,
    n_samples: int,
    link: str,
    l1_reg: str,
) -> np.ndarray:
    """Compute KernelSHAP attributions for a single instance.

    Summarises the background with k-means (``n_samples`` centroids),
    instantiates a ``shap.KernelExplainer`` against the surrogate, and returns
    a 1-D array of Shapley values — one per input feature.

    The three Shapley axioms satisfied by KernelSHAP:

    - **Local accuracy**: attributions sum exactly to ``f(x) - E[f(X)]``.
    - **Missingness**: features absent in the instance contribute zero
      (achieved approximately via L1 regularization).
    - **Consistency**: a feature whose marginal contribution increases
      regardless of coalition receives a higher attribution.

    Attributions explain the *surrogate* trained on stored predictions,
    not the original model directly.

    Args:
        instance: 1-D input vector to explain.
        background: 2-D array of background observations.
        surrogate: Fitted sklearn estimator.
        n_samples: Number of background centroids and KernelSHAP samples.
        link: Link function string for KernelExplainer (``"identity"`` or ``"logit"``).
        l1_reg: L1 regularization string (e.g. ``"bic"``, ``"num_features(10)"``).

    """
    n_bg = _compute_n_centroids(len(background), n_samples)
    if n_bg < n_samples:
        logger.warning(
            "Background size (%d) smaller than n_samples (%d); using %d centroids.",
            len(background),
            n_samples,
            n_bg,
        )
    bg_summary = shap.kmeans(background, n_bg)
    explainer = shap.KernelExplainer(surrogate.predict, bg_summary, link=link)
    raw = explainer.shap_values(
        instance.reshape(1, -1),
        nsamples=n_samples,
        l1_reg=l1_reg,
        silent=True,
    )
    return _normalize_shap_output(raw)


def compute_confidence_intervals(
    instance: np.ndarray,
    background: np.ndarray,
    surrogate: BaseEstimator,
    confidence: float,
    n_samples: int,
    link: str,
    l1_reg: str,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute bootstrap confidence intervals for SHAP attributions.

    Resamples ``background`` with replacement ``n_bootstrap`` times, refits
    a ``KernelExplainer`` each time, and returns percentile bounds at
    ``(1-confidence)/2`` and ``1-(1-confidence)/2``.

    Returns ``(None, None)`` immediately when ``confidence >= 1.0``
    (the sentinel value that disables interval computation), or when all
    bootstrap iterations fail due to degenerate samples.

    CI bounds are stochastic estimates; pass ``seed`` for reproducibility.

    Args:
        instance: 1-D input vector to explain.
        background: 2-D array of background observations.
        surrogate: Fitted sklearn estimator.
        confidence: Coverage level (e.g. 0.95 for 95% CI). Use 1.0 to skip.
        n_samples: KernelSHAP sample count per bootstrap iteration.
        link: Link function string (``"identity"`` or ``"logit"``).
        l1_reg: L1 regularization string.
        n_bootstrap: Number of bootstrap resamples.
        seed: Optional RNG seed for reproducibility.

    """
    if confidence >= 1.0:  # sentinel: 1.0 disables CI computation
        return None, None

    alpha = (1.0 - confidence) / 2.0
    rng = np.random.default_rng(seed)

    bootstrap_values: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(background), size=len(background))
        bg_sample = background[idx]
        n_bg = _compute_n_centroids(len(bg_sample), n_samples)
        try:
            bg_summary = shap.kmeans(bg_sample, n_bg)
            boot_explainer = shap.KernelExplainer(
                surrogate.predict, bg_summary, link=link
            )
            raw = boot_explainer.shap_values(
                instance.reshape(1, -1),
                nsamples=n_samples,
                l1_reg=l1_reg,
                silent=True,
            )
            bootstrap_values.append(_normalize_shap_output(raw))
        except (ValueError, RuntimeError):
            logger.debug(
                "Skipping degenerate bootstrap sample (k-means convergence failure)."
            )

    if not bootstrap_values:
        logger.warning(
            "All %d bootstrap iterations failed; confidence intervals unavailable.",
            n_bootstrap,
        )
        return None, None

    boot_arr = np.array(bootstrap_values)
    lower: np.ndarray = np.percentile(boot_arr, alpha * 100, axis=0)
    upper: np.ndarray = np.percentile(boot_arr, (1.0 - alpha) * 100, axis=0)
    return lower, upper
