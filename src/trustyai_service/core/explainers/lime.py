"""LIME computation: local interpretable model-agnostic explanations.

Pure computation module — no FastAPI, no storage, no Pydantic.
All parameters are primitive types (str, int, float, np.ndarray).
Enum-to-string mapping is the caller's responsibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from lime.lime_tabular import LimeTabularExplainer

    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_lime_explainer(
    training_data: np.ndarray,
    feature_names: list[str],
    mode: str,
    kernel_width: float = 0.75,
    *,
    discretize_continuous: bool = True,
    seed: int | None = None,
) -> LimeTabularExplainer:
    """Create a LIME explainer from training data.

    Args:
        training_data: 2-D array of background observations (n_samples, n_features).
        feature_names: Names of input features.
        mode: Explanation mode ("regression" or "classification").
        kernel_width: Kernel width for proximity weighting (default 0.75).
        discretize_continuous: Whether to discretize continuous features (default True).
        seed: Random seed for reproducibility.

    Returns:
        Configured LimeTabularExplainer instance.

    """
    return LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        mode=mode,
        kernel_width=kernel_width,
        discretize_continuous=discretize_continuous,
        random_state=seed,
    )


def compute_lime_explanation(
    explainer: LimeTabularExplainer,
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    num_samples: int = 5000,
    num_features: int = 10,
) -> tuple[list[tuple[str, float]], float, float, float]:
    """Compute LIME explanation for a single instance.

    Args:
        explainer: Fitted LimeTabularExplainer.
        instance: 1-D input vector to explain.
        predict_fn: Callable that maps instances to predictions.
        num_samples: Number of perturbation samples (default 5000).
        num_features: Number of top features to return (default 10).

    Returns:
        Tuple of:
        - feature_weights: list of (feature_name, importance) pairs
        - r2_score: R² of the local linear model (0 to 1)
        - local_pred: predicted value for the instance
        - intercept: intercept of the local linear model

    """
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_samples=num_samples,
        num_features=num_features,
    )

    feature_weights = explanation.as_list()
    r2_score = float(explanation.score)
    local_pred = float(explanation.local_pred[0])
    # For classification, intercept is dict {label: value}; for regression, scalar
    if isinstance(explanation.intercept, dict):
        intercept = float(next(iter(explanation.intercept.values())))
    else:
        intercept = float(explanation.intercept)

    return feature_weights, r2_score, local_pred, intercept


def compute_lime_confidence_intervals(
    training_data: np.ndarray,
    feature_names: list[str],
    mode: str,
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    confidence: float,
    num_samples: int = 5000,
    num_features: int = 10,
    kernel_width: float = 0.75,
    n_bootstrap: int = 50,
    seed: int | None = None,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Compute bootstrap confidence intervals for LIME explanations.

    Resamples the training data ``n_bootstrap`` times, recomputes explanation
    each time, and returns percentile bounds per feature. Always uses
    ``discretize_continuous=False`` to keep feature names stable across resamples,
    matching the main explanation's feature names.

    Returns ``(None, None)`` when ``confidence >= 1.0`` (sentinel to skip CI),
    or when all bootstrap iterations fail.

    Args:
        training_data: 2-D array of background observations.
        feature_names: Names of input features.
        mode: Explanation mode ("regression" or "classification").
        instance: 1-D input vector to explain.
        predict_fn: Callable that maps instances to predictions.
        confidence: Coverage level (e.g. 0.95 for 95% CI). Use 1.0 to skip.
        num_samples: LIME sample count per bootstrap iteration.
        num_features: Number of top features to return.
        kernel_width: Kernel width for proximity weighting.
        n_bootstrap: Number of bootstrap resamples.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Tuple of (lower_bounds, upper_bounds) dicts keyed by feature name,
        or (None, None) if CI computation skipped or all iterations failed.

    """
    if confidence >= 1.0:  # sentinel: 1.0 disables CI computation
        return None, None

    alpha = (1.0 - confidence) / 2.0
    rng = np.random.default_rng(seed)

    feature_values: dict[str, list[float]] = {}
    n_data = len(training_data)

    for _ in range(n_bootstrap):
        try:
            idx = rng.integers(0, n_data, size=n_data)
            bg_sample = training_data[idx]

            # Disable discretization in bootstrap to keep feature names stable
            # across resamples (discretized boundaries differ each resample)
            boot_explainer = LimeTabularExplainer(
                training_data=bg_sample,
                feature_names=feature_names,
                mode=mode,
                kernel_width=kernel_width,
                discretize_continuous=False,
                random_state=rng.integers(0, 2**31),
            )

            explanation = boot_explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_samples=num_samples,
                num_features=num_features,
            )

            for feature_name, importance in explanation.as_list():
                if feature_name not in feature_values:
                    feature_values[feature_name] = []
                feature_values[feature_name].append(importance)

        except (ValueError, RuntimeError):
            logger.debug("Skipping degenerate bootstrap sample.")

    if not feature_values:
        logger.warning(
            "All %d bootstrap iterations failed; confidence intervals unavailable.",
            n_bootstrap,
        )
        return None, None

    lower_bounds = {
        name: float(np.percentile(vals, alpha * 100))
        for name, vals in feature_values.items()
    }
    upper_bounds = {
        name: float(np.percentile(vals, (1.0 - alpha) * 100))
        for name, vals in feature_values.items()
    }

    return lower_bounds, upper_bounds
