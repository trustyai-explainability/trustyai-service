"""Pure KernelSHAP computation against a generic prediction callable."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    import shap
except ImportError:  # Optional explainability extra
    shap = None  # type: ignore[assignment]

_SHAP_AVAILABLE = shap is not None


@dataclass(frozen=True)
class ShapExplanationResult:
    """Attributions and link-space values from one KernelExplainer run."""

    values: np.ndarray
    base_value: float
    linked_prediction: float


def compute_shap_result(
    instance: np.ndarray,
    background: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_samples: int,
    link: str,
    l1_reg: str | float,
) -> ShapExplanationResult:
    """Compute attributions and prediction values in the requested link space."""
    if shap is None:
        msg = "SHAP dependency is unavailable"
        raise RuntimeError(msg)
    summary = shap.kmeans(background, min(n_samples, max(1, len(background) // 2)))
    explainer = shap.KernelExplainer(predict_fn, summary, link=link)
    raw_values = np.asarray(
        explainer.shap_values(
            instance.reshape(1, -1), nsamples=n_samples, l1_reg=l1_reg, silent=True
        )
    )
    values = raw_values
    while values.ndim > 1 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or values.size != instance.size:
        msg = "SHAP returned multi-output or malformed attributions"
        raise RuntimeError(msg)
    expected = np.asarray(explainer.expected_value).reshape(-1)
    if expected.size != 1 or not np.isfinite(expected[0]):
        msg = "SHAP returned a malformed base value"
        raise RuntimeError(msg)
    raw_prediction = np.asarray(predict_fn(instance.reshape(1, -1))).reshape(-1)
    if raw_prediction.size != 1 or not np.isfinite(raw_prediction[0]):
        msg = "Prediction callable returned a malformed scalar"
        raise RuntimeError(msg)
    linked_prediction = float(
        np.asarray(explainer.link.f(raw_prediction)).reshape(-1)[0]
    )
    return ShapExplanationResult(
        values=values.astype(float),
        base_value=float(expected[0]),
        linked_prediction=linked_prediction,
    )


def compute_shap_values(
    instance: np.ndarray,
    background: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    n_samples: int,
    link: str,
    l1_reg: str | float,
) -> tuple[np.ndarray, float]:
    """Compute SHAP attributions and the expected model value."""
    if shap is None:
        msg = "SHAP dependency is unavailable"
        raise RuntimeError(msg)
    result = compute_shap_result(
        instance,
        background,
        predict_fn,
        n_samples=n_samples,
        link=link,
        l1_reg=l1_reg,
    )
    return result.values, result.base_value


def compute_confidence_intervals(
    instance: np.ndarray,
    background: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    confidence: float,
    n_samples: int,
    link: str,
    l1_reg: str | float,
    n_bootstrap: int = 50,
    seed: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate SHAP attribution bounds by bootstrap resampling."""
    if confidence >= 1.0:
        return None, None
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        sample = background[rng.integers(0, len(background), len(background))]
        try:
            current, _ = compute_shap_values(
                instance,
                sample,
                predict_fn,
                n_samples=n_samples,
                link=link,
                l1_reg=l1_reg,
            )
            values.append(current)
        except Exception as exc:
            if getattr(exc, "code", None) is not None:
                raise
            if isinstance(exc, (ValueError, RuntimeError)):
                continue
            raise
    if not values:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    arr = np.asarray(values)
    return np.percentile(arr, alpha * 100, axis=0), np.percentile(
        arr, (1 - alpha) * 100, axis=0
    )
