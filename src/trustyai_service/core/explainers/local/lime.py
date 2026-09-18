"""Pure LIME computation against any synchronous prediction callable."""

from collections.abc import Callable

import numpy as np

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # Optional explainability extra
    LimeTabularExplainer = None  # type: ignore[assignment,misc]

_LIME_AVAILABLE = LimeTabularExplainer is not None


def create_lime_explainer(
    training_data: np.ndarray,
    feature_names: list[str],
    mode: str,
    *,
    kernel_width: float = 0.75,
    seed: int | None = None,
):
    if LimeTabularExplainer is None:
        raise RuntimeError("LIME dependency is unavailable")
    return LimeTabularExplainer(
        training_data,
        feature_names=feature_names,
        mode=mode,
        kernel_width=kernel_width,
        discretize_continuous=False,
        random_state=seed,
    )


def compute_lime_explanation(
    explainer: object,
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    num_samples: int,
    num_features: int,
    label: int | None = None,
) -> tuple[list[tuple[str, float]], float, float, float]:
    explanation = explainer.explain_instance(
        instance, predict_fn, num_samples=num_samples, num_features=num_features
    )
    selected_label = (
        label
        if label is not None
        else (
            explanation.available_labels()[0]
            if explanation.mode == "classification"
            else 1
        )
    )

    def scalar(value: object) -> float:
        if isinstance(value, dict):
            value = value[selected_label]
        return float(np.asarray(value).reshape(-1)[0])

    return (
        explanation.as_list(label=selected_label),
        scalar(explanation.score),
        scalar(explanation.local_pred),
        scalar(explanation.intercept),
    )


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
    label: int | None = None,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Estimate LIME attribution bounds by bootstrap resampling."""
    if confidence >= 1.0:
        return None, None
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        try:
            sample = training_data[
                rng.integers(0, len(training_data), len(training_data))
            ]
            explainer = create_lime_explainer(
                sample,
                feature_names,
                mode,
                kernel_width=kernel_width,
                seed=int(rng.integers(0, 2**31)),
            )
            items, *_ = compute_lime_explanation(
                explainer,
                instance,
                predict_fn,
                num_samples=num_samples,
                num_features=num_features,
                label=label,
            )
            current = dict(items)
            for name in feature_names:
                values.setdefault(name, []).append(float(current.get(name, 0.0)))
        except Exception as exc:
            if getattr(exc, "code", None) is not None:
                raise
            if isinstance(exc, (ValueError, RuntimeError)):
                continue
            raise
    if not values:
        return None, None
    alpha = (1.0 - confidence) / 2.0
    return (
        {
            name: float(np.percentile(items, alpha * 100))
            for name, items in values.items()
        },
        {
            name: float(np.percentile(items, (1 - alpha) * 100))
            for name, items in values.items()
        },
    )
