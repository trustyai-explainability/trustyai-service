"""Pure LIME computation against any synchronous prediction callable."""

from collections.abc import Callable
from importlib import import_module, util
from typing import cast

import numpy as np


def _optional_module_available(name: str) -> bool:
    """Report whether an optional distribution is installed without importing it."""
    try:
        return util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_LIME_AVAILABLE = _optional_module_available("lime")


def _load_lime_explainer() -> Callable[..., object]:
    """Load LIME only when an explanation is actually requested."""
    try:
        module = import_module("lime.lime_tabular")
        explainer = module.LimeTabularExplainer
    except (AttributeError, ImportError) as exc:
        msg = "LIME dependency is unavailable"
        raise ImportError(msg) from exc
    return cast("Callable[..., object]", explainer)


def _required_option(options: dict[str, object], name: str) -> object:
    try:
        return options.pop(name)
    except KeyError as exc:
        msg = f"Missing required keyword argument: {name}"
        raise TypeError(msg) from exc


def _reject_options(options: dict[str, object]) -> None:
    if options:
        names = ", ".join(sorted(options))
        msg = f"Unexpected keyword argument(s): {names}"
        raise TypeError(msg)


def _resolve_arguments(
    args: tuple[object, ...],
    options: dict[str, object],
    names: tuple[str, ...],
    required: int,
) -> dict[str, object]:
    if len(args) > len(names):
        msg = f"Expected at most {len(names)} positional arguments"
        raise TypeError(msg)
    values: dict[str, object] = {}
    for index, name in enumerate(names):
        if index < len(args):
            if name in options:
                msg = f"Multiple values for argument: {name}"
                raise TypeError(msg)
            values[name] = args[index]
        elif name in options:
            values[name] = options.pop(name)
        elif index < required:
            msg = f"Missing required argument: {name}"
            raise TypeError(msg)
    return values


def create_lime_explainer(
    training_data: np.ndarray,
    feature_names: list[str],
    mode: str,
    *,
    kernel_width: float = 0.75,
    seed: int | None = None,
) -> object:
    """Create a LIME tabular explainer for the supplied background data."""
    explainer = _load_lime_explainer()
    return explainer(
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
    **options: object,
) -> tuple[list[tuple[str, float]], float, float, float]:
    """Compute feature weights and prediction details for one instance."""
    num_samples = cast("int", _required_option(options, "num_samples"))
    num_features = cast("int", _required_option(options, "num_features"))
    label = cast("int | None", options.pop("label", None))
    _reject_options(options)
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
    *args: object,
    **options: object,
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Estimate LIME attribution bounds by bootstrap resampling."""
    names = (
        "training_data",
        "feature_names",
        "mode",
        "instance",
        "predict_fn",
        "confidence",
        "num_samples",
        "num_features",
        "kernel_width",
        "n_bootstrap",
        "seed",
        "label",
    )
    values = _resolve_arguments(args, options, names, required=6)
    training_data = cast("np.ndarray", values["training_data"])
    feature_names = cast("list[str]", values["feature_names"])
    mode = cast("str", values["mode"])
    instance = cast("np.ndarray", values["instance"])
    predict_fn = cast("Callable[[np.ndarray], np.ndarray]", values["predict_fn"])
    confidence = cast("float", values["confidence"])
    num_samples = cast("int", values.pop("num_samples", 5000))
    num_features = cast("int", values.pop("num_features", 10))
    kernel_width = cast("float", values.pop("kernel_width", 0.75))
    n_bootstrap = cast("int", values.pop("n_bootstrap", 50))
    seed = cast("int | None", values.pop("seed", None))
    label = cast("int | None", values.pop("label", None))
    _reject_options(options)
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
