"""Pure KernelSHAP computation against a generic prediction callable."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module, util
from typing import cast

import numpy as np


def _optional_module_available(name: str) -> bool:
    """Report whether an optional distribution is installed without importing it."""
    try:
        return util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_SHAP_AVAILABLE = _optional_module_available("shap")


def _load_shap_module() -> object:
    """Load SHAP only when an explanation is actually requested."""
    try:
        return import_module("shap")
    except ImportError as exc:
        msg = "SHAP dependency is unavailable"
        raise ImportError(msg) from exc


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


def _shap_options(options: dict[str, object]) -> tuple[int, str, str | float]:
    n_samples = cast("int", _required_option(options, "n_samples"))
    link = cast("str", _required_option(options, "link"))
    l1_reg = cast("str | float", _required_option(options, "l1_reg"))
    _reject_options(options)
    return n_samples, link, l1_reg


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
    **options: object,
) -> ShapExplanationResult:
    """Compute attributions and prediction values in the requested link space."""
    n_samples, link, l1_reg = _shap_options(options)
    shap_module = _load_shap_module()
    summary = shap_module.kmeans(
        background, min(n_samples, max(1, len(background) // 2))
    )
    explainer = shap_module.KernelExplainer(predict_fn, summary, link=link)
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
    **options: object,
) -> tuple[np.ndarray, float]:
    """Compute SHAP attributions and the expected model value."""
    result = compute_shap_result(instance, background, predict_fn, **options)
    return result.values, result.base_value


def compute_confidence_intervals(
    *args: object,
    **options: object,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate SHAP attribution bounds by bootstrap resampling."""
    names = (
        "instance",
        "background",
        "predict_fn",
        "confidence",
        "n_samples",
        "link",
        "l1_reg",
        "n_bootstrap",
        "seed",
    )
    arguments = _resolve_arguments(args, options, names, required=7)
    instance = cast("np.ndarray", arguments["instance"])
    background = cast("np.ndarray", arguments["background"])
    predict_fn = cast("Callable[[np.ndarray], np.ndarray]", arguments["predict_fn"])
    confidence = cast("float", arguments["confidence"])
    n_samples = cast("int", arguments["n_samples"])
    link = cast("str", arguments["link"])
    l1_reg = cast("str | float", arguments["l1_reg"])
    n_bootstrap = cast("int", arguments.pop("n_bootstrap", 50))
    seed = cast("int | None", arguments.pop("seed", None))
    _reject_options(options)
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
