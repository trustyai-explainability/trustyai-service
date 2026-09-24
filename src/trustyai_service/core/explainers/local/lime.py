"""Pure callable-based LIME computation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import import_module, util
from numbers import Integral, Real
from typing import NamedTuple, Protocol, cast

import numpy as np

logger = logging.getLogger(__name__)


def _optional_module_available(module_name: str) -> bool:
    """Report whether an optional distribution is discoverable without importing it."""
    try:
        return util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_LIME_AVAILABLE = _optional_module_available("lime")


class LIMEUnavailableError(ImportError):
    """The optional LIME algorithm dependency is not available."""

    def __init__(self) -> None:
        """Create an endpoint-neutral dependency error."""
        super().__init__("LIME dependency is unavailable")


def _load_lime_explainer() -> Callable[..., object]:
    """Load the optional LIME constructor only when requested."""
    if not _LIME_AVAILABLE:
        raise LIMEUnavailableError
    try:
        module = import_module("lime.lime_tabular")
        constructor = module.LimeTabularExplainer
    except (AttributeError, ImportError) as exc:
        raise LIMEUnavailableError from exc
    return cast("Callable[..., object]", constructor)


class LIMEExplanationResult(NamedTuple):
    """Numerical result of one LIME explanation."""

    feature_weights: list[tuple[str, float]]
    score: float
    local_prediction: float
    intercept: float


class LIMEConfidenceIntervalResult(NamedTuple):
    """Bootstrap confidence bounds for LIME feature weights."""

    lower_bounds: dict[str, float] | None
    upper_bounds: dict[str, float] | None


class _LIMEExplanationProtocol(Protocol):
    """Structural view of the result returned by LIME."""

    score: object
    local_pred: object
    intercept: object

    def available_labels(self) -> Sequence[int]:
        """Return labels for which LIME generated local explanations."""
        ...

    def as_list(self, *, label: int) -> Sequence[tuple[object, object]]:
        """Return feature names and weights for one label."""
        ...


class _LIMEExplainerProtocol(Protocol):
    """Structural view of the installed LIME explainer."""

    def explain_instance(
        self,
        data_row: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        **kwargs: object,
    ) -> object:
        """Generate one local explanation."""
        ...


def _positive_int(value: int, name: str) -> int:
    """Validate and normalize a positive integer algorithm option."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


def _label_value(label: int | None) -> int | None:
    """Validate an optional non-negative classification label."""
    if label is None:
        return None
    if isinstance(label, bool) or not isinstance(label, Integral) or label < 0:
        msg = "label must be a non-negative integer or None"
        raise ValueError(msg)
    return int(label)


def _feature_limit(explainer: object, num_features: int) -> int:
    """Cap the requested explanation size at the explainer's feature count."""
    feature_names = getattr(explainer, "feature_names", None)
    if feature_names is None:
        return num_features
    feature_count = len(feature_names)
    if feature_count < 1:
        msg = "LIME explainer has no features"
        raise ValueError(msg)
    return min(num_features, feature_count)


def _scalar(value: object, label: int) -> float:
    """Extract one finite-dimensional scalar from LIME's result containers."""
    if isinstance(value, Mapping):
        value = value[label]
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        msg = "LIME result field did not contain one scalar"
        raise ValueError(msg)
    return float(values[0])


def create_lime_explainer(
    training_data: np.ndarray,
    feature_names: Sequence[str],
    *,
    classification: bool,
    kernel_width: float = 0.75,
    seed: int | None = None,
) -> object:
    """Create a LIME tabular explainer for the supplied background data."""
    if not isinstance(classification, bool):
        msg = "classification must be a bool"
        raise TypeError(msg)
    constructor = _load_lime_explainer()
    mode = "classification" if classification else "regression"
    return constructor(
        training_data,
        feature_names=list(feature_names),
        mode=mode,
        kernel_width=kernel_width,
        discretize_continuous=False,
        random_state=seed,
    )


def compute_lime_explanation(  # noqa: PLR0913
    explainer: object,
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    num_samples: int,
    num_features: int,
    label: int | None,
) -> LIMEExplanationResult:
    """Compute one LIME explanation using the supplied prediction callable."""
    sample_count = _positive_int(num_samples, "num_samples")
    feature_count = _positive_int(num_features, "num_features")
    selected_label = _label_value(label)
    explainer_is_classification = getattr(explainer, "mode", None) == "classification"
    explain_kwargs: dict[str, object] = {
        "num_samples": sample_count,
        "num_features": _feature_limit(explainer, feature_count),
    }
    if selected_label is not None:
        explain_kwargs["labels"] = (selected_label,)

    # Do not catch exceptions from predict_fn here. In particular, provider errors
    # must remain the original exception object for the endpoint-neutral caller.
    explainer_api = cast("_LIMEExplainerProtocol", explainer)
    explanation = explainer_api.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        **explain_kwargs,
    )

    explanation_mode = getattr(explanation, "mode", None)
    is_classification = (
        explainer_is_classification or explanation_mode == "classification"
    )
    if is_classification:
        explanation_api = cast("_LIMEExplanationProtocol", explanation)
        available_labels = list(explanation_api.available_labels())
        if not available_labels:
            msg = "LIME produced no classification labels"
            raise ValueError(msg)
        result_label = (
            selected_label if selected_label is not None else int(available_labels[0])
        )
        if result_label not in available_labels:
            msg = f"LIME did not produce the requested label {result_label}"
            raise ValueError(msg)
    else:
        # LIME 0.2.0.1 uses label 1 as its regression sentinel.
        result_label = 1

    explanation_api = cast("_LIMEExplanationProtocol", explanation)
    feature_weights = [
        (str(feature_name), float(weight))
        for feature_name, weight in explanation_api.as_list(label=result_label)
    ]
    return LIMEExplanationResult(
        feature_weights=feature_weights,
        score=_scalar(explanation_api.score, result_label),
        local_prediction=_scalar(explanation_api.local_pred, result_label),
        intercept=_scalar(explanation_api.intercept, result_label),
    )


class _PredictionFunctionError(Exception):
    """Internal marker that separates prediction failures from LIME failures."""

    def __init__(self, original: Exception) -> None:
        """Keep the exact exception object raised by the prediction callable."""
        self.original = original
        super().__init__(str(original))


def _guard_prediction_fn(
    predict_fn: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Mark prediction-callable errors so bootstrap handling cannot skip them."""

    def guarded(values: np.ndarray) -> np.ndarray:
        try:
            return predict_fn(values)
        except Exception as exc:
            raise _PredictionFunctionError(exc) from exc

    return guarded


@contextmanager
def _seeded_explainer(explainer: object, seed: int | None) -> Iterator[None]:
    """Temporarily seed LIME's own random states for reproducible bootstrap runs."""
    if seed is None:
        yield
        return

    targets: list[tuple[object, str, object]] = []
    base = getattr(explainer, "base", None)
    for target in (explainer, base):
        if target is not None and hasattr(target, "random_state"):
            targets.append((target, "random_state", target.random_state))
            target.random_state = np.random.RandomState(seed)
    try:
        yield
    finally:
        for target, name, original in targets:
            setattr(target, name, original)


def _validate_confidence(confidence: float) -> float:
    """Validate a confidence level in the open/closed unit interval."""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        msg = "confidence must be a finite number in (0, 1]"
        raise TypeError(msg)
    confidence_value = float(confidence)
    if not np.isfinite(confidence_value) or not 0.0 < confidence_value <= 1.0:
        msg = "confidence must be a finite number in (0, 1]"
        raise ValueError(msg)
    return confidence_value


def _restore_prediction_error(error: _PredictionFunctionError) -> None:
    """Re-raise the exact original prediction exception object."""
    raise error.original


def compute_lime_confidence_intervals(  # noqa: PLR0913
    explainer: object,
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    num_samples: int,
    num_features: int,
    confidence: float,
    n_bootstrap: int = 50,
    seed: int | None,
    label: int | None,
) -> LIMEConfidenceIntervalResult:
    """Estimate LIME feature-weight bounds by repeated seeded explanations."""
    confidence_value = _validate_confidence(confidence)
    if confidence_value == 1.0:
        return LIMEConfidenceIntervalResult(None, None)

    sample_count = _positive_int(num_samples, "num_samples")
    feature_count = _positive_int(num_features, "num_features")
    bootstrap_count = _positive_int(n_bootstrap, "n_bootstrap")
    selected_label = _label_value(label)
    rng = np.random.default_rng(seed)
    known_names = list(getattr(explainer, "feature_names", ()))
    values: dict[str, list[float]] = {name: [] for name in known_names}
    successful = 0

    for _ in range(bootstrap_count):
        iteration_seed = int(rng.integers(0, 2**32 - 1))
        try:
            with _seeded_explainer(
                explainer, iteration_seed if seed is not None else None
            ):
                result = compute_lime_explanation(
                    explainer,
                    instance,
                    _guard_prediction_fn(predict_fn),
                    num_samples=sample_count,
                    num_features=feature_count,
                    label=selected_label,
                )
        except _PredictionFunctionError as exc:
            _restore_prediction_error(exc)
        except ValueError:
            # LIME's local regression can be numerically degenerate for an
            # individual stochastic neighborhood. Such a sample is skipped;
            # provider-callable failures are marked above and are never skipped.
            logger.debug("Skipping numerically degenerate LIME bootstrap sample")
            continue

        current = dict(result.feature_weights)
        new_names = [name for name in current if name not in values]
        for name in new_names:
            values[name] = [0.0] * successful
        for name, feature_values in values.items():
            feature_values.append(float(current.get(name, 0.0)))
        successful += 1

    if not successful:
        return LIMEConfidenceIntervalResult(None, None)

    alpha = (1.0 - confidence_value) / 2.0
    lower_bounds = {
        name: float(np.percentile(feature_values, alpha * 100.0))
        for name, feature_values in values.items()
    }
    upper_bounds = {
        name: float(np.percentile(feature_values, (1.0 - alpha) * 100.0))
        for name, feature_values in values.items()
    }
    return LIMEConfidenceIntervalResult(lower_bounds, upper_bounds)
