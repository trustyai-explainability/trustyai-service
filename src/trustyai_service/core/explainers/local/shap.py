"""Pure callable-based KernelSHAP computation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module, util
from numbers import Integral, Real
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_BOOTSTRAP_SAMPLES = 50
_BOOTSTRAP_CENTROID_CAP_FACTOR = 0.5
_MATRIX_RANK = 2
_VECTOR_RANK = 1
_SINGLETON = 1
_EMPTY = 0
_SUPPORTED_LINKS = frozenset({"identity", "logit"})
_SUPPORTED_L1_REGULARIZERS = frozenset({"auto", "aic", "bic"})
_NUM_FEATURES_PREFIX = "num_features("

if TYPE_CHECKING:
    from collections.abc import Callable


def _optional_module_available(module_name: str) -> bool:
    """Report whether an optional distribution is discoverable."""
    try:
        return util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_SHAP_AVAILABLE = _optional_module_available("shap")


class SHAPUnavailableError(ImportError):
    """The optional SHAP dependency is not available."""

    def __init__(self) -> None:
        """Create an endpoint-neutral dependency error."""
        super().__init__("SHAP dependency is unavailable")


def _load_shap_module() -> object:
    """Load SHAP only when an explanation is requested."""
    if not _SHAP_AVAILABLE:
        raise SHAPUnavailableError
    try:
        module = cast("_ShapModuleProtocol", import_module("shap"))
        kmeans = module.kmeans
        kernel_explainer = module.KernelExplainer
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SHAPUnavailableError from exc
    if not callable(kmeans) or not callable(kernel_explainer):
        raise SHAPUnavailableError
    return module


class _KernelExplainerProtocol(Protocol):
    """Structural view of the installed KernelExplainer."""

    expected_value: object
    link: object

    def shap_values(self, values: np.ndarray, **kwargs: object) -> object:
        """Return SHAP values for the supplied instances."""
        ...


class _LinkProtocol(Protocol):
    """Structural view of a SHAP link object."""

    def f(self, values: object) -> object:
        """Apply the link function."""
        ...


class _ShapModuleProtocol(Protocol):
    """Structural view of the lazily imported SHAP module."""

    kmeans: Callable[[np.ndarray, int], object]
    KernelExplainer: Callable[..., object]


class _PredictionFunctionError(Exception):
    """Marker that preserves a prediction callable's original exception."""

    def __init__(self, original: Exception) -> None:
        """Keep the exact exception object raised by the callable."""
        self.original = original
        super().__init__(str(original))


@dataclass(frozen=True)
class ShapExplanationResult:
    """Attributions and link-space prediction values from KernelSHAP."""

    values: np.ndarray
    base_value: float
    linked_prediction: float


def _positive_int(value: int, name: str) -> int:
    """Validate a positive integer algorithm option."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return int(value)


def _validate_link(link: str) -> str:
    """Validate the link names supported by the local SHAP contract."""
    if not isinstance(link, str) or link not in _SUPPORTED_LINKS:
        msg = "link must be 'identity' or 'logit'"
        raise ValueError(msg)
    return link


def _validate_l1_reg(value: str | float) -> str | float:
    """Validate the regularizer forms supported by KernelExplainer."""
    if isinstance(value, bool):
        msg = "l1_reg must be a supported string or finite non-negative number"
        raise TypeError(msg)
    if isinstance(value, Real):
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or numeric_value < 0.0:
            msg = "l1_reg must be a finite non-negative number"
            raise ValueError(msg)
        return numeric_value
    if not isinstance(value, str):
        msg = "l1_reg must be a supported string or finite non-negative number"
        raise TypeError(msg)
    if value in _SUPPORTED_L1_REGULARIZERS:
        return value
    if value.startswith(_NUM_FEATURES_PREFIX) and value.endswith(")"):
        feature_count = value[len(_NUM_FEATURES_PREFIX) : -1]
        if feature_count.isdigit() and int(feature_count) > 0:
            return value
    msg = "l1_reg must be 'auto', 'aic', 'bic', 'num_features(n)', or a finite non-negative number"
    raise ValueError(msg)


def _validate_confidence(confidence: float) -> float:
    """Validate a confidence level in the open/closed unit interval."""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        msg = "confidence must be a finite number in (0, 1]"
        raise TypeError(msg)
    value = float(confidence)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        msg = "confidence must be a finite number in (0, 1]"
        raise ValueError(msg)
    return value


def _validate_inputs(
    background: np.ndarray,
    instance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the tabular arrays required by KernelSHAP."""
    background_array = np.asarray(background)
    instance_array = np.asarray(instance)
    if background_array.ndim != _MATRIX_RANK or background_array.shape[0] == _EMPTY:
        msg = "background must be a non-empty two-dimensional array"
        raise ValueError(msg)
    if instance_array.ndim != _VECTOR_RANK:
        msg = "instance must be a one-dimensional array"
        raise ValueError(msg)
    if background_array.shape[1] != instance_array.shape[0]:
        msg = "background and instance feature widths must match"
        raise ValueError(msg)
    if background_array.dtype.kind == "c" or instance_array.dtype.kind == "c":
        msg = "background and instance must be numeric arrays"
        raise TypeError(msg)
    if background_array.shape[1] == _EMPTY:
        msg = "background and instance must contain at least one feature"
        raise ValueError(msg)
    try:
        background_array = background_array.astype(float, copy=False)
        instance_array = instance_array.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = "background and instance must be numeric arrays"
        raise TypeError(msg) from exc
    if not np.isfinite(background_array).all() or not np.isfinite(instance_array).all():
        msg = "background and instance must contain finite values"
        raise ValueError(msg)
    return background_array, instance_array


def _scalar_prediction(value: object) -> float:
    """Normalize the supplied scalar prediction without invoking a callable."""
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = "instance_prediction must contain one finite numeric value"
        raise ValueError(msg) from exc
    if values.size != 1 or not np.isfinite(values[0]):
        msg = "instance_prediction must contain one finite numeric value"
        raise ValueError(msg)
    return float(values[0])


def _compute_n_centroids(background_length: int, n_samples: int) -> int:
    """Bound the k-means summary size for stable small backgrounds."""
    return min(
        n_samples,
        max(1, int(background_length * _BOOTSTRAP_CENTROID_CAP_FACTOR)),
    )


def _normalize_values(raw: object, feature_count: int) -> np.ndarray:
    """Normalize scalar-output SHAP values to one value per feature."""
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = "SHAP returned malformed attributions"
        raise RuntimeError(msg) from exc
    if values.ndim == _MATRIX_RANK and values.shape[0] == _SINGLETON:
        values = values[0]
    if values.ndim != 1 or values.size != feature_count:
        msg = "SHAP returned multi-output or malformed attributions"
        raise RuntimeError(msg)
    if not np.isfinite(values).all():
        msg = "SHAP returned non-finite attributions"
        raise RuntimeError(msg)
    return values


def _normalize_base_value(value: object) -> float:
    """Normalize SHAP's scalar expected value."""
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = "SHAP returned a malformed base value"
        raise RuntimeError(msg) from exc
    if values.size != 1 or not np.isfinite(values[0]):
        msg = "SHAP returned a malformed base value"
        raise RuntimeError(msg)
    return float(values[0])


def _linked_prediction(explainer: _KernelExplainerProtocol, prediction: float) -> float:
    """Apply the KernelExplainer-owned link to the supplied prediction once."""
    try:
        link = cast("_LinkProtocol", explainer.link)
    except (AttributeError, RuntimeError) as exc:
        raise SHAPUnavailableError from exc
    try:
        values = np.asarray(link.f(np.asarray([prediction], dtype=float)), dtype=float)
    except AttributeError as exc:
        raise SHAPUnavailableError from exc
    except (TypeError, ValueError, OverflowError) as exc:
        msg = "SHAP link returned a malformed prediction"
        raise RuntimeError(msg) from exc
    values = values.reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        msg = "SHAP link returned a malformed prediction"
        raise RuntimeError(msg)
    return float(values[0])


def _compute_shap_result(  # noqa: PLR0913
    background: np.ndarray,
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    instance_prediction: float,
    n_samples: int,
    link: str,
    l1_reg: str | float,
) -> ShapExplanationResult:
    """Share point-computation logic between the public APIs."""
    background_array, instance_array = _validate_inputs(background, instance)
    sample_count = _positive_int(n_samples, "n_samples")
    selected_link = _validate_link(link)
    regularizer = _validate_l1_reg(l1_reg)
    prediction = _scalar_prediction(instance_prediction)

    shap_module = cast("_ShapModuleProtocol", _load_shap_module())
    try:
        kmeans = shap_module.kmeans
        kernel_explainer = shap_module.KernelExplainer
    except (AttributeError, RuntimeError) as exc:
        raise SHAPUnavailableError from exc
    if not callable(kmeans) or not callable(kernel_explainer):
        raise SHAPUnavailableError
    n_centroids = _compute_n_centroids(len(background_array), sample_count)
    try:
        summary = kmeans(background_array, n_centroids)
        explainer = cast(
            "_KernelExplainerProtocol",
            kernel_explainer(predict_fn, summary, link=selected_link),
        )
    except _PredictionFunctionError:
        raise
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise SHAPUnavailableError from exc
    try:
        shap_values = explainer.shap_values
    except (AttributeError, RuntimeError) as exc:
        raise SHAPUnavailableError from exc
    if not callable(shap_values):
        raise SHAPUnavailableError
    try:
        raw_values = shap_values(
            instance_array.reshape(1, -1),
            nsamples=sample_count,
            l1_reg=regularizer,
            silent=True,
        )
    except _PredictionFunctionError:
        raise
    except (AttributeError, TypeError) as exc:
        raise SHAPUnavailableError from exc
    values = _normalize_values(raw_values, instance_array.size)
    try:
        expected_value = explainer.expected_value
    except (AttributeError, RuntimeError) as exc:
        raise SHAPUnavailableError from exc
    base_value = _normalize_base_value(expected_value)
    linked_prediction = _linked_prediction(explainer, prediction)
    return ShapExplanationResult(
        values=values.astype(float, copy=False),
        base_value=base_value,
        linked_prediction=linked_prediction,
    )


def compute_shap_result(  # noqa: PLR0913
    background: np.ndarray,
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    instance_prediction: float,
    n_samples: int,
    link: str,
    l1_reg: str | float,
) -> ShapExplanationResult:
    """Compute one KernelSHAP result using the supplied prediction callable."""
    try:
        return _compute_shap_result(
            background,
            instance,
            _guard_prediction_fn(predict_fn),
            instance_prediction=instance_prediction,
            n_samples=n_samples,
            link=link,
            l1_reg=l1_reg,
        )
    except _PredictionFunctionError as exc:
        raise exc.original from None


def _guard_prediction_fn(
    predict_fn: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    """Mark callable failures so bootstrap handling cannot skip them."""

    def guarded(values: np.ndarray) -> np.ndarray:
        try:
            return predict_fn(values)
        except Exception as exc:
            raise _PredictionFunctionError(exc) from exc

    return guarded


def compute_confidence_intervals(  # noqa: PLR0913
    background: np.ndarray,
    instance: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    instance_prediction: float,
    n_samples: int,
    link: str,
    l1_reg: str | float,
    confidence: float,
    n_bootstrap: int = _DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Estimate KernelSHAP attribution bounds by seeded background resampling."""
    confidence_value = _validate_confidence(confidence)
    regularizer = _validate_l1_reg(l1_reg)
    if confidence_value == 1.0:
        return None, None

    background_array, instance_array = _validate_inputs(background, instance)
    sample_count = _positive_int(n_samples, "n_samples")
    bootstrap_count = _positive_int(n_bootstrap, "n_bootstrap")
    selected_link = _validate_link(link)
    prediction = _scalar_prediction(instance_prediction)
    rng = np.random.default_rng(seed)
    bootstrap_values: list[np.ndarray] = []

    for _ in range(bootstrap_count):
        indexes = rng.integers(0, len(background_array), size=len(background_array))
        sample = background_array[indexes]
        try:
            result = _compute_shap_result(
                sample,
                instance_array,
                _guard_prediction_fn(predict_fn),
                instance_prediction=prediction,
                n_samples=sample_count,
                link=selected_link,
                l1_reg=regularizer,
            )
        except _PredictionFunctionError as exc:
            raise exc.original from None
        except np.linalg.LinAlgError:
            logger.debug("Skipping numerically degenerate SHAP bootstrap sample")
            continue
        bootstrap_values.append(result.values)

    if not bootstrap_values:
        return None, None

    alpha = (1.0 - confidence_value) / 2.0
    values = np.asarray(bootstrap_values)
    lower = np.percentile(values, alpha * 100.0, axis=0)
    upper = np.percentile(values, (1.0 - alpha) * 100.0, axis=0)
    return lower.astype(float, copy=False), upper.astype(float, copy=False)
