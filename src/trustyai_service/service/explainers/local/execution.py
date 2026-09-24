"""Create one prediction path for a local explanation."""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from .error_mapping import LocalDataError, LocalExecutionError
from .model_provider import (
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderInvalidRequestError,
)
from .prediction_adapter import prediction_callable
from .transport_config import get_transport_config
from .types import PredictionSource, TaskType

_MODEL_PROVIDER_MODULE = "trustyai_service.service.explainers.local.kserve_v2_http"
_SURROGATE_MODULE = "trustyai_service.core.explainers.local.surrogate"
_MATRIX_RANK = 2
_MIN_CLASS_COUNT = 2

if TYPE_CHECKING:
    from collections.abc import Callable

    from trustyai_service.service.data.local_explanation import LocalExplanationData

    from .model_provider import (
        HttpTransportConfig,
        PredictionMetadata,
        PredictionProvider,
    )


class _SurrogateEstimator(Protocol):
    """Prediction methods exposed by the two supported surrogate estimators."""

    def predict(self, values: np.ndarray) -> object:
        """Predict regression values."""
        ...

    def predict_proba(self, values: np.ndarray) -> object:
        """Predict classification probabilities."""
        ...


def _remaining_deadline(deadline: float) -> float:
    """Return the positive time remaining for an absolute monotonic deadline."""
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ProviderDeadlineError
    return remaining


def _load_model_provider() -> tuple[type, type]:
    """Load the outbound provider module only for MODEL execution."""
    module = importlib.import_module(_MODEL_PROVIDER_MODULE)
    provider_type = cast("type", module.KServeV2HttpPredictionProvider)  # type: ignore[attr-defined]
    model_spec_type = cast("type", module.KServeModelSpec)  # type: ignore[attr-defined]
    return provider_type, model_spec_type


def _load_surrogate_builder() -> Callable[[np.ndarray, np.ndarray, str], object]:
    """Load the random-forest builder only for explicit SURROGATE execution."""
    module = importlib.import_module(_SURROGATE_MODULE)
    return cast(
        "Callable[[np.ndarray, np.ndarray, str], object]",
        module.build_surrogate,  # type: ignore[attr-defined]
    )


def _close_provider(provider: object) -> None:
    """Best-effort cleanup for a provider created before execution construction failed."""
    try:
        close = cast("PredictionProvider", provider).close
        close()
    except Exception:  # noqa: BLE001
        return


@dataclass(frozen=True)
class LocalExecutionSpec:
    """Immutable model and prediction-source selection for one explanation."""

    prediction_source: PredictionSource
    base_url: str | None
    model_name: str
    model_version: str | None
    input_name: str | None
    output_name: str | None
    task: TaskType


@dataclass
class PredictionExecution:
    """Prediction callable and provider owned by one worker execution."""

    predict_fn: Callable[[np.ndarray], np.ndarray]
    source: PredictionSource
    provider: PredictionProvider | None
    metadata: PredictionMetadata | None
    resolved_input_name: str | None
    resolved_output_name: str | None
    task: TaskType
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Close the owned provider at most once, including partial executions."""
        if self._closed:
            return
        self._closed = True
        if self.provider is not None:
            _close_provider(self.provider)


def create_prediction_execution(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    deadline: float,
    transport: HttpTransportConfig | None,
) -> PredictionExecution:
    """Create a MODEL or explicitly selected SURROGATE prediction execution."""
    if spec.prediction_source == PredictionSource.SURROGATE:
        return _create_surrogate_execution(spec, data)
    if spec.prediction_source != PredictionSource.MODEL:
        msg = "Invalid local prediction source"
        raise ProviderInvalidRequestError(msg)
    return _create_model_execution(spec, data, deadline, transport)


def _select_surrogate_output(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    targets: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    """Select one stored output column for a surrogate estimator."""
    if targets.ndim not in {1, _MATRIX_RANK}:
        msg = "Stored surrogate labels must be one- or two-dimensional"
        raise LocalDataError(msg)
    if data.background.ndim != _MATRIX_RANK:
        msg = "Stored background data must be a two-dimensional matrix"
        raise LocalDataError(msg)
    if targets.shape[0] != data.background.shape[0]:
        msg = "Stored surrogate labels are misaligned with the background"
        raise LocalDataError(msg)

    if targets.ndim == 1:
        return _select_surrogate_vector(spec, data, targets)
    return _select_surrogate_matrix(spec, data, targets)


def _select_surrogate_vector(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    targets: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    """Select the only available column from a vector of stored labels."""
    output_names = data.output_names
    if len(output_names) > 1 or (
        spec.output_name is not None
        and (not output_names or spec.output_name != output_names[0])
    ):
        msg = "Surrogate output_name is required for the selected stored output"
        raise LocalDataError(msg)
    selected_name = output_names[0] if output_names else None
    return targets, selected_name


def _select_surrogate_matrix(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    targets: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    """Select one named column from a matrix of stored labels."""
    output_names = data.output_names
    if output_names and len(output_names) != targets.shape[1]:
        msg = "Stored output columns do not match their aliases"
        raise LocalDataError(msg)
    if targets.shape[1] == 0:
        msg = "Stored surrogate labels must contain one output column"
        raise LocalDataError(msg)
    if spec.output_name is None:
        if targets.shape[1] != 1:
            msg = "Surrogate output_name is required for multiple outputs"
            raise LocalDataError(msg)
        selected_index = 0
    else:
        if not output_names or spec.output_name not in output_names:
            msg = "Requested stored output was not found"
            raise LocalDataError(msg)
        selected_index = output_names.index(spec.output_name)

    if spec.task == TaskType.CLASSIFICATION and _is_probability_matrix(targets):
        msg = (
            "Stored surrogate classification outputs must be discrete labels, "
            "not probability vectors or class scores"
        )
        raise LocalDataError(msg)

    selected_name = output_names[selected_index] if output_names else None
    return targets[:, selected_index], selected_name


def _is_probability_matrix(targets: np.ndarray) -> bool:
    """Identify multi-column stored class probabilities before label selection."""
    if targets.shape[1] < _MIN_CLASS_COUNT or targets.dtype.kind not in "biuf":
        return False
    try:
        numeric_targets = targets.astype(float)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        np.isfinite(numeric_targets).all()
        and np.all((numeric_targets >= 0) & (numeric_targets <= 1))
        and np.allclose(
            numeric_targets.sum(axis=1),
            1.0,
            rtol=1e-5,
            atol=1e-3,
        )
    )


def _validate_surrogate_labels(
    labels: np.ndarray,
    data: LocalExplanationData,
    task: TaskType,
) -> np.ndarray:
    """Validate selected stored labels before fitting the explicit surrogate."""
    if labels.size == 0:
        msg = "Surrogate labels must be finite and non-empty"
        raise LocalDataError(msg)
    if labels.dtype.kind not in "biuf":
        msg = "Surrogate labels must be numeric"
        raise LocalDataError(msg)
    try:
        numeric_labels = labels.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = "Surrogate labels must be numeric"
        raise LocalDataError(msg) from exc
    if not np.isfinite(numeric_labels).all():
        msg = "Surrogate labels must be finite and non-empty"
        raise LocalDataError(msg)
    if numeric_labels.shape[0] != data.background.shape[0]:
        msg = "Stored surrogate labels are misaligned with the background"
        raise LocalDataError(msg)
    if task == TaskType.CLASSIFICATION:
        if len(np.unique(numeric_labels)) < _MATRIX_RANK:
            msg = "Surrogate classification labels must contain two finite classes"
            raise LocalDataError(msg)
        if not np.equal(numeric_labels, np.floor(numeric_labels)).all():
            msg = "Surrogate classification labels must be discrete"
            raise LocalDataError(msg)
    return numeric_labels


def _create_surrogate_execution(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
) -> PredictionExecution:
    """Create a provider-free execution for an explicit stored-data surrogate."""
    if data.background_output is None:
        msg = "Surrogate mode requires stored organic labels"
        raise LocalDataError(msg)
    targets = np.asarray(data.background_output)
    selected_targets, selected_output = _select_surrogate_output(spec, data, targets)
    labels = _validate_surrogate_labels(selected_targets, data, spec.task)
    try:
        estimator = _load_surrogate_builder()(
            data.background,
            labels,
            spec.task.value,
        )
        fitted = cast("_SurrogateEstimator", estimator)
        raw_predict = (
            fitted.predict_proba
            if spec.task == TaskType.CLASSIFICATION
            else fitted.predict
        )
    except Exception as exc:
        msg = "Local surrogate training failed"
        raise LocalExecutionError(msg) from exc

    return PredictionExecution(
        predict_fn=prediction_callable(raw_predict, spec.task),
        source=PredictionSource.SURROGATE,
        provider=None,
        metadata=None,
        resolved_input_name=None,
        resolved_output_name=selected_output,
        task=spec.task,
    )


def _input_width(metadata: PredictionMetadata) -> int:
    """Return the flat feature width represented by negotiated metadata."""
    if not metadata.input_shape:
        msg = "Model input metadata does not define a feature width"
        raise LocalDataError(msg)
    width = 1 if metadata.input_shape == (-1,) else metadata.input_shape[-1]
    if width <= 0:
        msg = "Model input metadata does not define a feature width"
        raise LocalDataError(msg)
    return width


def _validate_model_width(data: LocalExplanationData, width: int) -> None:
    """Verify stored instance and background widths before any inference call."""
    if data.instance.ndim != 1 or data.background.ndim != _MATRIX_RANK:
        msg = (
            "Stored data must contain a one-dimensional instance and matrix background"
        )
        raise LocalDataError(msg)
    if data.instance.shape[0] != width or data.background.shape[1] != width:
        msg = "Stored data feature width does not match model input"
        raise LocalDataError(msg)


def _create_model_execution(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    deadline: float,
    transport: HttpTransportConfig | None,
) -> PredictionExecution:
    """Create a lazily imported, deadline-bounded KServe model execution."""
    _remaining_deadline(deadline)
    if spec.base_url is None:
        msg = "MODEL execution requires a base URL"
        raise ProviderInvalidRequestError(msg)

    resolved_transport = (
        transport
        if transport is not None
        else get_transport_config(PredictionSource.MODEL)
    )
    if resolved_transport is None:
        msg = "MODEL execution requires model transport settings"
        raise ProviderConfigurationError(msg)

    provider_type, model_spec_type = _load_model_provider()
    model_spec = model_spec_type(
        base_url=spec.base_url,
        model_name=spec.model_name,
        model_version=spec.model_version,
        input_name=spec.input_name,
        output_name=spec.output_name,
    )
    provider: PredictionProvider | None = None
    try:
        connect_timeout = _remaining_deadline(deadline)
        provider = cast(
            "PredictionProvider",
            provider_type.connect(
                model_spec,
                resolved_transport,
                timeout_seconds=connect_timeout,
            ),
        )
        _remaining_deadline(deadline)
        metadata = provider.metadata
        _validate_model_width(data, _input_width(metadata))

        def predict(values: np.ndarray) -> np.ndarray:
            """Call the provider with the remaining request deadline."""
            remaining = _remaining_deadline(deadline)
            timeout_seconds = min(connect_timeout, remaining)
            return provider.predict(values, timeout_seconds=timeout_seconds)

        return PredictionExecution(
            predict_fn=prediction_callable(predict, spec.task),
            source=PredictionSource.MODEL,
            provider=provider,
            metadata=metadata,
            resolved_input_name=metadata.input_name,
            resolved_output_name=metadata.output_name,
            task=spec.task,
        )
    except Exception:
        if provider is not None:
            _close_provider(provider)
        raise
