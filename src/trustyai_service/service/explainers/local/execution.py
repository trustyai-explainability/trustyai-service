"""Select a real model or an explicitly requested local surrogate."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from trustyai_service.service.data.local_explanation import LocalExplanationData

from .model_provider import (
    HttpTransportConfig,
    KServeModelSpec,
    LocalDataError,
    LocalExecutionError,
    PredictionMetadata,
    PredictionProvider,
)
from .prediction_adapter import prediction_callable
from .transport_config import get_transport_config
from .types import PredictionSource, TaskType

_MATRIX_RANK = 2


@dataclass(frozen=True)
class LocalExecutionSpec:
    """Immutable inputs used to construct one explanation prediction path."""

    prediction_source: PredictionSource
    base_url: str | None
    model_name: str
    model_version: str | None
    input_name: str | None
    output_name: str | None
    task: TaskType


@dataclass
class PredictionExecution:
    """Prediction callable and resources owned by one worker invocation."""

    predict_fn: Callable[[np.ndarray], np.ndarray]
    provider: PredictionProvider | None = None
    metadata: PredictionMetadata | None = None
    resolved_input_name: str | None = None
    resolved_output_name: str | None = None
    task: TaskType = TaskType.REGRESSION
    source: PredictionSource = PredictionSource.MODEL
    _closed: bool = False

    def close(self) -> None:
        """Close the provider exactly once, if this execution owns one."""
        if not self._closed and self.provider is not None:
            self.provider.close()
        self._closed = True


def create_prediction_execution(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    deadline: float,
    *,
    transport: HttpTransportConfig | None = None,
    allow_single_probability: bool = False,
) -> PredictionExecution:
    """Build the real-model or explicit-surrogate prediction execution."""
    if spec.prediction_source is PredictionSource.SURROGATE:
        return _create_surrogate_execution(
            spec, data, allow_single_probability=allow_single_probability
        )
    return _create_model_execution(
        spec,
        data,
        deadline,
        transport=transport,
        allow_single_probability=allow_single_probability,
    )


def _create_surrogate_execution(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    *,
    allow_single_probability: bool,
) -> PredictionExecution:
    from trustyai_service.core.explainers.local.surrogate import build_surrogate

    if data.background_output is None:
        msg = "Surrogate mode requires stored organic labels"
        raise LocalDataError(msg)
    selected_targets, selected_output = _select_surrogate_target(
        spec, data, np.asarray(data.background_output)
    )
    if selected_targets.size == 0 or not np.isfinite(selected_targets).all():
        msg = "Surrogate labels must be finite and non-empty"
        raise LocalDataError(msg)
    if spec.task is TaskType.CLASSIFICATION:
        if len(np.unique(selected_targets)) < _MATRIX_RANK:
            msg = "Surrogate classification labels must contain two finite classes"
            raise LocalDataError(msg)
        if not np.all(selected_targets == np.floor(selected_targets)):
            msg = "Surrogate classification labels must be discrete"
            raise LocalDataError(msg)
    try:
        estimator = build_surrogate(data.background, selected_targets, spec.task.value)
    except Exception as exc:
        msg = "Local surrogate training failed"
        raise LocalExecutionError(msg) from exc
    raw = (
        estimator.predict_proba
        if spec.task is TaskType.CLASSIFICATION
        else estimator.predict
    )
    return PredictionExecution(
        prediction_callable(
            raw, spec.task, allow_single_probability=allow_single_probability
        ),
        provider=None,
        resolved_output_name=selected_output,
        task=spec.task,
        source=spec.prediction_source,
    )


def _select_surrogate_target(
    spec: LocalExecutionSpec, data: LocalExplanationData, targets: np.ndarray
) -> tuple[np.ndarray, str | None]:
    if targets.ndim == 1:
        if len(data.output_names) > 1 or (
            spec.output_name is not None
            and (not data.output_names or spec.output_name != data.output_names[0])
        ):
            msg = "Surrogate output_name is required for the selected stored output"
            raise LocalDataError(msg)
        return targets, data.output_names[0] if data.output_names else None
    if targets.ndim != _MATRIX_RANK:
        msg = "Stored surrogate labels must be one- or two-dimensional"
        raise LocalDataError(msg)
    if spec.output_name is None:
        if targets.shape[1] != 1:
            msg = "Surrogate output_name is required for multiple outputs"
            raise LocalDataError(msg)
        selected_index = 0
    else:
        if spec.output_name not in data.output_names:
            msg = "Requested stored output was not found"
            raise LocalDataError(msg)
        selected_index = data.output_names.index(spec.output_name)
    return targets[:, selected_index], data.output_names[selected_index]


def _create_model_execution(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    deadline: float,
    *,
    transport: HttpTransportConfig | None,
    allow_single_probability: bool,
) -> PredictionExecution:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        from .model_provider import ProviderDeadlineError

        raise ProviderDeadlineError
    if spec.base_url is None:
        from .model_provider import ProviderConfigurationError

        msg = "MODEL execution requires a base URL"
        raise ProviderConfigurationError(msg)
    kserve_spec = KServeModelSpec(
        base_url=str(spec.base_url),
        model_name=spec.model_name,
        model_version=spec.model_version,
        input_name=spec.input_name,
        output_name=spec.output_name,
        task=spec.task,
    )
    from .kserve_v2_http import KServeV2HttpPredictionProvider

    provider = KServeV2HttpPredictionProvider.connect(
        kserve_spec,
        remaining,
        transport if transport is not None else get_transport_config(),
    )
    input_shape = provider.metadata.input_shape
    expected_width = (
        1 if input_shape == (-1,) else input_shape[-1] if input_shape else None
    )
    if (
        expected_width is not None
        and expected_width > 0
        and (
            data.instance.shape[0] != expected_width
            or data.background.shape[1] != expected_width
        )
    ):
        provider.close()
        msg = "Stored data feature width does not match model input"
        raise LocalDataError(msg)

    def predict(values: np.ndarray) -> np.ndarray:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            from .model_provider import ProviderDeadlineError

            raise ProviderDeadlineError
        return provider.predict(values, timeout_seconds=remaining)

    return PredictionExecution(
        prediction_callable(
            predict, spec.task, allow_single_probability=allow_single_probability
        ),
        provider=provider,
        resolved_input_name=provider.metadata.input_name,
        resolved_output_name=provider.metadata.output_name,
        metadata=provider.metadata,
        task=spec.task,
        source=spec.prediction_source,
    )


def create_execution(
    config: object,
    data: LocalExplanationData,
    timeout_seconds: float,
    *,
    transport: HttpTransportConfig | None = None,
    allow_single_probability: bool = False,
) -> PredictionExecution:
    """Compatibility wrapper for callers that still provide a duration."""
    spec = LocalExecutionSpec(
        prediction_source=config.prediction_source,
        base_url=str(config.base_url) if config.base_url is not None else None,
        model_name=config.model_name,
        model_version=config.model_version,
        input_name=config.input_name,
        output_name=config.output_name,
        task=config.task,
    )
    return create_prediction_execution(
        spec,
        data,
        time.monotonic() + timeout_seconds,
        transport=transport,
        allow_single_probability=allow_single_probability,
    )
