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


@dataclass(frozen=True)
class LocalExecutionSpec:
    prediction_source: PredictionSource
    base_url: str | None
    model_name: str
    model_version: str | None
    input_name: str | None
    output_name: str | None
    task: TaskType


@dataclass
class PredictionExecution:
    predict_fn: Callable[[np.ndarray], np.ndarray]
    provider: PredictionProvider | None = None
    metadata: PredictionMetadata | None = None
    resolved_input_name: str | None = None
    resolved_output_name: str | None = None
    task: TaskType = TaskType.REGRESSION
    source: PredictionSource = PredictionSource.MODEL
    _closed: bool = False

    def close(self) -> None:
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
    source = spec.prediction_source
    task = spec.task
    if source is PredictionSource.SURROGATE:
        from trustyai_service.core.explainers.local.surrogate import build_surrogate

        if data.background_output is None:
            raise LocalDataError("Surrogate mode requires stored organic labels")
        targets = np.asarray(data.background_output)
        if targets.ndim == 1:
            if len(data.output_names) > 1 or (
                spec.output_name is not None
                and (not data.output_names or spec.output_name != data.output_names[0])
            ):
                raise LocalDataError(
                    "Surrogate output_name is required for the selected stored output"
                )
            selected_targets = targets
            selected_output = data.output_names[0] if data.output_names else None
        elif targets.ndim == 2:
            if spec.output_name is None:
                if targets.shape[1] != 1:
                    raise LocalDataError(
                        "Surrogate output_name is required for multiple outputs"
                    )
                selected_index = 0
            else:
                if spec.output_name not in data.output_names:
                    raise LocalDataError("Requested stored output was not found")
                selected_index = data.output_names.index(spec.output_name)
            selected_targets = targets[:, selected_index]
            selected_output = data.output_names[selected_index]
        else:
            raise LocalDataError(
                "Stored surrogate labels must be one- or two-dimensional"
            )
        if selected_targets.size == 0 or not np.isfinite(selected_targets).all():
            raise LocalDataError("Surrogate labels must be finite and non-empty")
        if task is TaskType.CLASSIFICATION:
            if len(np.unique(selected_targets)) < 2:
                raise LocalDataError(
                    "Surrogate classification labels must contain two finite classes"
                )
            if not np.all(selected_targets == np.floor(selected_targets)):
                raise LocalDataError("Surrogate classification labels must be discrete")
        try:
            estimator = build_surrogate(data.background, selected_targets, task.value)
        except Exception as exc:
            raise LocalExecutionError("Local surrogate training failed") from exc
        raw = (
            estimator.predict_proba
            if task is TaskType.CLASSIFICATION
            else estimator.predict
        )
        return PredictionExecution(
            prediction_callable(
                raw, task, allow_single_probability=allow_single_probability
            ),
            provider=None,
            resolved_output_name=selected_output,
            task=task,
            source=source,
        )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        from .model_provider import ProviderDeadlineError

        raise ProviderDeadlineError
    if spec.base_url is None:
        from .model_provider import ProviderConfigurationError

        raise ProviderConfigurationError("MODEL execution requires a base URL")
    spec = KServeModelSpec(
        base_url=str(spec.base_url),
        model_name=spec.model_name,
        model_version=spec.model_version,
        input_name=spec.input_name,
        output_name=spec.output_name,
        task=task,
    )
    from .kserve_v2_http import KServeV2HttpPredictionProvider

    provider = KServeV2HttpPredictionProvider.connect(
        spec,
        remaining,
        transport if transport is not None else get_transport_config(),
    )
    expected_width = (
        provider.metadata.input_shape[-1] if provider.metadata.input_shape else None
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
        raise LocalDataError("Stored data feature width does not match model input")

    def predict(values: np.ndarray) -> np.ndarray:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            from .model_provider import ProviderDeadlineError

            raise ProviderDeadlineError
        return provider.predict(values, timeout_seconds=remaining)

    return PredictionExecution(
        prediction_callable(
            predict, task, allow_single_probability=allow_single_probability
        ),
        provider=provider,
        resolved_input_name=provider.metadata.input_name,
        resolved_output_name=provider.metadata.output_name,
        metadata=provider.metadata,
        task=task,
        source=source,
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
