"""Create one prediction path for a local explanation."""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from .error_mapping import LocalDataError
from .model_provider import (
    ProviderDeadlineError,
    ProviderInvalidRequestError,
)
from .prediction_adapter import prediction_callable
from .transport_config import get_transport_config

_MODEL_PROVIDER_MODULE = "trustyai_service.service.explainers.local.kserve_v2_http"
_MATRIX_RANK = 2

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from trustyai_service.service.data.local_explanation import LocalExplanationData

    from .model_provider import (
        HttpTransportConfig,
        PredictionMetadata,
        PredictionProvider,
    )
    from .types import TaskType


def _remaining_deadline(deadline: float) -> float:
    """Return the positive time remaining for an absolute monotonic deadline."""
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ProviderDeadlineError
    return remaining


def _load_model_provider() -> tuple[type, type]:
    """Load the outbound model-provider module when execution is created."""
    module = importlib.import_module(_MODEL_PROVIDER_MODULE)
    provider_type = cast("type", module.KServeV2HttpPredictionProvider)  # type: ignore[attr-defined]
    model_spec_type = cast("type", module.KServeModelSpec)  # type: ignore[attr-defined]
    return provider_type, model_spec_type


def _close_provider(provider: object) -> None:
    """Best-effort cleanup for a provider created before execution construction failed."""
    try:
        close = cast("PredictionProvider", provider).close
        close()
    except Exception:  # noqa: BLE001
        return


@dataclass(frozen=True)
class LocalExecutionSpec:
    """Immutable model and task selection for one explanation."""

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
    provider: PredictionProvider
    metadata: PredictionMetadata
    resolved_input_name: str | None
    resolved_output_name: str | None
    task: TaskType
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Close the owned provider at most once, including partial executions."""
        if self._closed:
            return
        self._closed = True
        _close_provider(self.provider)


def create_prediction_execution(
    spec: LocalExecutionSpec,
    data: LocalExplanationData,
    deadline: float,
    transport: HttpTransportConfig | None,
) -> PredictionExecution:
    """Create a model-provider prediction execution for one explanation."""
    return _create_model_execution(spec, data, deadline, transport)


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
        msg = "Model execution requires a base URL"
        raise ProviderInvalidRequestError(msg)

    resolved_transport = transport if transport is not None else get_transport_config()

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
