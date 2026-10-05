"""Local KernelSHAP explanations using the deployed model over KServe V2.

This route evaluates generated coalitions through the shared model execution's
outbound KServe V2 provider, with one deadline for the complete explanation.
"""

import asyncio
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from numbers import Real

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from trustyai_service.core.explainers.local.shap import (
    _MAX_NUM_FEATURES_REGULARIZER,
    _SHAP_AVAILABLE,
    SHAPAllocationError,
    ShapExplanationResult,
    SHAPUnavailableError,
    compute_confidence_intervals,
    compute_shap_result,
)
from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_models import (
    LocalExplanationModelConfig,
)
from trustyai_service.service.data.local_explanation import (
    LocalExplanationData,
    load_local_explanation_data,
)
from trustyai_service.service.explainers.local.error_mapping import (
    ErrorResponse,
    LocalDataError,
    LocalDataNotFoundError,
    map_error,
)
from trustyai_service.service.explainers.local.execution import (
    LocalExecutionSpec,
    PredictionExecution,
    create_prediction_execution,
)
from trustyai_service.service.explainers.local.model_provider import (
    ProviderDeadlineError,
    ProviderInvalidRequestError,
    ProviderRequestLimitError,
)
from trustyai_service.service.explainers.local.observability import (
    LocalExplanationObservability,
)
from trustyai_service.service.explainers.local.prediction_adapter import (
    normalize_predictions,
    selected_class_callable,
    selected_scalar_callable,
)
from trustyai_service.service.explainers.local.types import TaskType
from trustyai_service.service.explainers.local.worker import run_local_worker

logger = logging.getLogger(__name__)
router = APIRouter()

_MISSING_DATA_DETAIL = "Local explanation data was not found"
_LOADER_MISSING_DATA_DETAIL = "No stored data exists for the requested model"
_LOADER_DATA_ERROR_PREFIXES = (
    "No organic background data",
    "Prediction ID ",
    "Stored ",
)
_SINGLE_PROBABILITY_DETAIL = (
    "One-column classification requires single_probability=true and class_index=1"
)
_SHAP_ALLOCATION_DETAIL = (
    "SHAP request exceeds the memory budget; reduce n_samples or n_training_rows"
)
_WIDER_CLASS_DETAIL = "class_index is required for wider classification output"
_CLASS_INDEX_DETAIL = "class_index is invalid for model output"
_CLASS_INDEX_TASK_DETAIL = "class_index is only valid for classification"
_SUPPORTED_REGULARIZERS = frozenset({"auto", "aic", "bic"})
_NUM_FEATURES_REGULARIZER = re.compile(
    rf"num_features\([1-9][0-9]{{0,{len(str(_MAX_NUM_FEATURES_REGULARIZER)) - 1}}}\)"
)
_MATRIX_RANK = 2
_VECTOR_RANK = 1
_SINGLETON = 1
_BINARY_CLASS_COUNT = 2


def _unsupported_regularizer_value() -> ValueError:
    """Create the Pydantic-compatible error for unsupported JSON types."""
    msg = "regularizer must be a supported string or finite non-negative number"
    return ValueError(msg)


class LinkType(StrEnum):
    """Output link applied exactly once by KernelSHAP."""

    IDENTITY = "IDENTITY"
    LOGIT = "LOGIT"


def _normalize_regularizer(value: object) -> str | float:
    """Validate and normalize the regularizer accepted by the SHAP core."""
    if isinstance(value, bool):
        raise _unsupported_regularizer_value()
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in _SUPPORTED_REGULARIZERS:
            return normalized
        if _NUM_FEATURES_REGULARIZER.fullmatch(normalized):
            feature_count = normalized[len("num_features(") : -1]
            if int(feature_count) <= _MAX_NUM_FEATURES_REGULARIZER:
                return normalized
        msg = (
            "regularizer must be 'auto', 'aic', 'bic', 'num_features(n)', "
            "or a finite non-negative number"
        )
        raise ValueError(msg)
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            msg = "regularizer must be a finite non-negative number"
            raise ValueError(msg)
        return numeric
    raise _unsupported_regularizer_value()


class SHAPExplainerConfig(BaseModel):
    """Configuration for one bounded KernelSHAP explanation."""

    model_config = ConfigDict(extra="forbid")

    n_samples: int = Field(default=300, ge=1, le=100_000)
    n_training_rows: int = Field(default=10_000, ge=1, le=100_000)
    timeout: float = Field(default=300, gt=0, le=3600)
    link: LinkType = LinkType.IDENTITY
    regularizer: str | float = "auto"
    confidence: float | None = Field(default=None, gt=0, le=1)
    n_bootstrap: StrictInt = Field(default=10, ge=1, le=50)
    seed: StrictInt | None = Field(default=None, ge=0)
    class_index: StrictInt | None = Field(default=None, ge=0)
    single_probability: bool = False

    @field_validator("regularizer", mode="before")
    @classmethod
    def validate_regularizer(cls, value: object) -> str | float:
        """Keep invalid regularizers at the request-validation boundary."""
        return _normalize_regularizer(value)


class SHAPExplanationConfig(BaseModel):
    """Combined canonical model and KernelSHAP configuration."""

    model_config = ConfigDict(extra="forbid")

    model: LocalExplanationModelConfig
    explainer: SHAPExplainerConfig | None = None


class SHAPExplanationRequest(BaseModel):
    """Request for one stored-prediction KernelSHAP explanation."""

    model_config = ConfigDict(extra="forbid")

    predictionId: str = Field(min_length=1)
    config: SHAPExplanationConfig


class SHAPFeatureAttribution(BaseModel):
    """One feature attribution and optional confidence bounds."""

    feature_name: str
    importance: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None


class SHAPExplanationResponse(BaseModel):
    """Serialized local KernelSHAP explanation with explicit link semantics.

    ``prediction_output`` is the raw selected output. ``shap_base_value`` and
    ``linked_prediction_output`` are both expressed in the requested link
    space, so their additivity is not confused with the raw prediction units.
    """

    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    model: str
    task: TaskType
    output_name: str | None = None
    class_index: int | None = None
    prediction_output: float
    shap_base_value: float
    linked_prediction_output: float
    attributions: list[SHAPFeatureAttribution]


@dataclass(frozen=True)
class _SHAPWorkerResult:
    """Values returned by the synchronous SHAP worker."""

    result: ShapExplanationResult
    lower_bounds: np.ndarray | None
    upper_bounds: np.ndarray | None
    prediction_output: float
    class_index: int | None
    output_name: str | None
    feature_names: tuple[str, ...]
    provider_latency: float | None
    inference_batch_count: int | None


def _remaining(deadline: float) -> float:
    """Return the positive duration remaining before one monotonic deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProviderDeadlineError
    return remaining


def _is_loader_data_error(error: ValueError) -> bool:
    """Recognize semantic data errors emitted by the shared storage loader."""
    message = str(error)
    return message.startswith(_LOADER_DATA_ERROR_PREFIXES)


def _load_data_in_worker(
    model: LocalExplanationModelConfig,
    request: SHAPExplanationRequest,
    config: SHAPExplainerConfig,
    deadline: float,
) -> LocalExplanationData:
    """Run the shared async storage loader inside the synchronous worker."""

    async def load_with_deadline() -> LocalExplanationData:
        """Load stored data against the caller's absolute deadline."""
        return await asyncio.wait_for(
            load_local_explanation_data(
                model.model_name,
                request.predictionId,
                config.n_training_rows,
            ),
            timeout=_remaining(deadline),
        )

    try:
        return asyncio.run(load_with_deadline())
    except TimeoutError as exc:
        raise ProviderDeadlineError from exc
    except LocalDataNotFoundError:
        raise
    except LocalDataError:
        raise
    except LookupError as exc:
        if type(exc) is not LookupError or str(exc) != _LOADER_MISSING_DATA_DETAIL:
            raise
        raise LocalDataNotFoundError(_MISSING_DATA_DETAIL) from exc
    except ValueError as exc:
        if not _is_loader_data_error(exc):
            raise
        raise LocalDataError(str(exc)) from exc


def _provider_observability(
    execution: PredictionExecution,
) -> tuple[float | None, int | None]:
    """Read existing provider counters without expanding the provider contract."""
    provider = execution.provider
    latency = getattr(provider, "provider_latency", None)
    provider_latency = (
        float(latency)
        if isinstance(latency, (int, float)) and not isinstance(latency, bool)
        else None
    )
    batch_count = getattr(provider, "inference_batch_count", None)
    inference_batch_count = (
        int(batch_count)
        if isinstance(batch_count, int) and not isinstance(batch_count, bool)
        else None
    )
    return provider_latency, inference_batch_count


def _raw_prediction_callable(
    execution: PredictionExecution,
    deadline: float,
) -> Callable[[np.ndarray], object]:
    """Use the provider's raw output for SHAP classification selection."""
    predict = execution.provider.predict

    def raw_predict(values: np.ndarray) -> object:
        """Call the provider with the remaining request deadline."""
        timeout_seconds = _remaining(deadline)
        result = predict(values, timeout_seconds=timeout_seconds)
        _remaining(deadline)
        return result

    return raw_predict


def _cached_instance_callable(
    raw_predict: Callable[[np.ndarray], object],
    instance: np.ndarray,
    raw_instance: object,
) -> Callable[[np.ndarray], object]:
    """Reuse the first raw singleton output when the core asks for the instance."""
    instance_row = instance.reshape(1, -1)

    def cached_predict(values: np.ndarray) -> object:
        """Return the cached singleton without issuing a duplicate model call."""
        values_array = np.asarray(values)
        if values_array.shape == instance_row.shape and np.array_equal(
            values_array, instance_row
        ):
            return np.array(raw_instance, copy=True)
        return raw_predict(values)

    return cached_predict


def _raw_width(value: object) -> int | None:
    """Return a raw one-row output width when it can be determined safely."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim == _MATRIX_RANK and array.shape[0] == _SINGLETON:
        return int(array.shape[1])
    if array.ndim == _VECTOR_RANK and array.shape[0] == _SINGLETON:
        return _SINGLETON
    return None


def _select_class_index(
    prediction: np.ndarray,
    requested: int | None,
    *,
    single_probability: bool,
) -> int:
    """Apply the SHAP class-selection contract to normalized raw scores."""
    width = prediction.shape[1]
    if width == 1:
        if not single_probability:
            raise ProviderInvalidRequestError(_SINGLE_PROBABILITY_DETAIL)
        if requested not in {None, 1}:
            raise ProviderInvalidRequestError(_SINGLE_PROBABILITY_DETAIL)
        return 1
    if width == _BINARY_CLASS_COUNT:
        selected = 1 if requested is None else requested
    else:
        if requested is None:
            raise ProviderInvalidRequestError(_WIDER_CLASS_DETAIL)
        selected = requested
    if selected < 0 or selected >= width:
        raise ProviderInvalidRequestError(_CLASS_INDEX_DETAIL)
    return selected


def _build_selected_callable(
    execution: PredictionExecution,
    data: LocalExplanationData,
    config: SHAPExplainerConfig,
    task: TaskType,
    deadline: float,
) -> tuple[Callable[[np.ndarray], np.ndarray], float, int | None]:
    """Create the adapter callable and compute its singleton output once."""
    raw_predict = _raw_prediction_callable(execution, deadline)
    instance = data.instance.astype(float, copy=False)
    instance_row = instance.reshape(1, -1)
    raw_instance = raw_predict(instance_row)
    _remaining(deadline)
    cached_predict = _cached_instance_callable(raw_predict, instance, raw_instance)

    if task is TaskType.CLASSIFICATION:
        if not config.single_probability and _raw_width(raw_instance) == 1:
            raise ProviderInvalidRequestError(_SINGLE_PROBABILITY_DETAIL)
        normalized = normalize_predictions(
            raw_instance,
            TaskType.CLASSIFICATION,
            1,
            allow_single_probability=config.single_probability,
        )
        selected_class = _select_class_index(
            normalized,
            config.class_index,
            single_probability=config.single_probability,
        )
        selected_predict = selected_class_callable(
            cached_predict,
            selected_class,
            link=config.link.value,
            single_probability=config.single_probability,
        )
        selected_output = selected_predict(instance_row)
        return selected_predict, float(selected_output[0]), selected_class

    if config.class_index is not None:
        raise ProviderInvalidRequestError(_CLASS_INDEX_TASK_DETAIL)
    selected_predict = selected_scalar_callable(
        cached_predict,
        link=config.link.value,
    )
    selected_output = selected_predict(instance_row)
    return selected_predict, float(selected_output[0]), None


def _worker(
    request: SHAPExplanationRequest,
    model: LocalExplanationModelConfig,
    config: SHAPExplainerConfig,
    deadline: float,
    observability: LocalExplanationObservability | None = None,
) -> _SHAPWorkerResult:
    """Load, execute, and compute SHAP in one provider-owning worker."""
    execution: PredictionExecution | None = None
    try:
        _remaining(deadline)
        data = _load_data_in_worker(model, request, config, deadline)
        _remaining(deadline)
        execution = create_prediction_execution(
            LocalExecutionSpec(
                base_url=str(model.base_url),
                model_name=model.model_name,
                model_version=model.model_version,
                input_name=model.input_name,
                output_name=model.output_name,
                task=model.task,
            ),
            data,
            deadline,
            None,
        )
        _remaining(deadline)
        selected_predict, prediction_output, selected_class = _build_selected_callable(
            execution,
            data,
            config,
            model.task,
            deadline,
        )
        _remaining(deadline)
        result = compute_shap_result(
            data.background.astype(float, copy=False),
            data.instance.astype(float, copy=False),
            selected_predict,
            instance_prediction=prediction_output,
            n_samples=config.n_samples,
            link=config.link.value.lower(),
            l1_reg=config.regularizer,
        )
        _remaining(deadline)
        lower_bounds: np.ndarray | None = None
        upper_bounds: np.ndarray | None = None
        if config.confidence is not None:
            lower_bounds, upper_bounds = compute_confidence_intervals(
                data.background.astype(float, copy=False),
                data.instance.astype(float, copy=False),
                selected_predict,
                instance_prediction=prediction_output,
                n_samples=config.n_samples,
                link=config.link.value.lower(),
                l1_reg=config.regularizer,
                confidence=config.confidence,
                n_bootstrap=config.n_bootstrap,
                seed=config.seed,
            )
            _remaining(deadline)
        provider_latency, inference_batch_count = _provider_observability(execution)
        return _SHAPWorkerResult(
            result=result,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            prediction_output=prediction_output,
            class_index=selected_class,
            output_name=execution.resolved_output_name,
            feature_names=tuple(data.feature_names),
            provider_latency=provider_latency,
            inference_batch_count=inference_batch_count,
        )
    finally:
        if execution is not None:
            if observability is not None:
                observability.capture(execution)
            execution.close()


def _raise_http_error(error: Exception) -> None:
    """Raise one safe FastAPI error produced by the shared endpoint mapper."""
    mapped: ErrorResponse = map_error(error)
    raise HTTPException(
        status_code=mapped.status_code,
        detail=mapped.as_http_detail(),
    ) from error


def _map_algorithm_error(error: Exception) -> Exception:
    """Translate bounded core resource failures to the endpoint error contract."""
    if isinstance(error, SHAPAllocationError):
        return ProviderRequestLimitError(_SHAP_ALLOCATION_DETAIL)
    return error


def _validate_shap_request(
    model: LocalExplanationModelConfig,
    config: SHAPExplainerConfig,
) -> None:
    """Validate semantic model and SHAP options before starting the worker."""
    if model.task is TaskType.REGRESSION and config.class_index is not None:
        raise ProviderInvalidRequestError(_CLASS_INDEX_TASK_DETAIL)


def _log_failure(
    *,
    model: LocalExplanationModelConfig,
    request: SHAPExplanationRequest,
    error: Exception,
    started: float,
    observability: LocalExplanationObservability,
) -> None:
    """Write safe structured failure fields without exception or payload details."""
    observability.capture(error)
    mapped = map_error(error)
    logger.warning(
        "local_explanation_failed",
        extra={
            "explainer": "SHAP",
            "model_name": model.model_name,
            "model_version": model.model_version,
            "prediction_id": request.predictionId,
            "task": model.task.value,
            "output_name": model.output_name,
            "class_index": (
                request.config.explainer.class_index
                if request.config.explainer is not None
                else None
            ),
            "final_status": mapped.status_code,
            "error_code": mapped.code,
            "latency_seconds": time.monotonic() - started,
            "provider_latency": observability.provider_latency,
            "inference_batch_count": observability.inference_batch_count,
        },
    )


@router.post(routes.EXPLAINER_LOCAL_SHAP)
async def local_shap_explanation(
    request: SHAPExplanationRequest,
) -> SHAPExplanationResponse:
    """Compute one local KernelSHAP explanation using the deployed model."""
    started = time.monotonic()
    model = request.config.model
    config = request.config.explainer or SHAPExplainerConfig()
    observability = LocalExplanationObservability()
    if not _SHAP_AVAILABLE:
        error = SHAPUnavailableError()
        _log_failure(
            model=model,
            request=request,
            error=error,
            started=started,
            observability=observability,
        )
        _raise_http_error(error)

    try:
        _validate_shap_request(model, config)
    except Exception as exc:  # noqa: BLE001
        error = _map_algorithm_error(exc)
        _log_failure(
            model=model,
            request=request,
            error=error,
            started=started,
            observability=observability,
        )
        _raise_http_error(error)

    deadline = started + config.timeout
    try:
        worker_result = await run_local_worker(
            lambda: _worker(request, model, config, deadline, observability),
            _remaining(deadline),
        )
    except Exception as exc:  # noqa: BLE001
        error = _map_algorithm_error(exc)
        _log_failure(
            model=model,
            request=request,
            error=error,
            started=started,
            observability=observability,
        )
        _raise_http_error(error)

    result = worker_result.result
    lower_bounds = worker_result.lower_bounds
    upper_bounds = worker_result.upper_bounds
    logger.info(
        "local_explanation_complete",
        extra={
            "explainer": "SHAP",
            "model_name": model.model_name,
            "model_version": model.model_version,
            "prediction_id": request.predictionId,
            "task": model.task.value,
            "output_name": worker_result.output_name,
            "class_index": worker_result.class_index,
            "latency_seconds": time.monotonic() - started,
            "provider_latency": worker_result.provider_latency,
            "inference_batch_count": worker_result.inference_batch_count,
            "final_status": HTTPStatus.OK,
            "error_code": None,
        },
    )
    return SHAPExplanationResponse(
        prediction_id=request.predictionId,
        model=model.model_name,
        task=model.task,
        output_name=worker_result.output_name,
        class_index=worker_result.class_index,
        prediction_output=worker_result.prediction_output,
        shap_base_value=result.base_value,
        linked_prediction_output=result.linked_prediction,
        attributions=[
            SHAPFeatureAttribution(
                feature_name=feature_name,
                importance=float(importance),
                confidence_lower=(
                    float(lower_bounds[index])
                    if lower_bounds is not None and index < len(lower_bounds)
                    else None
                ),
                confidence_upper=(
                    float(upper_bounds[index])
                    if upper_bounds is not None and index < len(upper_bounds)
                    else None
                ),
            )
            for index, (feature_name, importance) in enumerate(
                zip(worker_result.feature_names, result.values, strict=False)
            )
        ],
    )


__all__ = [
    "LinkType",
    "SHAPExplainerConfig",
    "SHAPExplanationConfig",
    "SHAPExplanationRequest",
    "SHAPExplanationResponse",
    "SHAPFeatureAttribution",
    "local_shap_explanation",
    "router",
]
