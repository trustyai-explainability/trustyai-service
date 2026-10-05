"""Local LIME explanations using predictions from the deployed model.

Generated explanation rows are sent to the outbound KServe V2 provider.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt

from trustyai_service.core.explainers.local.lime import (
    _LIME_AVAILABLE,
    LIMEAllocationError,
    LIMEExplanationResult,
    LIMEUnavailableError,
    compute_lime_confidence_intervals,
    compute_lime_explanation,
    create_lime_explainer,
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
    LocalWorkerCapacityError,
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
)
from trustyai_service.service.explainers.local.observability import (
    LocalExplanationObservability,
)
from trustyai_service.service.explainers.local.types import TaskType
from trustyai_service.service.explainers.local.worker import run_local_worker

logger = logging.getLogger(__name__)
router = APIRouter()

_MISSING_DATA_DETAIL = "Local explanation data was not found"
_LOADER_MISSING_DATA_DETAIL = "No stored data exists for the requested model"
_CLASS_INDEX_TASK_DETAIL = "class_index is only valid for classification"
_CLASS_INDEX_INVALID_DETAIL = "class_index is invalid for model output"
_WIDER_CLASS_DETAIL = "class_index is required for wider classification output"
_BINARY_CLASS_COUNT = 2
_LOADER_DATA_ERROR_PREFIXES = (
    "No organic background data",
    "Prediction ID ",
    "Stored ",
)


class LimeExplainerConfig(BaseModel):
    """Configuration for one bounded LIME explanation."""

    model_config = ConfigDict(extra="forbid")

    num_samples: int = Field(default=5000, ge=2, le=100_000)
    n_training_rows: int = Field(default=10_000, ge=1, le=100_000)
    kernel_width: FiniteFloat = Field(default=0.75, gt=0)
    num_features: int = Field(default=10, ge=1, le=1000)
    timeout: float = Field(default=300, gt=0, le=3600)
    confidence: FiniteFloat | None = Field(default=None, gt=0, le=1)
    n_bootstrap: StrictInt = Field(default=10, ge=1, le=50)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**32 - 1)
    class_index: StrictInt | None = Field(default=None, ge=0)


class LimeExplanationConfig(BaseModel):
    """Combined canonical model and algorithm configuration."""

    model_config = ConfigDict(extra="forbid")

    model: LocalExplanationModelConfig
    explainer: LimeExplainerConfig | None = None


class LimeExplanationRequest(BaseModel):
    """Request for one stored-prediction LIME explanation."""

    model_config = ConfigDict(extra="forbid")

    predictionId: str = Field(min_length=1)
    config: LimeExplanationConfig


class LIMEFeatureAttribution(BaseModel):
    """One feature attribution and optional bootstrap confidence bounds."""

    feature_name: str
    importance: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None


class LIMEExplanationResponse(BaseModel):
    """Serialized local LIME explanation with distinct raw and local values.

    ``prediction_output`` is the raw selected deployed-model output. In contrast,
    ``local_prediction`` is the local linear model value for the explained
    class and must not be interpreted as the deployed model prediction.
    """

    prediction_id: str
    model: str
    attributions: list[LIMEFeatureAttribution]
    score: float
    local_prediction: float
    intercept: float
    task: TaskType
    output_name: str | None = None
    prediction_output: float | list[float] | None = None
    class_index: int | None = None


@dataclass(frozen=True)
class _LIMEWorkerResult:
    """Values returned by the synchronous explanation worker."""

    result: LIMEExplanationResult
    lower_bounds: dict[str, float] | None
    upper_bounds: dict[str, float] | None
    prediction_output: float | list[float]
    class_index: int | None
    output_name: str | None
    provider_latency: float | None
    inference_batch_count: int | None


def _remaining(deadline: float) -> float:
    """Return the positive duration remaining before one monotonic deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProviderDeadlineError
    return remaining


def _is_loader_data_error(error: ValueError) -> bool:
    """Recognize plain semantic errors emitted by the shared loader."""
    message = str(error)
    return message.startswith(_LOADER_DATA_ERROR_PREFIXES)


def _load_data_in_worker(
    model: LocalExplanationModelConfig,
    request: LimeExplanationRequest,
    config: LimeExplainerConfig,
    deadline: float,
) -> LocalExplanationData:
    """Run the shared async storage loader inside the synchronous worker."""

    async def load_with_deadline() -> LocalExplanationData:
        """Run storage loading against the caller's absolute deadline."""
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


def _prediction_output(
    prediction: np.ndarray,
    task: TaskType,
) -> float | list[float]:
    """Serialize the complete normalized instance prediction."""
    if task is TaskType.CLASSIFICATION:
        return prediction[0].astype(float).tolist()
    return float(prediction.reshape(-1)[0])


def _select_class_index(
    prediction: np.ndarray,
    task: TaskType,
    requested: int | None,
) -> int | None:
    """Default binary output to class 1 and require selectors for wider output."""
    if task is not TaskType.CLASSIFICATION:
        if requested is not None:
            raise ProviderInvalidRequestError(_CLASS_INDEX_TASK_DETAIL)
        return None
    width = prediction.shape[1]
    if requested is None and width != _BINARY_CLASS_COUNT:
        raise ProviderInvalidRequestError(_WIDER_CLASS_DETAIL)
    selected = _BINARY_CLASS_COUNT - 1 if requested is None else requested
    if not 0 <= selected < width:
        raise ProviderInvalidRequestError(_CLASS_INDEX_INVALID_DETAIL)
    return selected


def _deadline_guarded_predict(
    predict_fn: Callable[[np.ndarray], np.ndarray], deadline: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Check the same absolute deadline around every explainer prediction."""

    def guarded_predict(values: np.ndarray) -> np.ndarray:
        """Stop model-backed work when the deadline expires."""
        _remaining(deadline)
        result = predict_fn(values)
        _remaining(deadline)
        return result

    return guarded_predict


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


def _worker(
    request: LimeExplanationRequest,
    model: LocalExplanationModelConfig,
    config: LimeExplainerConfig,
    deadline: float,
    observability: LocalExplanationObservability | None = None,
) -> _LIMEWorkerResult:
    """Load, execute, and compute LIME in one provider-owning worker."""
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

        predict_fn = _deadline_guarded_predict(execution.predict_fn, deadline)
        instance_prediction = predict_fn(data.instance.reshape(1, -1))
        _remaining(deadline)
        selected_class = _select_class_index(
            instance_prediction,
            model.task,
            config.class_index,
        )
        _remaining(deadline)
        algorithm = create_lime_explainer(
            data.background,
            data.feature_names,
            classification=model.task is TaskType.CLASSIFICATION,
            kernel_width=config.kernel_width,
            seed=config.seed,
        )
        _remaining(deadline)
        explanation = compute_lime_explanation(
            algorithm,
            data.instance.astype(float),
            predict_fn,
            num_samples=config.num_samples,
            num_features=config.num_features,
            label=selected_class,
        )
        _remaining(deadline)
        lower_bounds: dict[str, float] | None = None
        upper_bounds: dict[str, float] | None = None
        if config.confidence is not None:
            confidence = compute_lime_confidence_intervals(
                algorithm,
                data.instance.astype(float),
                predict_fn,
                num_samples=config.num_samples,
                num_features=config.num_features,
                confidence=config.confidence,
                n_bootstrap=config.n_bootstrap,
                seed=config.seed,
                label=selected_class,
            )
            lower_bounds = confidence.lower_bounds
            upper_bounds = confidence.upper_bounds
            _remaining(deadline)
        prediction_output = _prediction_output(instance_prediction, model.task)
        _remaining(deadline)
        provider_latency, inference_batch_count = _provider_observability(execution)
        return _LIMEWorkerResult(
            result=explanation,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            prediction_output=prediction_output,
            class_index=selected_class,
            output_name=execution.resolved_output_name,
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
    if isinstance(error, LIMEAllocationError):
        return LocalWorkerCapacityError()
    return error


def _validate_lime_request(
    model: LocalExplanationModelConfig,
    config: LimeExplainerConfig,
) -> None:
    """Validate semantic model and LIME options before starting the worker."""
    if model.task is TaskType.REGRESSION and config.class_index is not None:
        raise ProviderInvalidRequestError(_CLASS_INDEX_TASK_DETAIL)


def _log_failure(
    *,
    model: LocalExplanationModelConfig,
    request: LimeExplanationRequest,
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
            "explainer": "LIME",
            "model_name": model.model_name,
            "model_version": model.model_version,
            "prediction_id": request.predictionId,
            "task": model.task.value,
            "final_status": mapped.status_code,
            "error_code": mapped.code,
            "latency_seconds": time.monotonic() - started,
            "provider_latency": observability.provider_latency,
            "inference_batch_count": observability.inference_batch_count,
        },
    )


@router.post(routes.EXPLAINER_LOCAL_LIME)
async def local_lime_explanation(
    request: LimeExplanationRequest,
) -> LIMEExplanationResponse:
    """Compute one local LIME explanation using predictions from the deployed model.

    Stored inputs provide the target and organic background rows. The deployed
    KServe V2 model predicts the target and every generated perturbation batch.
    """
    started = time.monotonic()
    model = request.config.model
    config = request.config.explainer or LimeExplainerConfig()
    observability = LocalExplanationObservability()
    if not _LIME_AVAILABLE:
        error = LIMEUnavailableError()
        _log_failure(
            model=model,
            request=request,
            error=error,
            started=started,
            observability=observability,
        )
        _raise_http_error(error)

    try:
        _validate_lime_request(model, config)
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
            "explainer": "LIME",
            "model_name": model.model_name,
            "model_version": model.model_version,
            "prediction_id": request.predictionId,
            "task": model.task.value,
            "latency_seconds": time.monotonic() - started,
            "provider_latency": worker_result.provider_latency,
            "inference_batch_count": worker_result.inference_batch_count,
            "final_status": HTTPStatus.OK,
            "error_code": None,
        },
    )
    return LIMEExplanationResponse(
        prediction_id=request.predictionId,
        model=model.model_name,
        attributions=[
            LIMEFeatureAttribution(
                feature_name=feature_name,
                importance=importance,
                confidence_lower=(
                    lower_bounds.get(feature_name) if lower_bounds is not None else None
                ),
                confidence_upper=(
                    upper_bounds.get(feature_name) if upper_bounds is not None else None
                ),
            )
            for feature_name, importance in result.feature_weights
        ],
        score=result.score,
        local_prediction=result.local_prediction,
        intercept=result.intercept,
        task=model.task,
        output_name=worker_result.output_name,
        prediction_output=worker_result.prediction_output,
        class_index=worker_result.class_index,
    )


__all__ = [
    "LIMEExplanationResponse",
    "LIMEFeatureAttribution",
    "LimeExplainerConfig",
    "LimeExplanationConfig",
    "LimeExplanationRequest",
    "local_lime_explanation",
    "router",
]
