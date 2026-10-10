"""LIME endpoint orchestration for local model explanations."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from trustyai_service.core.explainers.local.lime import (
    _LIME_AVAILABLE,
    compute_lime_confidence_intervals,
    compute_lime_explanation,
    create_lime_explainer,
)
from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_models import (
    MAX_PREDICTION_ID_LENGTH,
    LocalExplanationModelConfig,
)
from trustyai_service.endpoints.explainers.local_models import (
    validate_prediction_id as validate_prediction_id_value,
)
from trustyai_service.service.data.local_explanation import load_local_explanation_data
from trustyai_service.service.explainers.local.error_mapping import map_error
from trustyai_service.service.explainers.local.execution import (
    LocalExecutionSpec,
    create_prediction_execution,
)
from trustyai_service.service.explainers.local.model_provider import (
    ProviderInvalidRequestError,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType
from trustyai_service.service.explainers.local.worker import run_local_worker

router = APIRouter()
logger = logging.getLogger(__name__)


class LimeExplainerConfig(BaseModel):
    """Configuration for LIME sampling and confidence estimation."""

    num_samples: int = Field(5000, ge=1, le=100_000)
    n_training_rows: int = Field(10_000, ge=1, le=100_000)
    num_features: int = Field(10, ge=1, le=1000)
    kernel_width: float = Field(0.75, gt=0, le=10)
    timeout: int = Field(300, ge=1, le=3600)
    confidence: float = Field(0.95, gt=0, le=1)
    seed: int | None = None
    class_index: int | None = Field(default=None, ge=0)


class LimeExplanationConfig(BaseModel):
    """Combined model and LIME configuration."""

    model: LocalExplanationModelConfig
    explainer: LimeExplainerConfig | None = None


class LimeExplanationRequest(BaseModel):
    """Request for one local LIME explanation."""

    predictionId: str = Field(min_length=1, max_length=MAX_PREDICTION_ID_LENGTH)
    config: LimeExplanationConfig

    @field_validator("predictionId")
    @classmethod
    def validate_prediction_id(cls, value: str) -> str:
        """Reject control characters before the ID is logged or queried."""
        return validate_prediction_id_value(value)


class LIMEFeatureAttribution(BaseModel):
    """One feature's LIME weight and optional confidence bounds."""

    feature_name: str
    importance: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None


class LIMEExplanationResponse(BaseModel):
    """Serialized local LIME explanation response."""

    prediction_id: str
    model: str
    prediction_source: PredictionSource
    attributions: list[LIMEFeatureAttribution]
    score: float
    local_prediction: float
    intercept: float
    task: TaskType
    output_name: str | None = None
    prediction_output: float | list[float] | None = None
    class_index: int | None = None


def _select_lime_label(
    prediction: np.ndarray, task: TaskType, class_index: int | None
) -> int | None:
    """Validate or infer the LIME class label for one model prediction."""
    if task is not TaskType.CLASSIFICATION:
        return None
    if class_index is not None and not 0 <= class_index < prediction.shape[1]:
        msg = "class_index is invalid for model output"
        raise ProviderInvalidRequestError(msg)
    return class_index if class_index is not None else int(np.argmax(prediction[0]))


@router.post(routes.EXPLAINER_LOCAL_LIME, response_model=LIMEExplanationResponse)
async def local_lime_explanation(
    request: LimeExplanationRequest,
) -> LIMEExplanationResponse:
    """Compute a local LIME explanation using a model or explicit surrogate."""
    if not _LIME_AVAILABLE:
        raise HTTPException(503, "LIME dependency is unavailable")
    config = request.config.explainer or LimeExplainerConfig()
    model = request.config.model
    deadline = time.monotonic() + config.timeout

    def remaining() -> float:
        return max(0.001, deadline - time.monotonic())

    try:
        data = await asyncio.wait_for(
            load_local_explanation_data(
                model.model_name,
                request.predictionId,
                config.n_training_rows,
                include_stored_output=model.prediction_source
                is PredictionSource.SURROGATE,
            ),
            remaining(),
        )
    except TimeoutError as exc:
        logger.warning(
            "local_explanation_failed",
            extra={
                "explainer": "LIME",
                "model_name": model.model_name,
                "prediction_id": request.predictionId,
                "final_status": 504,
            },
        )
        raise HTTPException(504, "Explanation exceeded its deadline") from exc
    except Exception as exc:
        mapped = map_error(exc)
        logger.warning(
            "local_explanation_failed",
            extra={
                "explainer": "LIME",
                "model_name": model.model_name,
                "prediction_id": request.predictionId,
                "final_status": mapped.status_code,
            },
        )
        raise HTTPException(mapped.status_code, mapped.detail) from exc

    def compute() -> tuple[
        list[tuple[str, float]],
        float,
        float,
        float,
        dict[str, float] | None,
        dict[str, float] | None,
        PredictionSource,
        float | list[float] | None,
        int | None,
        str | None,
        float,
        int,
    ]:
        execution = create_prediction_execution(
            LocalExecutionSpec(
                prediction_source=model.prediction_source,
                base_url=str(model.base_url) if model.base_url is not None else None,
                model_name=model.model_name,
                model_version=model.model_version,
                input_name=model.input_name,
                output_name=model.output_name,
                task=model.task,
            ),
            data,
            deadline,
        )
        try:
            mode = (
                "classification"
                if model.task is TaskType.CLASSIFICATION
                else "regression"
            )
            algorithm = create_lime_explainer(
                data.background,
                data.feature_names,
                mode,
                kernel_width=config.kernel_width,
                seed=config.seed,
            )
            prediction = execution.predict_fn(data.instance.reshape(1, -1))
            selected_label = _select_lime_label(
                prediction, model.task, config.class_index
            )
            result = compute_lime_explanation(
                algorithm,
                data.instance.astype(float),
                execution.predict_fn,
                num_samples=config.num_samples,
                num_features=config.num_features,
                label=selected_label,
            )
            lower, upper = compute_lime_confidence_intervals(
                data.background,
                data.feature_names,
                mode,
                data.instance.astype(float),
                execution.predict_fn,
                config.confidence,
                num_samples=config.num_samples,
                num_features=config.num_features,
                kernel_width=config.kernel_width,
                seed=config.seed,
                label=selected_label,
            )
            output_value = prediction.reshape(-1)
            serialized_output = (
                float(output_value[0])
                if output_value.size == 1
                else output_value.astype(float).tolist()
            )
            return (
                *result,
                lower,
                upper,
                execution.source,
                serialized_output,
                selected_label,
                execution.resolved_output_name,
                float(getattr(execution.provider, "provider_latency", 0.0)),
                int(getattr(execution.provider, "inference_batch_count", 0)),
            )
        finally:
            execution.close()

    try:
        (
            weights,
            score,
            local_prediction,
            intercept,
            lower,
            upper,
            source,
            prediction_output,
            class_index,
            output_name,
            provider_latency,
            inference_batch_count,
        ) = await run_local_worker(compute, remaining())
    except TimeoutError as exc:
        logger.warning(
            "local_explanation_failed",
            extra={
                "explainer": "LIME",
                "model_name": model.model_name,
                "prediction_id": request.predictionId,
                "final_status": 504,
            },
        )
        raise HTTPException(504, "Explanation exceeded its deadline") from exc
    except Exception as exc:
        mapped = map_error(exc)
        logger.warning(
            "local_explanation_failed",
            extra={
                "explainer": "LIME",
                "model_name": model.model_name,
                "model_version": model.model_version,
                "prediction_id": request.predictionId,
                "prediction_source": model.prediction_source.value,
                "final_status": mapped.status_code,
            },
        )
        raise HTTPException(mapped.status_code, mapped.detail) from exc
    logger.info(
        "local_explanation_complete",
        extra={
            "explainer": "LIME",
            "model_name": model.model_name,
            "model_version": model.model_version,
            "prediction_id": request.predictionId,
            "prediction_source": source.value,
            "provider_latency": provider_latency,
            "inference_batch_count": inference_batch_count,
            "final_status": 200,
        },
    )
    return LIMEExplanationResponse(
        prediction_id=request.predictionId,
        model=model.model_name,
        prediction_source=source,
        attributions=[
            LIMEFeatureAttribution(
                feature_name=name,
                importance=value,
                confidence_lower=lower.get(name) if lower else None,
                confidence_upper=upper.get(name) if upper else None,
            )
            for name, value in weights
        ],
        score=score,
        local_prediction=local_prediction,
        intercept=intercept,
        task=model.task,
        output_name=output_name,
        prediction_output=prediction_output,
        class_index=class_index,
    )
