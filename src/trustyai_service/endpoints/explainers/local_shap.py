"""KernelSHAP endpoint orchestration for local model explanations."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import StrEnum
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from trustyai_service.core.explainers.local.shap import (
    _SHAP_AVAILABLE,
    compute_confidence_intervals,
    compute_shap_result,
)
from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_models import (
    LocalExplanationModelConfig,
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
from trustyai_service.service.explainers.local.prediction_adapter import (
    selected_class_callable,
    selected_scalar_callable,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType
from trustyai_service.service.explainers.local.worker import run_local_worker

router = APIRouter()
logger = logging.getLogger(__name__)
_BINARY_CLASS_COUNT = 2


class LinkType(StrEnum):
    LOGIT = "LOGIT"
    IDENTITY = "IDENTITY"


class RegularizerType(StrEnum):
    AUTO = "AUTO"
    AIC = "AIC"
    BIC = "BIC"
    TOP_N_FEATURES = "TOP_N_FEATURES"
    NONE = "NONE"


class SHAPExplainerConfig(BaseModel):
    n_samples: int = Field(300, ge=1, le=100_000)
    n_training_rows: int = Field(10_000, ge=1, le=100_000)
    timeout: int = Field(300, ge=1, le=3600)
    link: LinkType = LinkType.IDENTITY
    regularizer: RegularizerType = RegularizerType.AUTO
    confidence: float = Field(0.95, gt=0, le=1)
    track_counterfactuals: bool = False
    class_index: int | None = Field(default=None, ge=0)
    single_probability: bool = False


class SHAPExplanationConfig(BaseModel):
    model: LocalExplanationModelConfig
    explainer: SHAPExplainerConfig | None = None


class SHAPExplanationRequest(BaseModel):
    predictionId: str = Field(min_length=1)
    config: SHAPExplanationConfig


@router.post(routes.EXPLAINER_LOCAL_SHAP)
async def local_shap_explanation(request: SHAPExplanationRequest) -> dict[str, Any]:
    if not _SHAP_AVAILABLE:
        raise HTTPException(503, "SHAP dependency is unavailable")
    config = request.config.explainer or SHAPExplainerConfig()
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
                "explainer": "SHAP",
                "model_name": model.model_name,
                "prediction_id": request.predictionId,
                "final_status": 504,
            },
        )
        raise HTTPException(504, "Explanation exceeded its deadline") from exc
    except LookupError as exc:
        logger.warning(
            "local_explanation_failed",
            extra={
                "explainer": "SHAP",
                "model_name": model.model_name,
                "prediction_id": request.predictionId,
                "final_status": 404,
            },
        )
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        logger.warning(
            "local_explanation_failed",
            extra={
                "explainer": "SHAP",
                "model_name": model.model_name,
                "prediction_id": request.predictionId,
                "final_status": 400,
            },
        )
        raise HTTPException(400, str(exc)) from exc

    def compute() -> tuple[
        np.ndarray,
        float,
        np.ndarray | None,
        np.ndarray | None,
        PredictionSource,
        float | list[float] | None,
        int | None,
        str | None,
        float,
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
            allow_single_probability=config.single_probability,
        )
        try:
            prediction = execution.predict_fn(data.instance.reshape(1, -1))
            selected_predict = execution.predict_fn
            selected_label = None
            if model.task is TaskType.CLASSIFICATION:
                if prediction.shape[1] == 1 and config.single_probability:
                    selected_label = 1
                elif prediction.shape[1] == _BINARY_CLASS_COUNT:
                    selected_label = (
                        config.class_index if config.class_index is not None else 1
                    )
                elif config.class_index is None:
                    raise ProviderInvalidRequestError(
                        "class_index is required for multi-class SHAP"
                    )
                else:
                    selected_label = config.class_index
                if selected_label >= prediction.shape[1] and not (
                    prediction.shape[1] == 1 and selected_label == 1
                ):
                    raise ProviderInvalidRequestError(
                        "class_index is invalid for model output"
                    )

                selected_predict = selected_class_callable(
                    execution.predict_fn,
                    selected_label,
                    link=config.link.value,
                    single_probability=config.single_probability,
                )
                # Validate the selected target before SHAP creates any samples.
                selected_predict(data.instance.reshape(1, -1))
            else:
                selected_predict = selected_scalar_callable(
                    execution.predict_fn, link=config.link.value
                )
                selected_predict(data.instance.reshape(1, -1))

            reg = (
                "num_features(10)"
                if config.regularizer is RegularizerType.TOP_N_FEATURES
                else config.regularizer.value.lower()
            )
            result = compute_shap_result(
                data.instance.astype(float),
                data.background,
                selected_predict,
                n_samples=config.n_samples,
                link=config.link.value.lower(),
                l1_reg=reg,
            )
            lower, upper = compute_confidence_intervals(
                data.instance.astype(float),
                data.background,
                selected_predict,
                config.confidence,
                config.n_samples,
                config.link.value.lower(),
                reg,
            )
            output_value = prediction.reshape(-1)
            serialized_output = (
                float(output_value[0])
                if output_value.size == 1
                else output_value.astype(float).tolist()
            )
            return (
                result.values,
                result.base_value,
                lower,
                upper,
                execution.source,
                serialized_output,
                selected_label,
                execution.resolved_output_name,
                result.linked_prediction,
                float(getattr(execution.provider, "provider_latency", 0.0)),
                int(getattr(execution.provider, "inference_batch_count", 0)),
            )
        finally:
            execution.close()

    try:
        (
            values,
            base,
            lower,
            upper,
            source,
            prediction_output,
            class_index,
            output_name,
            linked_prediction_output,
            provider_latency,
            inference_batch_count,
        ) = await run_local_worker(compute, remaining())
    except TimeoutError as exc:
        logger.warning(
            "local_explanation_failed",
            extra={
                "explainer": "SHAP",
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
                "explainer": "SHAP",
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
            "explainer": "SHAP",
            "model_name": model.model_name,
            "model_version": model.model_version,
            "prediction_id": request.predictionId,
            "prediction_source": source.value,
            "provider_latency": provider_latency,
            "inference_batch_count": inference_batch_count,
            "final_status": 200,
        },
    )
    return {
        "prediction_id": request.predictionId,
        "model": model.model_name,
        "prediction_source": source,
        "task": model.task,
        "output_name": output_name,
        "prediction_output": prediction_output,
        "class_index": class_index,
        "shap_base_value": base,
        "base_value": base,
        "linked_prediction_output": linked_prediction_output,
        "attributions": [
            {
                "feature_name": name,
                "importance": float(value),
                "confidence_lower": float(lower[index])
                if lower is not None and index < len(lower)
                else None,
                "confidence_upper": float(upper[index])
                if upper is not None and index < len(upper)
                else None,
            }
            for index, (name, value) in enumerate(
                zip(data.feature_names, values, strict=False)
            )
        ],
    }
