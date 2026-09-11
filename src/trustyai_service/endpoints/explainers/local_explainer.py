"""Local explainer endpoint for instance-level explanation requests."""

import asyncio
import logging
from enum import StrEnum
from http import HTTPStatus
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from trustyai_service.core.explainers.lime import (
    _LIME_AVAILABLE,
    compute_lime_confidence_intervals,
    compute_lime_explanation,
    create_lime_explainer,
)
from trustyai_service.core.explainers.surrogate import build_surrogate
from trustyai_service.endpoints import routes
from trustyai_service.service.constants import (
    INPUT_SUFFIX,
    METADATA_SUFFIX,
    OUTPUT_SUFFIX,
)
from trustyai_service.service.data.model_data import ModelData
from trustyai_service.service.data.shared_data_source import get_shared_data_source
from trustyai_service.service.data.storage import get_storage_interface

router = APIRouter()
logger = logging.getLogger(__name__)

# Module-level storage — same pattern as endpoints/metadata.py
storage_interface = get_storage_interface()

# Metadata column indices (same as in metadata.py)
METADATA_ID_COL = 0


class ModelConfig(BaseModel):
    """Model configuration for explainer requests."""

    target: str
    name: str
    version: str | None = None


class LimeExplainerConfig(BaseModel):
    """LIME explainer configuration parameters."""

    num_samples: int = Field(
        default=5000,
        description="Number of perturbation samples for LIME.",
    )
    n_training_rows: int = Field(
        default=1000,
        description="Maximum number of organic observations used to train the surrogate model.",
    )
    kernel_width: float = Field(
        default=0.75,
        description="Kernel width for proximity weighting in LIME.",
    )
    num_features: int = Field(
        default=10,
        description="Number of top features to return in the explanation.",
    )
    timeout: int = Field(
        default=300,
        description="Computation timeout in seconds.",
    )
    confidence: float = Field(
        default=0.95,
        description=(
            "Coverage of bootstrap confidence intervals (e.g. 0.95 for 95%)."
            " Set to 1.0 to skip confidence interval computation."
        ),
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is in (0, 1]; 1.0 is sentinel to disable CI."""
        if not (0.0 < v <= 1.0):
            msg = (
                "confidence must be in (0, 1]; use 1.0 to disable confidence intervals"
            )
            raise ValueError(msg)
        return v


class LimeExplanationConfig(BaseModel):
    """LIME explanation configuration."""

    model: ModelConfig
    explainer: LimeExplainerConfig | None = None


class LimeExplanationRequest(BaseModel):
    """LIME explanation request."""

    predictionId: str
    config: LimeExplanationConfig


class LIMEFeatureAttribution(BaseModel):
    """LIME attribution for a single input feature."""

    feature_name: str = Field(description="Name of the input feature.")
    importance: float = Field(
        description="Local importance weight for this feature from LIME."
    )
    confidence_lower: float | None = Field(
        default=None,
        description=(
            "Lower bound of the bootstrap confidence interval for the importance,"
            " or null when confidence intervals were not requested."
        ),
    )
    confidence_upper: float | None = Field(
        default=None,
        description=(
            "Upper bound of the bootstrap confidence interval for the importance,"
            " or null when confidence intervals were not requested."
        ),
    )


class LIMEExplanationResponse(BaseModel):
    """LIME explanation response containing per-feature local importance scores."""

    prediction_id: str = Field(description="ID of the explained stored prediction.")
    model: str = Field(description="Name of the model that produced the prediction.")
    attributions: list[LIMEFeatureAttribution] = Field(
        description="Per-feature LIME importances, one entry per returned feature."
    )
    score: float = Field(
        description="R² of the local linear model (goodness-of-fit metric)."
    )
    local_prediction: float = Field(description="Predicted value for this instance.")
    intercept: float = Field(description="Intercept of the local linear model.")


# ---------------------------------------------------------------------------
# Stored prediction retrieval
# ---------------------------------------------------------------------------


async def get_stored_prediction(
    model: str,
    prediction_id: str,
) -> tuple[np.ndarray, list[str]]:
    """Retrieve a stored input row by model name and prediction ID.

    Scans the metadata dataset for a row whose ID column matches
    ``prediction_id``, then reads the corresponding input row.

    Raises:
        HTTPException(404): Model has no stored data.
        HTTPException(400): ``prediction_id`` not found in the model's metadata.

    """
    metadata_dataset = model + METADATA_SUFFIX
    if not await storage_interface.dataset_exists(metadata_dataset):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No data found for model={model}.",
        )

    model_data = ModelData(model)
    _, _, metadata = await model_data.data(get_input=False, get_output=False)

    if metadata is None or len(metadata) == 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"No predictions stored for model={model}.",
        )

    row_idx: int | None = next(
        (
            i
            for i, row in enumerate(metadata)
            if str(row[METADATA_ID_COL]) == prediction_id
        ),
        None,
    )

    if row_idx is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Prediction ID {prediction_id!r} not found for model={model}.",
        )

    input_dataset = model + INPUT_SUFFIX
    input_data = await storage_interface.read_data(
        input_dataset, start_row=row_idx, n_rows=1
    )
    input_names = await storage_interface.get_aliased_column_names(input_dataset)

    row = np.asarray(input_data).flatten()
    return row, list(input_names)


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------


async def _prepare_training_data(
    model: str,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch organic observations and return (X, y, training_column_names).

    Raises:
        HTTPException(400): No input/output columns in organic data, or no rows after filtering.

    """
    data_source = get_shared_data_source()
    organic_df = await data_source.get_organic_dataframe(model, n_samples)

    input_cols = await storage_interface.get_aliased_column_names(model + INPUT_SUFFIX)
    output_cols = await storage_interface.get_aliased_column_names(
        model + OUTPUT_SUFFIX
    )

    training_cols = [c for c in input_cols if c in organic_df.columns]
    if not training_cols:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"No input columns found in organic data for model={model}.",
        )

    if not output_cols or output_cols[0] not in organic_df.columns:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"No output columns found in organic data for model={model}.",
        )

    x_train = organic_df[training_cols].values.astype(np.float64)
    y_train = organic_df[output_cols[0]].values

    if len(x_train) == 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"No organic observations for model={model}. Cannot train surrogate.",
        )

    return x_train, y_train, training_cols


# ---------------------------------------------------------------------------
# LIME endpoint handler
# ---------------------------------------------------------------------------


@router.post(routes.EXPLAINER_LOCAL_LIME)
async def local_lime_explanation(
    request: LimeExplanationRequest,
) -> LIMEExplanationResponse:
    """Compute a local LIME explanation for a stored prediction.

    LIME (Local Interpretable Model-Agnostic Explanations) provides per-feature
    importance weights for a single instance. It works by fitting a local linear
    model to perturbed variations of the input.

    Attributions explain a ``RandomForest`` surrogate trained on all stored
    organic observations, **not** the original model directly. The R² score
    indicates the quality of the local linear approximation. Confidence intervals
    are estimated via bootstrap resampling of the training data.
    Set ``config.explainer.confidence = 1.0`` to skip CI computation.

    """
    if not _LIME_AVAILABLE:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=(
                "explainability extra not installed."
                " Install with: pip install trustyai-service[explainability]"
            ),
        )

    logger.info("Computing LIME explanation for prediction: %s", request.predictionId)

    config = request.config.explainer or LimeExplainerConfig()
    model = request.config.model.name

    instance, all_input_names = await get_stored_prediction(model, request.predictionId)
    x_train, y_train, training_cols = await _prepare_training_data(
        model, config.n_training_rows
    )

    # Validate training columns exist in stored instance
    missing = set(training_cols) - set(all_input_names)
    if missing:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Training columns not found in stored prediction: {missing}",
        )

    # Slice instance to the training feature set (guards against organic_df column drift)
    name_to_idx = {name: i for i, name in enumerate(all_input_names)}
    col_idx = [name_to_idx[c] for c in training_cols]
    instance_f64 = instance[col_idx].astype(np.float64)

    # Detect regression vs classification
    mode = (
        "regression" if np.issubdtype(y_train.dtype, np.floating) else "classification"
    )

    def _compute() -> tuple[
        list[tuple[str, float]],
        float,
        float,
        float,
        dict[str, float] | None,
        dict[str, float] | None,
    ]:
        _surrogate = build_surrogate(x_train, y_train)
        # Use discretize_continuous=False to match CI bootstrap for stable feature names
        _explainer = create_lime_explainer(
            x_train,
            training_cols,
            mode,
            kernel_width=config.kernel_width,
            discretize_continuous=False,
        )
        # LIME requires probability outputs for classifiers, not class labels
        _predict_fn = (
            _surrogate.predict_proba if mode == "classification" else _surrogate.predict
        )
        _weights, _r2, _local_pred, _intercept = compute_lime_explanation(
            _explainer,
            instance_f64,
            _predict_fn,
            num_samples=config.num_samples,
            num_features=config.num_features,
        )
        _lower, _upper = compute_lime_confidence_intervals(
            x_train,
            training_cols,
            mode,
            instance_f64,
            _predict_fn,
            confidence=config.confidence,
            num_samples=config.num_samples,
            num_features=config.num_features,
            kernel_width=config.kernel_width,
        )
        return _weights, _r2, _local_pred, _intercept, _lower, _upper

    try:
        weights, r2, local_pred, intercept, lower, upper = await asyncio.wait_for(
            asyncio.to_thread(_compute),
            timeout=config.timeout,
        )
    except TimeoutError as exc:
        msg = f"LIME computation exceeded {config.timeout}s timeout."
        raise HTTPException(
            status_code=HTTPStatus.GATEWAY_TIMEOUT,
            detail=msg,
        ) from exc
    except (ValueError, KeyError) as exc:
        msg = f"LIME computation failed: {exc}"
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=msg,
        ) from exc

    attributions = [
        LIMEFeatureAttribution(
            feature_name=feature_name,
            importance=importance,
            confidence_lower=lower.get(feature_name) if lower is not None else None,
            confidence_upper=upper.get(feature_name) if upper is not None else None,
        )
        for feature_name, importance in weights
    ]

    return LIMEExplanationResponse(
        prediction_id=request.predictionId,
        model=model,
        attributions=attributions,
        score=r2,
        local_prediction=local_pred,
        intercept=intercept,
    )


class LinkType(StrEnum):
    """SHAP link function types."""

    LOGIT = "LOGIT"
    IDENTITY = "IDENTITY"


class RegularizerType(StrEnum):
    """SHAP regularizer types."""

    AUTO = "AUTO"
    AIC = "AIC"
    BIC = "BIC"
    TOP_N_FEATURES = "TOP_N_FEATURES"
    NONE = "NONE"


class SHAPExplainerConfig(BaseModel):
    """SHAP explainer configuration parameters."""

    n_samples: int = 300
    timeout: int = 10
    link: LinkType = LinkType.IDENTITY
    regularizer: RegularizerType = RegularizerType.AUTO
    confidence: float = 0.95
    track_counterfactuals: bool = False


class SHAPExplanationConfig(BaseModel):
    """SHAP explanation configuration."""

    model: ModelConfig
    explainer: SHAPExplainerConfig | None = None


class SHAPExplanationRequest(BaseModel):
    """SHAP explanation request."""

    predictionId: str
    config: SHAPExplanationConfig


@router.post(routes.EXPLAINER_LOCAL_SHAP)
async def local_shap_explanation(request: SHAPExplanationRequest) -> dict[str, Any]:
    """Compute a SHAP explanation."""
    logger.info(
        "Computing SHAP explanation for prediction: %s",
        request.predictionId,
    )
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Local SHAP explanation is not yet implemented",
    )


class CounterfactualExplainerConfig(BaseModel):
    """Counterfactual explainer configuration parameters."""

    n_samples: int = 100


class CounterfactualExplanationConfig(BaseModel):
    """Counterfactual explanation configuration."""

    model: ModelConfig
    explainer: CounterfactualExplainerConfig | None = None


class CounterfactualExplanationRequest(BaseModel):
    """Counterfactual explanation request."""

    predictionId: str
    config: CounterfactualExplanationConfig
    goals: dict[str, str] | None = None
    explanationConfig: CounterfactualExplanationConfig | None = None


@router.post(routes.EXPLAINER_LOCAL_CF)
async def local_counterfactual_explanation(
    request: CounterfactualExplanationRequest,
) -> dict[str, Any]:
    """Compute a Counterfactual explanation."""
    logger.info(
        "Computing Counterfactual explanation for prediction: %s",
        request.predictionId,
    )
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Local Counterfactual explanation is not yet implemented",
    )


class TSSaliencyExplainerConfig(BaseModel):
    """Time series saliency explainer configuration parameters."""

    timeout: int = 10
    mu: float = 0.01
    n_samples: int = 50
    n_alpha: int = 50
    sigma: float = 50
    base_values: list[float] | None = None


class TSSaliencyExplanationConfig(BaseModel):
    """Time series saliency explanation configuration."""

    model: ModelConfig
    explainer: TSSaliencyExplainerConfig | None = None


class TSSaliencyExplanationRequest(BaseModel):
    """Time series saliency explanation request."""

    predictionIds: list[str]
    config: TSSaliencyExplanationConfig


@router.post(routes.EXPLAINER_LOCAL_TSSALIENCY)
async def local_tssaliency_explanation(
    request: TSSaliencyExplanationRequest,
) -> dict[str, Any]:
    """Compute a TSSaliency explanation."""
    logger.info(
        "Computing TSSaliency explanation for predictions: %s",
        request.predictionIds,
    )
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Local TSSaliency explanation is not yet implemented",
    )
