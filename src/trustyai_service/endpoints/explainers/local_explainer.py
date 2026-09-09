"""Local explainer endpoint for instance-level explanation requests."""

import logging
from enum import StrEnum
from http import HTTPStatus
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from trustyai_service.core.explainers.shap import (
    _SHAP_AVAILABLE,
    build_surrogate,
    compute_confidence_intervals,
    compute_shap_values,
)
from trustyai_service.endpoints import routes
from trustyai_service.service.constants import (
    INPUT_SUFFIX,
    METADATA_ID_COL,
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


# ---------------------------------------------------------------------------
# Shared model config
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Model configuration for explainer requests."""

    target: str
    name: str
    version: str | None = None


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


class LimeExplainerConfig(BaseModel):
    """LIME explainer configuration parameters."""

    n_samples: int = 300
    timeout: int = 10
    separable_dataset_ratio: float = 0.9
    retries: int = 3
    adaptive_variance: bool = True
    penalize_balance_sparse: bool = True
    proximity_filter: bool = True
    proximity_threshold: float = 0.83
    proximity_kernel_width: float = 0.5
    encoding_cluster_threshold: float = 0.07
    encoding_gaussian_filter_width: float = 0.07
    normalize_weights: bool = False
    high_score_feature_zones: bool = True
    feature_selection: bool = True
    n_features: int = 10
    track_counterfactuals: bool = False
    use_wlr_model: bool = True
    filter_interpretable: bool = False


class LimeExplanationConfig(BaseModel):
    """LIME explanation configuration."""

    model: ModelConfig
    explainer: LimeExplainerConfig | None = None


class LimeExplanationRequest(BaseModel):
    """LIME explanation request."""

    predictionId: str
    config: LimeExplanationConfig


@router.post(routes.EXPLAINER_LOCAL_LIME)
async def local_lime_explanation(request: LimeExplanationRequest) -> dict[str, Any]:
    """Compute a LIME explanation."""
    logger.info(
        "Computing LIME explanation for prediction: %s",
        request.predictionId,
    )
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Local LIME explanation is not yet implemented",
    )


# ---------------------------------------------------------------------------
# SHAP enums and config
# ---------------------------------------------------------------------------


class LinkType(StrEnum):
    """SHAP link function applied to model output before computing attributions."""

    LOGIT = "LOGIT"
    IDENTITY = "IDENTITY"


class RegularizerType(StrEnum):
    """L1 regularization strategy used by KernelSHAP to select non-zero attributions."""

    AUTO = "AUTO"
    AIC = "AIC"
    BIC = "BIC"
    TOP_N_FEATURES = "TOP_N_FEATURES"
    NONE = "NONE"


class SHAPExplainerConfig(BaseModel):
    """SHAP explainer configuration parameters."""

    n_samples: int = Field(
        default=300,
        description="Number of background samples used for Shapley value integration.",
    )
    timeout: int = Field(
        default=10,
        description="Computation timeout in seconds. Reserved for future use; not currently enforced.",
    )
    link: LinkType = Field(
        default=LinkType.IDENTITY,
        description="Link function applied to model output (IDENTITY or LOGIT).",
    )
    regularizer: RegularizerType = Field(
        default=RegularizerType.AUTO,
        description="L1 regularization strategy for feature selection.",
    )
    confidence: float = Field(
        default=0.95,
        description=(
            "Coverage of bootstrap confidence intervals (e.g. 0.95 for 95%)."
            " Set to 1.0 to skip confidence interval computation."
        ),
    )
    track_counterfactuals: bool = Field(
        default=False,
        description="Whether to track counterfactual explanations alongside SHAP.",
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


class SHAPExplanationConfig(BaseModel):
    """SHAP explanation configuration."""

    model: ModelConfig
    explainer: SHAPExplainerConfig | None = None


class SHAPExplanationRequest(BaseModel):
    """SHAP explanation request."""

    predictionId: str
    config: SHAPExplanationConfig


# ---------------------------------------------------------------------------
# SHAP response models
# ---------------------------------------------------------------------------


class SHAPFeatureAttribution(BaseModel):
    """SHAP attribution for a single input feature."""

    feature_name: str = Field(description="Name of the input feature.")
    shap_value: float = Field(
        description=(
            "Shapley value for this feature: its additive contribution to"
            " f(x) - E[f(X)] under the local accuracy axiom."
        )
    )
    confidence_lower: float | None = Field(
        description=(
            "Lower bound of the bootstrap confidence interval for the SHAP value,"
            " or null when confidence intervals were not requested."
        )
    )
    confidence_upper: float | None = Field(
        description=(
            "Upper bound of the bootstrap confidence interval for the SHAP value,"
            " or null when confidence intervals were not requested."
        )
    )


class SHAPExplanationResponse(BaseModel):
    """SHAP explanation response containing per-feature attributions."""

    prediction_id: str = Field(description="ID of the explained stored prediction.")
    model: str = Field(description="Name of the model that produced the prediction.")
    attributions: list[SHAPFeatureAttribution] = Field(
        description="Per-feature SHAP attributions, one entry per input feature."
    )
    link: LinkType = Field(description="Link function used during explanation.")
    regularizer: RegularizerType = Field(
        description="L1 regularization strategy used during explanation."
    )


# ---------------------------------------------------------------------------
# Internal mappings
# ---------------------------------------------------------------------------

_LINK_MAP: dict[LinkType, str] = {
    LinkType.IDENTITY: "identity",
    LinkType.LOGIT: "logit",
}

_REG_MAP: dict[RegularizerType, str] = {
    # AUTO previously mapped to l1_reg="auto" which shap 0.47 deprecated; "bic" is the equivalent.
    RegularizerType.AUTO: "bic",
    RegularizerType.AIC: "aic",
    RegularizerType.BIC: "bic",
    RegularizerType.TOP_N_FEATURES: "num_features(10)",
    RegularizerType.NONE: "num_features(0)",
}


# ---------------------------------------------------------------------------
# Stored prediction retrieval (Epic 2)
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
# SHAP endpoint handler
# ---------------------------------------------------------------------------


@router.post(routes.EXPLAINER_LOCAL_SHAP)
async def local_shap_explanation(
    request: SHAPExplanationRequest,
) -> SHAPExplanationResponse:
    """Compute a KernelSHAP explanation for a stored prediction.

    KernelSHAP satisfies the following Shapley axioms:

    - **Local accuracy**: SHAP values sum exactly to ``f(x) - E[f(X)]``.
    - **Missingness**: features with no influence receive value ≈ 0
      (achieved approximately through L1 regularisation).
    - **Consistency**: a feature whose contribution increases over all
      coalitions always receives a higher attribution.

    Attributions explain a ``RandomForest`` surrogate trained on all stored
    organic observations, **not** the original model directly.  Confidence
    intervals are estimated via bootstrap resampling of the background data.
    Set ``config.explainer.confidence = 1.0`` to skip CI computation.

    """
    if not _SHAP_AVAILABLE:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail=(
                "explainability extra not installed."
                " Install with: pip install trustyai-service[explainability]"
            ),
        )

    logger.info("Computing SHAP explanation for prediction: %s", request.predictionId)

    config = request.config.explainer or SHAPExplainerConfig()
    model = request.config.model.name

    instance, all_input_names = await get_stored_prediction(model, request.predictionId)
    x_train, y_train, training_cols = await _prepare_training_data(
        model, config.n_samples
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

    link_str = _LINK_MAP[config.link]
    l1_reg_str = _REG_MAP[config.regularizer]

    surrogate = build_surrogate(x_train, y_train)
    shap_vals = compute_shap_values(
        instance_f64,
        x_train,
        surrogate,
        n_samples=config.n_samples,
        link=link_str,
        l1_reg=l1_reg_str,
    )
    lower, upper = compute_confidence_intervals(
        instance_f64,
        x_train,
        surrogate,
        confidence=config.confidence,
        n_samples=config.n_samples,
        link=link_str,
        l1_reg=l1_reg_str,
    )

    attributions = [
        SHAPFeatureAttribution(
            feature_name=training_cols[i],
            shap_value=float(shap_vals[i]),
            confidence_lower=float(lower[i]) if lower is not None else None,
            confidence_upper=float(upper[i]) if upper is not None else None,
        )
        for i in range(len(shap_vals))
    ]

    return SHAPExplanationResponse(
        prediction_id=request.predictionId,
        model=model,
        attributions=attributions,
        link=config.link,
        regularizer=config.regularizer,
    )


# ---------------------------------------------------------------------------
# Counterfactual
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Time-series saliency
# ---------------------------------------------------------------------------


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
