"""Local explainer endpoint for instance-level explanation requests."""

import logging
from enum import StrEnum
from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration for explainer requests."""

    target: str
    name: str
    version: str | None = None


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


@router.post("/explainers/local/lime")
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


@router.post("/explainers/local/shap")
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


@router.post("/explainers/local/cf")
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


@router.post("/explainers/local/tssaliency")
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
