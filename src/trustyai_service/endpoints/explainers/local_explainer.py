"""Shared local-explainer contract and unfinished local route placeholders."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_models import (
    LocalExplanationModelConfig,
)

router = APIRouter()

__all__ = ["LocalExplanationModelConfig", "router"]


class ModelConfig(BaseModel):
    """Legacy placeholder retained for unfinished local routes."""

    target: str
    name: str
    version: str | None = None


class CounterfactualExplainerConfig(BaseModel):
    n_samples: int = 100


class CounterfactualExplanationConfig(BaseModel):
    model: ModelConfig
    explainer: CounterfactualExplainerConfig | None = None


class CounterfactualExplanationRequest(BaseModel):
    predictionId: str
    config: CounterfactualExplanationConfig
    goals: dict[str, str] | None = None
    explanationConfig: CounterfactualExplanationConfig | None = None


@router.post(routes.EXPLAINER_LOCAL_CF)
async def local_counterfactual_explanation(
    request: CounterfactualExplanationRequest,
) -> dict[str, Any]:
    raise HTTPException(
        HTTPStatus.NOT_IMPLEMENTED,
        "Local Counterfactual explanation is not yet implemented",
    )


class TSSaliencyExplainerConfig(BaseModel):
    timeout: int = 10
    mu: float = 0.01
    n_samples: int = 50
    n_alpha: int = 50
    sigma: float = 50
    base_values: list[float] | None = None


class TSSaliencyExplanationConfig(BaseModel):
    model: ModelConfig
    explainer: TSSaliencyExplainerConfig | None = None


class TSSaliencyExplanationRequest(BaseModel):
    predictionIds: list[str]
    config: TSSaliencyExplanationConfig


@router.post(routes.EXPLAINER_LOCAL_TSSALIENCY)
async def local_tssaliency_explanation(
    request: TSSaliencyExplanationRequest,
) -> dict[str, Any]:
    raise HTTPException(
        HTTPStatus.NOT_IMPLEMENTED,
        "Local TSSaliency explanation is not yet implemented",
    )


# Keep algorithm orchestration out of this shared contract module.  Imports are
# intentionally at the end so the child routers can import the contract above
# without a circular initialization failure.
from trustyai_service.endpoints.explainers.local_lime import (  # noqa: E402
    LimeExplainerConfig,
    LimeExplanationConfig,
    LimeExplanationRequest,
    LIMEExplanationResponse,
    LIMEFeatureAttribution,
    local_lime_explanation,
)
from trustyai_service.endpoints.explainers.local_lime import (  # noqa: E402
    router as lime_router,
)
from trustyai_service.endpoints.explainers.local_shap import (  # noqa: E402
    LinkType,
    RegularizerType,
    SHAPExplainerConfig,
    SHAPExplanationConfig,
    SHAPExplanationRequest,
    local_shap_explanation,
)
from trustyai_service.endpoints.explainers.local_shap import (  # noqa: E402
    router as shap_router,
)

__all__ += [
    "LIMEExplanationResponse",
    "LIMEFeatureAttribution",
    "LimeExplainerConfig",
    "LimeExplanationConfig",
    "LimeExplanationRequest",
    "LinkType",
    "RegularizerType",
    "SHAPExplainerConfig",
    "SHAPExplanationConfig",
    "SHAPExplanationRequest",
    "local_lime_explanation",
    "local_shap_explanation",
]

router.include_router(lime_router)
router.include_router(shap_router)
