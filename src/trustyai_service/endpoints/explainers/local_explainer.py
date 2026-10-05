"""Local explainer router and unfinished local route placeholders."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trustyai_service.endpoints import routes

_placeholder_router = APIRouter()


class ModelConfig(BaseModel):
    """Placeholder model contract for unfinished local explainers."""

    target: str
    name: str
    version: str | None = None


class LocalExplanationPlaceholderRequest(BaseModel):
    """Shared request shape for unfinished LIME and SHAP routes."""

    predictionId: str
    config: dict[str, Any]


@_placeholder_router.post(routes.EXPLAINER_LOCAL_LIME)
async def local_lime_explanation(
    request: LocalExplanationPlaceholderRequest,
) -> dict[str, Any]:
    """Return the existing not-implemented LIME response."""
    del request
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Local LIME explanation is not yet implemented",
    )


@_placeholder_router.post(routes.EXPLAINER_LOCAL_SHAP)
async def local_shap_explanation(
    request: LocalExplanationPlaceholderRequest,
) -> dict[str, Any]:
    """Return the existing not-implemented SHAP response."""
    del request
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Local SHAP explanation is not yet implemented",
    )


class CounterfactualExplainerConfig(BaseModel):
    """Configuration placeholder for the unfinished counterfactual route."""

    n_samples: int = 100


class CounterfactualExplanationConfig(BaseModel):
    """Request model placeholder for counterfactual explanations."""

    model: ModelConfig
    explainer: CounterfactualExplainerConfig | None = None


class CounterfactualExplanationRequest(BaseModel):
    """Request model placeholder for counterfactual explanations."""

    predictionId: str
    config: CounterfactualExplanationConfig
    goals: dict[str, str] | None = None
    explanationConfig: CounterfactualExplanationConfig | None = None


@_placeholder_router.post(routes.EXPLAINER_LOCAL_CF)
async def local_counterfactual_explanation(
    request: CounterfactualExplanationRequest,
) -> dict[str, Any]:
    """Return a stable not-implemented response for the placeholder route."""
    del request
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Local Counterfactual explanation is not yet implemented",
    )


class TSSaliencyExplainerConfig(BaseModel):
    """Configuration placeholder for the unfinished time-series route."""

    timeout: int = 10
    mu: float = 0.01
    n_samples: int = 50
    n_alpha: int = 50
    sigma: float = 50
    base_values: list[float] | None = None


class TSSaliencyExplanationConfig(BaseModel):
    """Request model placeholder for time-series saliency explanations."""

    model: ModelConfig
    explainer: TSSaliencyExplainerConfig | None = None


class TSSaliencyExplanationRequest(BaseModel):
    """Request model placeholder for time-series saliency explanations."""

    predictionIds: list[str]
    config: TSSaliencyExplanationConfig


@_placeholder_router.post(routes.EXPLAINER_LOCAL_TSSALIENCY)
async def local_tssaliency_explanation(
    request: TSSaliencyExplanationRequest,
) -> dict[str, Any]:
    """Return a stable not-implemented response for the placeholder route."""
    del request
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Local TSSaliency explanation is not yet implemented",
    )


def build_router() -> APIRouter:
    """Build a fresh local router containing only shared placeholders."""
    local_router = APIRouter()
    local_router.include_router(_placeholder_router)
    return local_router


router = build_router()

__all__ = ["build_router", "router"]
