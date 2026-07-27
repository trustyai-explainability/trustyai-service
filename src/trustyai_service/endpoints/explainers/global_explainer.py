"""Global explainer endpoint for model-wide explanation requests."""

import logging
from http import HTTPStatus
from typing import Never

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for model identification and targeting."""

    target: str
    name: str
    version: str | None = None


class GlobalExplanationRequest(BaseModel):
    """Request payload for global model explanation."""

    modelConfig: ModelConfig


@router.post("/explainers/global/lime")
async def global_lime_explanation(request: GlobalExplanationRequest) -> None:
    """Compute a global LIME explanation."""
    logger.info(
        "Computing global LIME explanation for model: %s", request.modelConfig.name
    )
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Global LIME explanation is not yet implemented",
    )


@router.post("/explainers/global/pdp", response_model=None)
async def global_pdp_explanation(request: GlobalExplanationRequest) -> Never:
    """Compute a global PDP explanation."""
    logger.info(
        "Computing global PDP explanation for model: %s", request.modelConfig.name
    )
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Global PDP explanation is not yet implemented",
    )
