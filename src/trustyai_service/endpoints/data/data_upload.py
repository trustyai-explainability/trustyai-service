"""Data upload endpoint for submitting inference data to the TrustyAI service."""

import logging
import uuid
from http import HTTPStatus

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.endpoints.consumer.consumer_endpoint import consume_cloud_event
from trustyai_service.exceptions import ReconciliationError
from trustyai_service.service.constants import TRUSTYAI_TAG_PREFIX
from trustyai_service.service.data.model_data import ModelData

router = APIRouter()
logger = logging.getLogger(__name__)


class UploadPayload(BaseModel):
    """Payload model for uploading inference request/response pairs."""

    model_name: str
    data_tag: str | None = None
    is_ground_truth: bool = (
        False  # Reserved for future ground truth storage implementation
    )
    request: KServeInferenceRequest
    response: KServeInferenceResponse


def validate_data_tag(tag: str | None) -> str | None:
    """Validate data tag format and content."""
    if not tag:
        return None
    if tag.startswith(TRUSTYAI_TAG_PREFIX):
        return (
            f"The tag prefix '{TRUSTYAI_TAG_PREFIX}' is reserved for internal TrustyAI use only. "
            f"Provided tag '{tag}' violates this restriction."
        )
    return None


@router.post("/data/upload")
async def upload(payload: UploadPayload) -> dict[str, str]:
    """Upload model data."""
    # validate tag
    tag_validation_msg = validate_data_tag(payload.data_tag)
    if tag_validation_msg:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail=tag_validation_msg
        )

    # Validate ground truth parameter
    if payload.is_ground_truth:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            detail="Ground truth upload is not yet implemented. "
            "This parameter is reserved for future use.",
        )

    try:
        logger.info("Received upload request for model: %s", payload.model_name)

        # overwrite response model name with provided model name
        if payload.response.model_name != payload.model_name:
            logger.warning(
                "Response model name '%s' differs from request model name '%s'. Using '%s'.",
                payload.response.model_name,
                payload.model_name,
                payload.model_name,
            )
            payload.response.model_name = payload.model_name

        req_id = str(uuid.uuid4())
        model_data = ModelData(payload.model_name)
        datasets_exist = await model_data.datasets_exist()

        if all(datasets_exist):
            previous_data_points = (await model_data.row_counts())[0]
        else:
            previous_data_points = 0

        await consume_cloud_event(payload.response, req_id)
        await consume_cloud_event(payload.request, req_id, tag=payload.data_tag)

        model_data = ModelData(payload.model_name)
        new_data_points = (await model_data.row_counts())[0]

    except ReconciliationError as e:
        logger.exception("Reconciliation error for model %s", payload.model_name)
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Could not upload payload for model {payload.model_name}: {e.message}",
        ) from e
    except HTTPException:
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown upload errors
        logger.exception(
            "Unexpected error in upload endpoint for model %s", payload.model_name
        )
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred during data upload. Check server logs for details.",
        ) from e
    else:
        logger.info("Upload completed for model: %s", payload.model_name)

        return {
            "status": "success",
            "message": (
                f"{new_data_points - previous_data_points} datapoints successfully "
                f"added to {payload.model_name} data."
            ),
        }
