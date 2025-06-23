import logging
from typing import Dict, Optional

import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.endpoints.consumer.consumer_endpoint import consume_cloud_event
from src.endpoints.consumer import KServeInferenceRequest, KServeInferenceResponse
from src.service.constants import TRUSTYAI_TAG_PREFIX
from src.service.data.model_data import ModelData


router = APIRouter()
logger = logging.getLogger(__name__)


class UploadPayload(BaseModel):
    model_name: str
    data_tag: Optional[str] = None
    is_ground_truth: bool = False
    request: KServeInferenceRequest
    response:  KServeInferenceResponse


def validate_data_tag(tag: str) -> Optional[str]:
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
async def upload(payload: UploadPayload) -> Dict[str, str]:
    """Upload model data"""

    # validate tag
    tag_validation_msg = validate_data_tag(payload.data_tag)
    if tag_validation_msg:
        raise HTTPException(status_code=400, detail=tag_validation_msg)
    try:
        logger.info(f"Received upload request for model: {payload.model_name}")

        # overwrite response model name with provided model name
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

        model_data =  ModelData(payload.model_name)
        new_data_points = (await model_data.row_counts())[0]

        logger.info(f"Upload completed for model: {payload.model_name}")

        return {
            "status": "success",
            "message": f"{new_data_points-previous_data_points} datapoints successfully added to {payload.model_name} data."
        }

    except HTTPException as e:
        if "Could not reconcile_kserve KServe Inference" in str(e):
            raise HTTPException(status_code=400, detail=f"Could not upload payload for model {payload.model_name}: {str(e)}") from e
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint for model {payload.model_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
