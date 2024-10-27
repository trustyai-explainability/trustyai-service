from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class ModelInferJointPayload(BaseModel):
    model_name: str
    data_tag: str = None
    is_ground_truth: bool = False
    request: Dict[str, Any]
    response: Dict[str, Any]


@router.post("/data/upload")
async def upload_data(payload: ModelInferJointPayload):
    """Upload a batch of model data to TrustyAI."""
    try:
        logger.info(f"Received data upload for model: {payload.model_name}")
        # TODO: Implement
        return {"status": "success", "message": "Data uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")
