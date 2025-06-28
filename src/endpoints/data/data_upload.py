import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.service.constants import INPUT_SUFFIX, METADATA_SUFFIX, OUTPUT_SUFFIX
from src.service.data.modelmesh_parser import ModelMeshPayloadParser
from src.service.data.storage import get_storage_interface
from src.service.utils.upload import process_upload_request

router = APIRouter()
logger = logging.getLogger(__name__)


class UploadPayload(BaseModel):
    model_name: str
    data_tag: Optional[str] = None
    is_ground_truth: bool = False
    request: Dict[str, Any]
    response:  Optional[Dict[str, Any]] = None


@router.post("/data/upload")
async def upload(payload: UploadPayload) -> Dict[str, str]:
    """Upload model data - regular or ground truth."""
    try:
        logger.info(f"Received upload request for model: {payload.model_name}")
        result = await process_upload_request(payload)
        logger.info(f"Upload completed for model: {payload.model_name}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint for model {payload.model_name}: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")