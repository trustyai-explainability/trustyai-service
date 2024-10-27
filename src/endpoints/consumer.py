# endpoints/consumer.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class PartialKind(str):
    REQUEST = "request"
    RESPONSE = "response"


class PartialPayloadId(BaseModel):
    predictionId: str
    kind: PartialKind


class InferencePartialPayload(BaseModel):
    partialPayloadId: Optional[PartialPayloadId] = None
    metadata: Optional[Dict[str, str]] = None
    id: Optional[str] = None
    kind: Optional[PartialKind] = None
    data: Optional[str] = None
    modelid: Optional[str] = None


@router.post("/consumer/kserve/v2")
async def consume_inference_payload(payload: InferencePartialPayload):
    """Send a single input or output payload to TrustyAI."""
    try:
        logger.info(f"Received inference payload for model: {payload.modelid}")
        # TODO: Implement
        return {"status": "success", "message": "Payload processed successfully"}
    except Exception as e:
        logger.error(f"Error processing inference payload: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing payload: {str(e)}"
        )
