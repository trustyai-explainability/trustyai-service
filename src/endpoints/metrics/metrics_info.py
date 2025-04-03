from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics/all/requests")
async def get_all_metric_requests(type: Optional[str] = None):
    """Retrieve a list of all currently scheduled metric computations."""
    try:
        logger.info(f"Retrieving all metric requests, type filter: {type}")
        # TODO: Implement
        return {"requests": []}
    except Exception as e:
        logger.error(f"Error retrieving metric requests: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving metric requests: {str(e)}"
        )
