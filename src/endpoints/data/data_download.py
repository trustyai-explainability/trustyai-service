import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.service.utils.download import (
    DataRequestPayload,
    DataResponsePayload,
    apply_filters,  # ← New utility function
    load_model_dataframe,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/data/download")
async def download_data(payload: DataRequestPayload) -> DataResponsePayload:
    """Download model data with filtering."""
    try:
        logger.info(f"Received data download request for model: {payload.modelId}")
        df = await load_model_dataframe(payload.modelId)
        if df.empty:
            return DataResponsePayload(dataCSV="")
        df = apply_filters(df, payload)
        csv_data = df.to_csv(index=False)
        return DataResponsePayload(dataCSV=csv_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading data: {str(e)}")