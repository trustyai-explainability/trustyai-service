from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class NameMapping(BaseModel):
    modelId: str
    inputMapping: Dict[str, str] = {}
    outputMapping: Dict[str, str] = {}


class DataTagging(BaseModel):
    modelId: str
    dataTagging: Dict[str, List[List[int]]] = {}


@router.get("/info")
async def get_service_info():
    """Get a list of all inference ids within a particular model inference."""
    try:
        # TODO: Implement
        return {"models": [], "metrics": [], "version": "1.0.0rc0"}
    except Exception as e:
        logger.error(f"Error retrieving service info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving service info: {str(e)}"
        )


@router.get("/info/inference/ids/{model}")
async def get_inference_ids(model: str, type: str = "all"):
    """Get a list of all inference ids within a particular model inference."""
    try:
        logger.info(f"Retrieving inference IDs for model: {model}, type: {type}")
        # TODO: Implement
        return {"inferenceIds": []}
    except Exception as e:
        logger.error(f"Error retrieving inference IDs: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving inference IDs: {str(e)}"
        )


@router.post("/info/names")
async def apply_column_names(name_mapping: NameMapping):
    """Apply a set of human-readable column names to a particular inference."""
    try:
        logger.info(f"Applying column names for model: {name_mapping.modelId}")
        # TODO: Implement
        return {"status": "success", "message": "Column names applied successfully"}
    except Exception as e:
        logger.error(f"Error applying column names: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error applying column names: {str(e)}"
        )


@router.delete("/info/names")
async def remove_column_names(model_id: str):
    """Remove any column names that have been applied to a particular inference."""
    try:
        logger.info(f"Removing column names for model: {model_id}")
        # TODO: Implement
        return {"status": "success", "message": "Column names removed successfully"}
    except Exception as e:
        logger.error(f"Error removing column names: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error removing column names: {str(e)}"
        )


@router.get("/info/tags")
async def get_tags():
    """Retrieve the tags that have been applied to a particular model dataset, as well as a count of that tag's frequency within the dataset."""
    try:
        # TODO: Implement
        return {"tags": {}}
    except Exception as e:
        logger.error(f"Error retrieving tags: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving tags: {str(e)}")


@router.post("/info/tags")
async def apply_tags(data_tagging: DataTagging):
    """Apply per-row tags to a particular inference model dataset, to label certain rows as training or drift reference data, etc."""
    try:
        logger.info(f"Applying tags for model: {data_tagging.modelId}")
        # TODO: Implement
        return {"status": "success", "message": "Tags applied successfully"}
    except Exception as e:
        logger.error(f"Error applying tags: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error applying tags: {str(e)}")
