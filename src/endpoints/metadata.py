from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging

from src.service.data.storage import get_storage_interface
from src.service.data.shared_data_source import get_shared_data_source
from src.service.constants import INPUT_SUFFIX, OUTPUT_SUFFIX

router = APIRouter()
logger = logging.getLogger(__name__)

storage_interface = get_storage_interface()

def get_data_source():
    """Get the shared data source instance."""
    return get_shared_data_source()


class NameMapping(BaseModel):
    modelId: str
    inputMapping: Dict[str, str] = {}
    outputMapping: Dict[str, str] = {}


class DataTagging(BaseModel):
    modelId: str
    dataTagging: Dict[str, List[List[int]]] = {}


class ModelIdRequest(BaseModel):
    modelId: str


@router.get("/info")
async def get_service_info():
    """Get a comprehensive overview of the model inference datasets collected by TrustyAI and the metric computations that are scheduled over those datasets."""
    try:
        logger.info("Retrieving service info")

        # Get all known models from shared data source
        data_source = get_data_source()
        known_models = await data_source.get_known_models()
        logger.info(f"DataSource instance id: {id(data_source)}")
        logger.info(f"Found {len(known_models)} known models: {list(known_models)}")

        service_metadata = {}

        for model_id in known_models:
            try:
                # Get metadata for each model
                model_metadata = await data_source.get_metadata(model_id)
                num_observations = await data_source.get_num_observations(model_id)
                has_inferences = await data_source.has_recorded_inferences(model_id)

                # Transform to match expected format
                service_metadata[model_id] = {
                    "data": {
                        "observations": num_observations,
                        "hasRecordedInferences": has_inferences,
                        "inputTensorName": model_metadata.input_tensor_name if model_metadata else "input",
                        "outputTensorName": model_metadata.output_tensor_name if model_metadata else "output"
                    },
                    "metrics": {
                        "scheduledMetadata": {}  # TODO: Integrate with prometheus scheduler
                    }
                }

                logger.debug(f"Retrieved metadata for model {model_id}: observations={num_observations}, hasInferences={has_inferences}")

            except Exception as e:
                logger.warning(f"Error retrieving metadata for model {model_id}: {e}")
                # Still include the model in the response but with basic info
                service_metadata[model_id] = {
                    "data": {
                        "observations": 0,
                        "hasRecordedInferences": False,
                        "inputTensorName": "input",
                        "outputTensorName": "output"
                    },
                    "metrics": {"scheduledMetadata": {}},
                    "error": str(e)
                }

        logger.info(f"Successfully retrieved service info for {len(service_metadata)} models")
        return service_metadata

    except Exception as e:
        logger.error(f"Error retrieving service info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving service info: {str(e)}"
        ) from e


@router.get("/info/inference/ids/{model}")
async def get_inference_ids(model: str, type: str = "all"):
    """Get a list of all inference ids within a particular model inference."""
    try:
        logger.info(f"Retrieving inference IDs for model: {model}, type: {type}")
        # TODO: Implement
        return {"inferenceIds": []}
    except Exception as e:
        logger.error(f"Error retrieving inference IDs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving inference IDs: {str(e)}")


@router.get("/info/names")
async def get_column_names():
    """Get the current name mappings for all models."""
    try:
        logger.info("Retrieving name mappings for all models")

        # Get all known models from shared data source
        data_source = get_data_source()
        known_models = await data_source.get_known_models()
        logger.info(f"Found {len(known_models)} known models: {list(known_models)}")

        name_mappings = {}

        for model_id in known_models:
            try:
                input_dataset_name = model_id + INPUT_SUFFIX
                output_dataset_name = model_id + OUTPUT_SUFFIX

                input_exists = await storage_interface.dataset_exists(input_dataset_name)
                output_exists = await storage_interface.dataset_exists(output_dataset_name)

                model_mappings = {
                    "modelId": model_id,
                    "inputMapping": {},
                    "outputMapping": {}
                }

                # Get input name mappings
                if input_exists:
                    try:
                        original_input_names = await storage_interface.get_original_column_names(input_dataset_name)
                        aliased_input_names = await storage_interface.get_aliased_column_names(input_dataset_name)

                        if original_input_names is not None and aliased_input_names is not None:
                            # Create mapping from original to aliased names
                            input_mapping = {}
                            for orig, alias in zip(list(original_input_names), list(aliased_input_names)):
                                if orig != alias:  # Only include if there's an actual mapping
                                    input_mapping[orig] = alias
                            model_mappings["inputMapping"] = input_mapping

                    except Exception as e:
                        logger.warning(f"Error getting input name mappings for {model_id}: {e}")

                # Get output name mappings
                if output_exists:
                    try:
                        original_output_names = await storage_interface.get_original_column_names(output_dataset_name)
                        aliased_output_names = await storage_interface.get_aliased_column_names(output_dataset_name)

                        if original_output_names is not None and aliased_output_names is not None:
                            # Create mapping from original to aliased names
                            output_mapping = {}
                            for orig, alias in zip(list(original_output_names), list(aliased_output_names)):
                                if orig != alias:  # Only include if there's an actual mapping
                                    output_mapping[orig] = alias
                            model_mappings["outputMapping"] = output_mapping

                    except Exception as e:
                        logger.warning(f"Error getting output name mappings for {model_id}: {e}")

                name_mappings[model_id] = model_mappings

            except Exception as e:
                logger.warning(f"Error getting name mappings for model {model_id}: {e}")

        logger.info(f"Successfully retrieved name mappings for {len(name_mappings)} models")
        return name_mappings

    except Exception as e:
        logger.error(f"Error retrieving name mappings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving name mappings: {str(e)}")


@router.post("/info/names")
async def apply_column_names(name_mapping: NameMapping):
    """Apply a set of human-readable column names to a particular inference."""
    try:
        logger.info(f"Applying column names for model: {name_mapping.modelId}")

        model_id = name_mapping.modelId
        input_dataset_name = model_id + INPUT_SUFFIX
        output_dataset_name = model_id + OUTPUT_SUFFIX

        # Check if the model datasets exist
        input_exists = await storage_interface.dataset_exists(input_dataset_name)
        output_exists = await storage_interface.dataset_exists(output_dataset_name)

        if not input_exists and not output_exists:
            error_msg = f"No metadata found for model={model_id}. This can happen if TrustyAI has not yet logged any inferences from this model."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Apply input mappings if provided and dataset exists
        if name_mapping.inputMapping and input_exists:
            logger.info(f"Applying input mappings for model {model_id}: {name_mapping.inputMapping}")
            await storage_interface.apply_name_mapping(input_dataset_name, name_mapping.inputMapping)

        # Apply output mappings if provided and dataset exists
        if name_mapping.outputMapping and output_exists:
            logger.info(f"Applying output mappings for model {model_id}: {name_mapping.outputMapping}")
            await storage_interface.apply_name_mapping(output_dataset_name, name_mapping.outputMapping)

        logger.info(f"Name mappings successfully applied to model={model_id}")
        return {"message": "Feature and output name mapping successfully applied."}

    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except Exception as e:
        logger.error(f"Error applying column names: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error applying column names: {str(e)}"
        ) from e


@router.delete("/info/names")
async def remove_column_names(request: ModelIdRequest):
    """Remove any column names that have been applied to a particular inference."""
    try:
        model_id = request.modelId
        logger.info(f"Removing column names for model: {model_id}")

        input_dataset_name = model_id + INPUT_SUFFIX
        output_dataset_name = model_id + OUTPUT_SUFFIX

        # Check if the model datasets exist
        input_exists = await storage_interface.dataset_exists(input_dataset_name)
        output_exists = await storage_interface.dataset_exists(output_dataset_name)

        if not input_exists and not output_exists:
            error_msg = f"No metadata found for model={model_id}. This can happen if TrustyAI has not yet logged any inferences from this model."
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Clear name mappings from input dataset if it exists
        if input_exists:
            logger.info(f"Clearing input name mappings for model {model_id}")
            await storage_interface.clear_name_mapping(input_dataset_name)

        # Clear name mappings from output dataset if it exists
        if output_exists:
            logger.info(f"Clearing output name mappings for model {model_id}")
            await storage_interface.clear_name_mapping(output_dataset_name)

        logger.info(f"Name mappings successfully cleared from model={model_id}")
        return {"message": "Feature and output name mapping successfully cleared."}

    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except Exception as e:
        logger.error(f"Error removing column names: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error removing column names: {str(e)}"
        ) from e


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
