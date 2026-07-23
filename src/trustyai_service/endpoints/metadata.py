"""Metadata endpoint for managing model metadata and schema information."""

import logging
from http import HTTPStatus
from typing import Never

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from trustyai_service.service.constants import INPUT_SUFFIX, OUTPUT_SUFFIX
from trustyai_service.service.data.datasources.data_source import DataSource
from trustyai_service.service.data.shared_data_source import get_shared_data_source
from trustyai_service.service.data.storage import get_storage_interface
from trustyai_service.service.payloads.service.schema import Schema
from trustyai_service.service.prometheus.prometheus_scheduler import PrometheusScheduler
from trustyai_service.service.prometheus.shared_prometheus_scheduler import (
    get_shared_prometheus_scheduler,
)

router = APIRouter()
logger = logging.getLogger(__name__)

storage_interface = get_storage_interface()


def _build_readable_schema(
    schema: Schema,
    original_names: list[str],
    aliased_names: list[str],
) -> dict:
    """Convert a Schema to the Java-compatible ReadableSchema format."""
    items = {}
    for name, item in schema.items.items():
        items[name] = {
            "type": item.type.value if hasattr(item.type, "value") else str(item.type),
            "index": item.column_index,
        }
    name_mapping = {
        orig: alias
        for orig, alias in zip(original_names, aliased_names, strict=False)
        if orig != alias
    }
    return {
        "items": items,
        "nameMapping": name_mapping,
    }


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


def get_prometheus_scheduler() -> PrometheusScheduler:
    """Get the shared prometheus scheduler instance."""
    return get_shared_prometheus_scheduler()


class NameMapping(BaseModel):
    """Mapping of column names for model inputs and outputs."""

    modelId: str
    inputMapping: dict[str, str] = {}
    outputMapping: dict[str, str] = {}


class DataTagging(BaseModel):
    """Per-row tags for labeling model dataset observations."""

    modelId: str
    dataTagging: dict[str, list[list[int]]] = {}


class ModelIdRequest(BaseModel):
    """Request payload containing a model identifier."""

    modelId: str


@router.get("/info")
async def get_service_info() -> dict[str, dict]:
    """Get a comprehensive overview of the model inference datasets collected by TrustyAI.

    Returns metadata about the metric computations that are scheduled over those datasets.
    """
    try:
        logger.info("Retrieving service info")

        # Get all known models from shared data source
        data_source = get_data_source()
        known_models = await data_source.get_known_models()
        logger.info("DataSource instance id: %s", id(data_source))
        logger.info("Found %s known models: %s", len(known_models), list(known_models))

        service_metadata = {}

        for model_id in known_models:
            try:
                # Get metadata for each model
                model_metadata = await data_source.get_metadata(model_id)
                num_observations = await data_source.get_num_observations(model_id)
                has_inferences = await data_source.has_recorded_inferences(model_id)

                # Get scheduled metrics for this model
                scheduled_metadata = {}
                try:
                    scheduler = get_prometheus_scheduler()
                    if scheduler:
                        # Get all metric types and count scheduled requests per model
                        all_requests = scheduler.get_all_requests()  # Should return dict of metric_name -> {request_id -> request}
                        for metric_name, requests_dict in all_requests.items():
                            count = 0
                            for request in requests_dict.values():
                                # Check if request is for this model (defensive access)
                                request_model_id = getattr(
                                    request,
                                    "model_id",
                                    getattr(request, "modelId", None),
                                )
                                if request_model_id == model_id:
                                    count += 1
                            if count > 0:
                                scheduled_metadata[metric_name] = count
                        logger.debug(
                            "Found %s scheduled metric types for model %s",
                            len(scheduled_metadata),
                            model_id,
                        )
                except Exception as e:  # Intentional: scheduler errors should not break metadata retrieval
                    logger.warning(
                        "Error retrieving scheduled metrics for model %s: %s",
                        model_id,
                        e,
                    )

                # Fetch column names for name mapping
                input_dataset = model_id + INPUT_SUFFIX
                output_dataset = model_id + OUTPUT_SUFFIX
                input_original = await storage_interface.get_original_column_names(
                    input_dataset
                )
                input_aliased = await storage_interface.get_aliased_column_names(
                    input_dataset
                )
                output_original = await storage_interface.get_original_column_names(
                    output_dataset
                )
                output_aliased = await storage_interface.get_aliased_column_names(
                    output_dataset
                )

                # Transform to match expected format
                service_metadata[model_id] = {
                    "data": {
                        "observations": num_observations,
                        "hasRecordedInferences": has_inferences,
                        "inputTensorName": model_metadata.input_tensor_name
                        if model_metadata
                        else "input",
                        "outputTensorName": model_metadata.output_tensor_name
                        if model_metadata
                        else "output",
                        "inputSchema": _build_readable_schema(
                            model_metadata.input_schema,
                            input_original,
                            input_aliased,
                        )
                        if model_metadata
                        else {"items": {}, "nameMapping": {}},
                        "outputSchema": _build_readable_schema(
                            model_metadata.output_schema,
                            output_original,
                            output_aliased,
                        )
                        if model_metadata
                        else {"items": {}, "nameMapping": {}},
                    },
                    "metrics": {"scheduledMetadata": scheduled_metadata},
                }

                logger.debug(
                    "Retrieved metadata for model %s: observations=%s, hasInferences=%s",
                    model_id,
                    num_observations,
                    has_inferences,
                )

            except Exception as e:  # Intentional: per-model errors should not break entire info endpoint
                logger.warning(
                    "Error retrieving metadata for model %s: %s", model_id, e
                )
                # Still include the model in the response but with basic info
                service_metadata[model_id] = {
                    "data": {
                        "observations": 0,
                        "hasRecordedInferences": False,
                        "inputTensorName": "input",
                        "outputTensorName": "output",
                        "inputSchema": {"items": {}, "nameMapping": {}},
                        "outputSchema": {"items": {}, "nameMapping": {}},
                    },
                    "metrics": {"scheduledMetadata": {}},
                    "error": str(e),
                }

    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown retrieval errors
        logger.exception("Error retrieving service info")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving service info: {e!s}",
        ) from e
    else:
        logger.info(
            "Successfully retrieved service info for %s models", len(service_metadata)
        )
        return service_metadata


@router.get("/info/inference/ids/{model}", response_model=None)
async def get_inference_ids(model: str, inference_type: str = "all") -> Never:
    """Get a list of all inference ids within a particular model inference."""
    logger.info(
        "Retrieving inference IDs for model: %s, type: %s", model, inference_type
    )
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Inference ID retrieval is not yet implemented",
    )


@router.get("/info/names")
async def get_column_names() -> dict[str, dict]:
    """Get the current name mappings for all models."""
    try:
        logger.info("Retrieving name mappings for all models")

        # Get all known models from shared data source
        data_source = get_data_source()
        known_models = await data_source.get_known_models()
        logger.info("Found %s known models: %s", len(known_models), list(known_models))

        name_mappings = {}

        for model_id in known_models:
            try:
                input_dataset_name = model_id + INPUT_SUFFIX
                output_dataset_name = model_id + OUTPUT_SUFFIX

                input_exists = await storage_interface.dataset_exists(
                    input_dataset_name
                )
                output_exists = await storage_interface.dataset_exists(
                    output_dataset_name
                )

                model_mappings = {
                    "modelId": model_id,
                    "inputMapping": {},
                    "outputMapping": {},
                }

                # Get input name mappings
                if input_exists:
                    try:
                        original_input_names = (
                            await storage_interface.get_original_column_names(
                                input_dataset_name
                            )
                        )
                        aliased_input_names = (
                            await storage_interface.get_aliased_column_names(
                                input_dataset_name
                            )
                        )

                        if (
                            original_input_names is not None
                            and aliased_input_names is not None
                        ):
                            # Create mapping from original to aliased names
                            input_mapping = {
                                orig: alias
                                for orig, alias in zip(
                                    list(original_input_names),
                                    list(aliased_input_names),
                                    strict=False,
                                )
                                if orig
                                != alias  # Only include if there's an actual mapping
                            }
                            model_mappings["inputMapping"] = input_mapping

                    except Exception as e:  # Intentional: input mapping errors should not break entire mapping retrieval
                        logger.warning(
                            "Error getting input name mappings for %s: %s", model_id, e
                        )

                # Get output name mappings
                if output_exists:
                    try:
                        original_output_names = (
                            await storage_interface.get_original_column_names(
                                output_dataset_name
                            )
                        )
                        aliased_output_names = (
                            await storage_interface.get_aliased_column_names(
                                output_dataset_name
                            )
                        )

                        if (
                            original_output_names is not None
                            and aliased_output_names is not None
                        ):
                            # Create mapping from original to aliased names
                            output_mapping = {
                                orig: alias
                                for orig, alias in zip(
                                    list(original_output_names),
                                    list(aliased_output_names),
                                    strict=False,
                                )
                                if orig
                                != alias  # Only include if there's an actual mapping
                            }
                            model_mappings["outputMapping"] = output_mapping

                    except Exception as e:  # Intentional: output mapping errors should not break entire mapping retrieval
                        logger.warning(
                            "Error getting output name mappings for %s: %s", model_id, e
                        )

                name_mappings[model_id] = model_mappings

            except Exception as e:  # Intentional: per-model errors should not break entire mapping endpoint
                logger.warning(
                    "Error getting name mappings for model %s: %s", model_id, e
                )

    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown mapping errors
        logger.exception("Error retrieving name mappings")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving name mappings: {e!s}",
        ) from e
    else:
        logger.info(
            "Successfully retrieved name mappings for %s models", len(name_mappings)
        )
        return name_mappings


@router.post("/info/names")
async def apply_column_names(name_mapping: NameMapping) -> dict[str, str]:
    """Apply a set of human-readable column names to a particular inference."""
    logger.info("Applying column names for model: %s", name_mapping.modelId)

    model_id = name_mapping.modelId
    input_dataset_name = model_id + INPUT_SUFFIX
    output_dataset_name = model_id + OUTPUT_SUFFIX

    # Check if the model datasets exist
    input_exists = await storage_interface.dataset_exists(input_dataset_name)
    output_exists = await storage_interface.dataset_exists(output_dataset_name)

    # Validate datasets exist before try block
    if not input_exists and not output_exists:
        error_msg = (
            f"No metadata found for model={model_id}. "
            "This can happen if TrustyAI has not yet logged any inferences from this model."
        )
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=error_msg)

    try:
        # Apply input mappings if provided and dataset exists
        if name_mapping.inputMapping and input_exists:
            logger.info(
                "Applying input mappings for model %s: %s",
                model_id,
                name_mapping.inputMapping,
            )
            await storage_interface.apply_name_mapping(
                input_dataset_name, name_mapping.inputMapping
            )

        # Apply output mappings if provided and dataset exists
        if name_mapping.outputMapping and output_exists:
            logger.info(
                "Applying output mappings for model %s: %s",
                model_id,
                name_mapping.outputMapping,
            )
            await storage_interface.apply_name_mapping(
                output_dataset_name, name_mapping.outputMapping
            )

    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except Exception as e:  # Broad catch intentional: endpoint catch-all for unknown application errors
        logger.exception("Error applying column names")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error applying column names: {e!s}",
        ) from e
    else:
        logger.info("Name mappings successfully applied to model=%s", model_id)
        return {"message": "Feature and output name mapping successfully applied."}


@router.delete("/info/names")
async def remove_column_names(request: ModelIdRequest) -> dict[str, str]:
    """Remove any column names that have been applied to a particular inference."""
    model_id = request.modelId
    logger.info("Removing column names for model: %s", model_id)

    input_dataset_name = model_id + INPUT_SUFFIX
    output_dataset_name = model_id + OUTPUT_SUFFIX

    # Check if the model datasets exist
    input_exists = await storage_interface.dataset_exists(input_dataset_name)
    output_exists = await storage_interface.dataset_exists(output_dataset_name)

    # Validate datasets exist before try block
    if not input_exists and not output_exists:
        error_msg = (
            f"No metadata found for model={model_id}. "
            "This can happen if TrustyAI has not yet logged any inferences from this model."
        )
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=error_msg)

    try:
        # Clear name mappings from input dataset if it exists
        if input_exists:
            logger.info("Clearing input name mappings for model %s", model_id)
            await storage_interface.clear_name_mapping(input_dataset_name)

        # Clear name mappings from output dataset if it exists
        if output_exists:
            logger.info("Clearing output name mappings for model %s", model_id)
            await storage_interface.clear_name_mapping(output_dataset_name)

    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown removal errors
        logger.exception("Error removing column names")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error removing column names: {e!s}",
        ) from e
    else:
        logger.info("Name mappings successfully cleared from model=%s", model_id)
        return {"message": "Feature and output name mapping successfully cleared."}


@router.get("/info/tags", response_model=None)
async def get_tags() -> Never:
    """Retrieve the tags that have been applied to a particular model dataset, as well as a count of that tag's frequency within the dataset."""
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Tag retrieval is not yet implemented",
    )


@router.post("/info/tags", response_model=None)
async def apply_tags(data_tagging: DataTagging) -> Never:
    """Apply per-row tags to a particular inference model dataset, to label certain rows as training or drift reference data, etc."""
    logger.info("Applying tags for model: %s", data_tagging.modelId)
    raise HTTPException(
        status_code=HTTPStatus.NOT_IMPLEMENTED,
        detail="Tag application is not yet implemented",
    )
