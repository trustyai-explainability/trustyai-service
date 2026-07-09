"""Metadata endpoint for managing model metadata and schema information."""

import logging
from collections import Counter
from http import HTTPStatus
from typing import Annotated, Never

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.endpoints.data.data_upload import validate_data_tag
from src.service.constants import INPUT_SUFFIX, METADATA_SUFFIX, OUTPUT_SUFFIX
from src.service.data.datasources.data_source import DataSource
from src.service.data.model_data import ModelData
from src.service.data.shared_data_source import get_shared_data_source
from src.service.data.storage import get_storage_interface
from src.service.payloads.service.schema import Schema
from src.service.prometheus.prometheus_scheduler import PrometheusScheduler
from src.service.prometheus.shared_prometheus_scheduler import (
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


def _extract_tags(cell: object) -> list[str]:
    """Extract a list of tag strings from a metadata cell value."""
    if isinstance(cell, np.ndarray):
        return cell.tolist()
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        return [cell]
    return []


def _find_tags_column(metadata_names: list[str]) -> int:
    """Return the column index of the ``tags`` column, or -1 if absent."""
    return list(metadata_names).index("tags") if "tags" in metadata_names else -1


async def _read_metadata(
    model_id: str,
) -> tuple[np.ndarray | None, list[str]]:
    """Read the metadata array and column names for a model."""
    model_data = ModelData(model_id)
    try:
        _, _, metadata = await model_data.data(get_input=False, get_output=False)
    except Exception as exc:
        logger.exception("Error reading metadata for model=%s", model_id)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error reading metadata for model={model_id}",
        ) from exc
    _, _, metadata_names = await model_data.column_names()
    return metadata, list(metadata_names)


async def _ensure_model_exists(
    model_id: str,
    data_source: DataSource,
) -> None:
    """Raise 404 if the model is not known to the data source."""
    if not await data_source.has_metadata(model_id):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"No model found with id={model_id}",
        )


def _parse_range(r: list[int]) -> tuple[int, int]:
    """Parse a ``[start, end]`` or ``[index]`` range into ``(start, end)``."""
    expected_pair_len = 2
    if len(r) == 1:
        return r[0], r[0] + 1
    if len(r) == expected_pair_len:
        return r[0], r[1]
    msg = f"Each range must be [start, end] or [index], got {r}"
    raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=msg)


def _validate_range(start: int, end: int, total_rows: int) -> None:
    """Raise 400 if the range is invalid or out of bounds."""
    if start < 0 or end < 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Row indices must be non-negative, got [{start}, {end})",
        )
    if start >= end:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Range start must be less than end, got [{start}, {end})",
        )
    if end > total_rows:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=(f"Range [{start}, {end}) exceeds dataset size ({total_rows} rows)"),
        )


@router.get("/info/tags")
async def get_tags(
    model_id: Annotated[str | None, Query(alias="modelId")] = None,
) -> dict:
    """Retrieve per-tag row counts for a model's dataset.

    If ``modelId`` is provided, returns ``dict[str, int]`` for that model.
    If omitted, returns ``dict[str, dict[str, int]]`` for every known model.
    """
    data_source = get_data_source()

    if model_id is not None:
        return await _get_tag_counts_for_model(model_id, data_source)

    known_models = await data_source.get_verified_models()
    result: dict[str, dict[str, int]] = {}
    for mid in known_models:
        try:
            result[mid] = await _get_tag_counts_for_model(mid, data_source)
        except HTTPException as exc:
            if exc.status_code == HTTPStatus.NOT_FOUND:
                continue
            raise
    return result


async def _get_tag_counts_for_model(
    model_id: str,
    data_source: DataSource,
) -> dict[str, int]:
    """Return a Counter-style dict mapping tag name to row count."""
    await _ensure_model_exists(model_id, data_source)

    metadata, metadata_names = await _read_metadata(model_id)
    if metadata is None or len(metadata) == 0:
        return {}

    tags_col = _find_tags_column(metadata_names)
    if tags_col < 0:
        return {}

    counter: Counter[str] = Counter()
    for row in metadata:
        counter.update(_extract_tags(row[tags_col]))

    return dict(counter)


@router.post("/info/tags")
async def apply_tags(data_tagging: DataTagging) -> dict:
    """Apply per-row tags to a model dataset.

    Tags are appended idempotently (a tag already present on a row is not
    duplicated).  The ``_trustyai`` prefix is reserved for internal use.
    """
    model_id = data_tagging.modelId
    logger.info("Applying tags for model: %s", model_id)

    data_source = get_data_source()
    await _ensure_model_exists(model_id, data_source)

    if not data_tagging.dataTagging:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="dataTagging must contain at least one tag with ranges",
        )

    for tag_name in data_tagging.dataTagging:
        if not tag_name:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Tag names must be non-empty",
            )
        validation_msg = validate_data_tag(tag_name)
        if validation_msg is not None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=validation_msg,
            )

    metadata, metadata_names = await _read_metadata(model_id)
    if metadata is None or len(metadata) == 0:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Model {model_id} has no observation data to tag",
        )

    tags_col = _find_tags_column(metadata_names)
    if tags_col < 0:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Metadata dataset is missing the 'tags' column",
        )

    applied = _apply_tags_to_metadata(data_tagging.dataTagging, metadata, tags_col)

    await _persist_metadata(model_id, metadata, metadata_names)

    logger.info("Successfully applied tags to model=%s: %s", model_id, applied)
    return {"message": "Datapoints successfully tagged.", "applied": applied}


def _apply_tags_to_metadata(
    tagging: dict[str, list[list[int]]],
    metadata: np.ndarray,
    tags_col: int,
) -> dict[str, int]:
    """Mutate *metadata* in place, appending tags idempotently.

    Returns a dict mapping each tag name to the number of rows processed.
    """
    total_rows = len(metadata)
    applied: dict[str, int] = {}

    for tag_name, ranges in tagging.items():
        rows_tagged = 0
        for r in ranges:
            start, end = _parse_range(r)
            _validate_range(start, end, total_rows)
            for idx in range(start, end):
                existing = _extract_tags(metadata[idx][tags_col])
                if tag_name not in existing:
                    existing.append(tag_name)
                    metadata[idx][tags_col] = existing
                rows_tagged += 1
        applied[tag_name] = rows_tagged

    return applied


async def _persist_metadata(
    model_id: str,
    metadata: np.ndarray,
    metadata_names: list[str],
) -> None:
    """Replace the metadata dataset for a model.

    Reads the old data first as a backup, then deletes and rewrites.
    If the write fails, attempts to restore the original data to
    prevent permanent data loss.

    .. todo:: Read-modify-write is not atomic; concurrent tag requests may conflict.
    """
    metadata_dataset = model_id + METADATA_SUFFIX
    try:
        old_metadata = await storage_interface.read_data(metadata_dataset)
    except Exception:  # Intentional: if read fails, there's nothing to restore
        old_metadata = None

    try:
        await storage_interface.delete_dataset(metadata_dataset)
        await storage_interface.write_data(metadata_dataset, metadata, metadata_names)
    except Exception as exc:
        logger.exception("Error writing tagged metadata for model=%s", model_id)
        if old_metadata is not None:
            try:
                await storage_interface.write_data(
                    metadata_dataset, old_metadata, metadata_names
                )
                logger.info("Restored original metadata for model=%s", model_id)
            except Exception:  # Intentional: restoration is best-effort
                logger.exception(
                    "Failed to restore metadata for model=%s — data may be lost",
                    model_id,
                )
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Error persisting tags for model={model_id}",
        ) from exc
