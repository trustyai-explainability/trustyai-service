# endpoints/consumer.py
"""Consumer endpoint for handling KServe inference requests and cloud events."""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Annotated, Never

import numpy as np
from fastapi import APIRouter, Header, HTTPException
from numpy import ndarray

from trustyai_service.endpoints.consumer import (
    InferencePartialPayload,
    KServeData,
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.exceptions import ReconciliationError
from trustyai_service.service.data.datasources.data_source import DataSource

# Import local dependencies
from trustyai_service.service.data.model_data import ModelData
from trustyai_service.service.data.modelmesh_parser import (
    ModelMeshPayloadParser,
    PartialPayload,
)
from trustyai_service.service.data.shared_data_source import get_shared_data_source
from trustyai_service.service.data.storage import get_global_storage_interface
from trustyai_service.service.utils import list_utils

# Define constants locally to avoid import issues
INPUT_SUFFIX = "_inputs"
OUTPUT_SUFFIX = "_outputs"
METADATA_SUFFIX = "_metadata"
SYNTHETIC_TAG = "synthetic"
UNLABELED_TAG = "unlabeled"
BIAS_IGNORE_PARAM = "bias-ignore"

router = APIRouter()
logger = logging.getLogger(__name__)

unreconciled_inputs = {}
unreconciled_outputs = {}


def get_data_source() -> DataSource:
    """Get the shared data source instance."""
    return get_shared_data_source()


def _validate_payload_type(payload: object, expected_type: type) -> None:
    """Validate payload type from storage.

    :param payload: The payload to validate
    :param expected_type: Expected type class
    :raises HTTPException: If payload type doesn't match expected
    """
    if not isinstance(payload, expected_type):
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Invalid payload type from storage",
        )


@router.post("/consumer/kserve/v2")
async def consume_inference_payload(
    payload: InferencePartialPayload,
) -> dict[str, str]:
    """Process a KServe v2 payload.

    This endpoint accepts both input (request) and output (response) payloads from ModelMesh-served models
    and stores them for reconciliation. When both input and output payloads for the same ID are available,
    they are reconciled and stored as data.

    Args:
        payload: The KServe v2 payload containing either request or response data

    Returns:
        A JSON response indicating success or failure

    """
    storage_interface = get_global_storage_interface()

    # Validate required fields before processing
    if not payload.id:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Payload requires 'id' field"
        )

    if not payload.kind:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Payload must specify 'kind' as either 'request' or 'response'",
        )

    if not payload.modelid:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Payload requires 'modelid' field",
        )

    if not payload.data:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Payload requires 'data' field containing base64-encoded data",
        )

    try:
        partial_payload = PartialPayload(data=payload.data)
        if payload.kind == "request":
            logger.info(
                "Received partial input payload from model=%s, id=%s",
                payload.modelid,
                payload.id,
            )

            try:
                ModelMeshPayloadParser.parse_input_payload(partial_payload)
                is_input = True
            except ValueError as e:
                logger.exception("Invalid input payload")
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Invalid input payload: {e!s}",
                ) from e

            # Store the input payload
            await storage_interface.persist_partial_payload(
                partial_payload, payload_id=payload.id, is_input=is_input
            )

            output_payload = await storage_interface.get_partial_payload(
                payload.id, is_input=False, is_modelmesh=True
            )

            if output_payload:
                _validate_payload_type(output_payload, PartialPayload)
                await reconcile_modelmesh_payloads(
                    partial_payload, output_payload, payload.id, payload.modelid
                )

        elif payload.kind == "response":
            logger.info(
                "Received partial output payload from model=%s, id=%s",
                payload.modelid,
                payload.id,
            )

            try:
                ModelMeshPayloadParser.parse_output_payload(partial_payload)
                is_input = False
            except ValueError as e:
                logger.exception("Invalid output payload")
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f"Invalid output payload: {e!s}",
                ) from e

            # Store the output payload
            await storage_interface.persist_partial_payload(
                payload=partial_payload, payload_id=payload.id, is_input=is_input
            )

            input_payload = await storage_interface.get_partial_payload(
                payload.id, is_input=True, is_modelmesh=True
            )

            if input_payload:
                # We have both input and output. Reconcile them
                _validate_payload_type(input_payload, PartialPayload)
                await reconcile_modelmesh_payloads(
                    input_payload, partial_payload, payload.id, payload.modelid
                )
    except HTTPException:
        # HTTPException always goes through
        raise
    except (
        Exception
    ) as e:  # Broad catch intentional: endpoint catch-all for unknown processing errors
        logger.exception("Error processing payload")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the payload",
        ) from e
    else:
        return {
            "status": "success",
            "message": f"Payload for {payload.id} processed successfully",
        }


async def write_reconciled_data(
    input_array: ndarray[tuple[int, int]] | ndarray,
    input_names: list[str],
    output_array: ndarray[tuple[int, int]] | ndarray,
    output_names: list[str],
    model_id: str,
    tags: list[str],
    id_: str,
) -> None:
    """Write reconciled input/output data and metadata to storage.

    :param input_array: NumPy array of input data
    :param input_names: List of input column names
    :param output_array: NumPy array of output data
    :param output_names: List of output column names
    :param model_id: Model identifier
    :param tags: List of tags to associate with the data
    :param id_: Request ID for this inference
    """
    storage_interface = get_global_storage_interface()

    iso_time = datetime.now(UTC).isoformat()
    unix_timestamp = time.time()
    metadata = np.array(
        [[None, iso_time, unix_timestamp, tags]] * len(input_array), dtype="O"
    )
    metadata[:, 0] = [f"{id_}_{i}" for i in range(len(input_array))]
    metadata_names = ["id", "iso_time", "unix_timestamp", "tags"]

    input_dataset = model_id + INPUT_SUFFIX
    output_dataset = model_id + OUTPUT_SUFFIX
    metadata_dataset = model_id + METADATA_SUFFIX

    await asyncio.gather(
        storage_interface.write_data(input_dataset, input_array, input_names),
        storage_interface.write_data(output_dataset, output_array, output_names),
        storage_interface.write_data(metadata_dataset, metadata, metadata_names),
    )

    shapes = await ModelData(model_id).shapes()
    logger.info(
        "Successfully reconciled inference %s, consisting of %s rows from %s.",
        id_,
        f"{len(input_array):,}",
        model_id,
    )
    logger.debug(
        "Current storage shapes for %s: Inputs=%s, Outputs=%s, Metadata=%s",
        model_id,
        shapes[0],
        shapes[1],
        shapes[2],
    )

    # Add model to known models set so it can be discovered by the scheduler
    data_source = get_data_source()
    await data_source.add_model_to_known(model_id)
    known_models = await data_source.get_known_models()
    logger.info(
        "Added model %s to known models set. Current known models: %s",
        model_id,
        list(known_models),
    )
    logger.debug("DataSource instance id: %s", id(data_source))

    # Mark that inference data has been recorded for this model
    try:
        metadata = await data_source.get_metadata(model_id)
        metadata.set_recorded_inferences(recorded_inferences=True)
    except (
        Exception
    ) as e:  # Intentional: metadata update is non-critical; continue on failure
        logger.warning(
            "Could not update recorded_inferences flag for model %s: %s", model_id, e
        )
    else:
        logger.info("Marked model %s as having recorded inferences", model_id)

    # Clean up
    await storage_interface.delete_partial_payload(id_, is_input=True)
    await storage_interface.delete_partial_payload(id_, is_input=False)


async def reconcile_modelmesh_payloads(
    input_payload: PartialPayload,
    output_payload: PartialPayload,
    request_id: str,
    model_id: str,
) -> None:
    """Reconcile the input and output ModelMesh payloads into dataset entries."""
    df = ModelMeshPayloadParser.payloads_to_dataframe(
        input_payload, output_payload, request_id, model_id
    )

    input_cols = [
        col
        for col in df.columns
        if not col.startswith("output_") and col not in ["id", "model_id", "synthetic"]
    ]
    output_cols = [col for col in df.columns if col.startswith("output_")]

    # Create metadata array
    tags = [SYNTHETIC_TAG] if any(df["synthetic"]) else [UNLABELED_TAG]

    await write_reconciled_data(
        df[input_cols].values,
        input_cols,
        df[output_cols].values,
        output_cols,
        model_id=model_id,
        tags=tags,
        id_=request_id,
    )


async def reconcile_kserve(
    input_payload: KServeInferenceRequest,
    output_payload: KServeInferenceResponse,
    tag: str | None,
) -> None:
    """Reconcile KServe v2 request and response payloads into storage.

    :param input_payload: KServe inference request containing inputs
    :param output_payload: KServe inference response containing outputs
    :param tag: Optional tag to associate with the data
    """
    input_array, input_names = process_payload(input_payload, lambda p: p.inputs)
    output_array, output_names = process_payload(
        output_payload, lambda p: p.outputs, input_array.shape[0]
    )

    if tag is not None:
        tags = [tag]
    elif (
        input_payload.parameters is not None
        and input_payload.parameters.get(BIAS_IGNORE_PARAM, "false") == "true"
    ):
        tags = [SYNTHETIC_TAG]
    else:
        tags = [UNLABELED_TAG]

    await write_reconciled_data(
        input_array,
        input_names,
        output_array,
        output_names,
        model_id=output_payload.model_name,
        tags=tags,
        id_=input_payload.id,
    )


def reconcile_mismatching_shape_error(
    shape_tuples: list[tuple[str, list[int]]], payload_type: str, payload_id: str
) -> Never:
    """Raise ReconciliationError for mismatched tensor shapes.

    :param shape_tuples: List of (name, shape) tuples for tensors
    :param payload_type: Type of payload ('input' or 'output')
    :param payload_id: ID of the payload being reconciled
    :raises ReconciliationError: Always raises with detailed shape information
    """
    msg = (
        f"Could not reconcile KServe Inference {payload_id}, because {payload_type} shapes were mismatched. "
        f"When using multiple {payload_type}s to describe data columns, all shapes must match. "
        f"However, the following tensor shapes were found:"
    )
    for i, (name, shape) in enumerate(shape_tuples):
        msg += f"\n{i}:\t{name}:\t{shape}"
    raise ReconciliationError(msg, payload_id=payload_id)


def reconcile_mismatching_row_count_error(
    payload_id: str, input_shape: int, output_shape: int
) -> Never:
    """Raise ReconciliationError for mismatched input/output row counts.

    :param payload_id: ID of the payload being reconciled
    :param input_shape: Number of input rows
    :param output_shape: Number of output rows
    :raises ReconciliationError: Always raises with row count details
    """
    msg = (
        f"Could not reconcile KServe Inference {payload_id}, because the number of "
        f"output rows ({output_shape}) did not match the number of input rows "
        f"({input_shape})."
    )
    raise ReconciliationError(msg, payload_id=payload_id)


def process_payload(
    payload: KServeInferenceRequest | KServeInferenceResponse,
    get_data: Callable,
    enforced_first_shape: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Process a KServe payload and extract data array and column names.

    :param payload: KServe request or response payload
    :param get_data: Function to extract inputs or outputs from payload
    :param enforced_first_shape: Expected number of rows (for validation)
    :return: Tuple of (data array, column names list)
    :raises ReconciliationError: If shapes don't match expectations
    """
    if (
        len(get_data(payload)) > 1
    ):  # multi tensor case: we have ncols of data of shape [nrows]
        data = []
        shapes = set()
        shape_tuples = []
        column_names = []
        for kserve_data in get_data(payload):
            data.append(kserve_data.data)
            shapes.add(tuple(kserve_data.shape))
            column_names.append(kserve_data.name)
            shape_tuples.append((kserve_data.name, kserve_data.shape))
        if len(shapes) == 1:
            row_count = next(iter(shapes))[0]
            if enforced_first_shape is not None and row_count != enforced_first_shape:
                reconcile_mismatching_row_count_error(
                    payload.id, enforced_first_shape, row_count
                )
            if list_utils.contains_non_numeric(data):
                return np.array(data, dtype="O").T, column_names
            return np.array(data).T, column_names
        reconcile_mismatching_shape_error(
            shape_tuples,
            "input" if enforced_first_shape is None else "output",
            payload.id,
        )
    else:  # single tensor case: we have one tensor of shape [nrows, d1, d2, ...., dN]
        kserve_data: KServeData = get_data(payload)[0]
        if (
            enforced_first_shape is not None
            and kserve_data.shape[0] != enforced_first_shape
        ):
            reconcile_mismatching_row_count_error(
                payload.id, enforced_first_shape, kserve_data.shape[0]
            )

        if len(kserve_data.shape) > 1:
            column_names = [
                f"{kserve_data.name}-{i}" for i in range(kserve_data.shape[1])
            ]
        else:
            column_names = [kserve_data.name]
        if list_utils.contains_non_numeric(kserve_data.data):
            return np.array(kserve_data.data, dtype="O"), column_names
        return np.array(kserve_data.data), column_names


@router.post("/")
async def consume_cloud_event(
    payload: KServeInferenceRequest | KServeInferenceResponse,
    ce_id: Annotated[str | None, Header()] = None,
    tag: str | None = None,
) -> dict[str, str]:
    """Consume KServe v2 payloads from cloud events.

    This endpoint accepts both input (request) and output (response) payloads
    from ModelMesh-served models and stores them for reconciliation.

    :param payload: KServe inference request or response
    :param ce_id: Cloud event ID from header
    :param tag: Optional tag to associate with the data
    :raises HTTPException: If payload processing fails
    """
    # set payload id from cloud event header if present
    if ce_id is not None:
        payload.id = ce_id

    if payload.id is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Payload requires 'id' field or 'ce-id' header",
        )

    # get global storage interface
    storage_interface = get_global_storage_interface()

    try:
        if isinstance(payload, KServeInferenceRequest):
            if len(payload.inputs) == 0:
                msg = (
                    f"KServe Inference Input {payload.id} received, but data field was empty. "
                    f"Payload will not be saved."
                )
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=msg)
            logger.info("KServe Inference Input %s received.", payload.id)
            # if a match is found, the payload is auto-deleted from data
            partial_output = await storage_interface.get_partial_payload(
                payload.id, is_input=False, is_modelmesh=False
            )
            if partial_output is not None:
                if not isinstance(partial_output, KServeInferenceResponse):
                    # This should never happen - indicates storage interface error
                    raise HTTPException(
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        detail="Invalid payload type from storage",
                    )
                await reconcile_kserve(payload, partial_output, tag)
            else:
                await storage_interface.persist_partial_payload(
                    payload, payload_id=payload.id, is_input=True
                )
            return {
                "status": "success",
                "message": f"Input payload {payload.id} processed successfully",
            }

        if isinstance(payload, KServeInferenceResponse):
            if len(payload.outputs) == 0:
                msg = (
                    f"KServe Inference Output {payload.id} received from model={payload.model_name}, "
                    f"but data field was empty. Payload will not be saved."
                )
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=msg)
            logger.info(
                "KServe Inference Output %s received from model=%s.",
                payload.id,
                payload.model_name,
            )
            partial_input = await storage_interface.get_partial_payload(
                payload.id, is_input=True, is_modelmesh=False
            )
            if partial_input is not None:
                if not isinstance(partial_input, KServeInferenceRequest):
                    # This should never happen - indicates storage interface error
                    raise HTTPException(
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        detail="Invalid payload type from storage",
                    )
                await reconcile_kserve(partial_input, payload, tag)
            else:
                await storage_interface.persist_partial_payload(
                    payload, payload_id=payload.id, is_input=False
                )

            return {
                "status": "success",
                "message": f"Output payload {payload.id} processed successfully",
            }

        # Defensive programming: this should never happen due to type annotation
        # but adding explicit fallback for type safety
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Payload must be either KServeInferenceRequest or KServeInferenceResponse",
        )

    except ReconciliationError as e:
        logger.exception("Reconciliation failed for payload %s", payload.id)
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e)) from e
