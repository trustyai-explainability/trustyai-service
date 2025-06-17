# endpoints/consumer.py
import asyncio
import time
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Optional, Literal, List, Union, Callable, Annotated
import logging

# Import local dependencies
from src.service.data.model_data import ModelData
from src.service.data.storage import get_storage_interface
from src.service.utils import list_utils
from src.service.data.modelmesh_parser import ModelMeshPayloadParser, PartialPayload

# Define constants locally to avoid import issues
INPUT_SUFFIX = "_inputs"
OUTPUT_SUFFIX = "_outputs"
METADATA_SUFFIX = "_metadata"
SYNTHETIC_TAG = "synthetic"
UNLABELED_TAG = "unlabeled"
BIAS_IGNORE_PARAM = "bias-ignore"

router = APIRouter()
logger = logging.getLogger(__name__)

PartialKind = Literal["request", "response"]
storage_interface = get_storage_interface()
unreconciled_inputs = {}
unreconciled_outputs = {}


class PartialPayloadId(BaseModel):
    prediction_id: Optional[str] = None
    kind: Optional[PartialKind] = None

    def get_prediction_id(self) -> str:
        return self.prediction_id

    def set_prediction_id(self, id: str):
        self.prediction_id = id

    def get_kind(self) -> PartialKind:
        return self.kind

    def set_kind(self, kind: PartialKind):
        self.kind = kind


class InferencePartialPayload(BaseModel):
    partialPayloadId: Optional[PartialPayloadId] = None
    metadata: Optional[Dict[str, str]] = {}
    data: Optional[str] = None
    modelid: Optional[str] = None

    def get_id(self) -> str:
        return self.partialPayloadId.prediction_id if self.partialPayloadId else None

    def set_id(self, id: str):
        if not self.partialPayloadId:
            self.partialPayloadId = PartialPayloadId()
        self.partialPayloadId.prediction_id = id

    def get_kind(self) -> PartialKind:
        return self.partialPayloadId.kind if self.partialPayloadId else None

    def set_kind(self, kind: PartialKind):
        if not self.partialPayloadId:
            self.partialPayloadId = PartialPayloadId()
        self.partialPayloadId.kind = kind

    def get_model_id(self) -> str:
        return self.modelid

    def set_model_id(self, model_id: str):
        self.modelid = model_id


class KServeData(BaseModel):
    name: str
    shape: List[int]
    datatype: str
    parameters: Optional[Dict[str, str]] = None
    data: List


class KServeInferenceRequest(BaseModel):
    id: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None
    inputs: List[KServeData]
    outputs: Optional[List[KServeData]] = None


class KServeInferenceResponse(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    id: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None
    outputs: List[KServeData]


@router.post("/consumer/kserve/v2")
async def consume_inference_payload(
    payload: InferencePartialPayload,
):
    """
    Process a KServe v2 payload.

    This endpoint accepts both input (request) and output (response) payloads from ModelMesh-served models
    and stores them for reconciliation. When both input and output payloads for the same ID are available,
    they are reconciled and stored as data.

    Args:
        payload: The KServe v2 payload containing either request or response data

    Returns:
        A JSON response indicating success or failure
    """
    try:
        if not payload.modelid:
            raise HTTPException(
                status_code=400, detail="Payload requires 'modelid' field"
            )

        payload_id = payload.get_id()
        payload_kind = payload.get_kind()
        model_id = payload.get_model_id()

        if not payload_id:
            raise HTTPException(status_code=400, detail="Payload requires 'id' field")

        if not payload_kind:
            raise HTTPException(
                status_code=400,
                detail="Payload must specify 'kind' as either 'request' or 'response'",
            )

        if not payload.data:
            raise HTTPException(
                status_code=400,
                detail="Payload requires 'data' field containing base64-encoded data",
            )

        partial_payload = PartialPayload(data=payload.data)

        if payload_kind == "request":
            logger.info(
                f"Received partial input payload from model={model_id}, id={payload_id}"
            )

            try:
                ModelMeshPayloadParser.parse_input_payload(partial_payload)
                is_input = True
            except Exception as e:
                logger.error(f"Invalid input payload: {str(e)}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid input payload: {str(e)}"
                ) from e

            # Store the input payload
            await storage_interface.persist_modelmesh_payload(
                partial_payload, payload_id, is_input
            )

            output_payload = await storage_interface.get_modelmesh_payload(
                payload_id, False
            )

            if output_payload:
                await reconcile_modelmesh_payloads(
                    partial_payload, output_payload, payload_id, model_id
                )

        elif payload_kind == "response":
            logger.info(
                f"Received partial output payload from model={model_id}, id={payload_id}"
            )

            try:
                ModelMeshPayloadParser.parse_output_payload(partial_payload)
                is_input = False
            except Exception as e:
                logger.error(f"Invalid output payload: {str(e)}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid output payload: {str(e)}"
                ) from e

            # Store the output payload
            await storage_interface.persist_modelmesh_payload(
                partial_payload, payload_id, is_input
            )

            input_payload = await storage_interface.get_modelmesh_payload(
                payload_id, True
            )

            if input_payload:
                # We have both input and output. Reconcile them
                await reconcile_modelmesh_payloads(
                    input_payload, partial_payload, payload_id, model_id
                )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported payload kind={payload_kind}"
            )

        return {
            "status": "success",
            "message": f"Payload for {payload_id} processed successfully",
        }

    except HTTPException:
        # HTTPException always goes through
        raise
    except Exception as e:
        logger.error(f"Error processing payload: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing payload: {str(e)}"
        ) from e


async def reconcile_modelmesh_payloads(
    input_payload: PartialPayload,
    output_payload: PartialPayload,
    request_id: str,
    model_id: str,
):
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
    iso_time = datetime.isoformat(datetime.utcnow())
    unix_timestamp = time.time()
    metadata = np.array([[iso_time, unix_timestamp, tags]] * len(df), dtype="O")

    # Get dataset names
    input_dataset = model_id + INPUT_SUFFIX
    output_dataset = model_id + OUTPUT_SUFFIX
    metadata_dataset = model_id + METADATA_SUFFIX

    metadata_cols = ["iso_time", "unix_timestamp", "tags"]

    await asyncio.gather(
        storage_interface.write_data(input_dataset, df[input_cols].values, input_cols),
        storage_interface.write_data(
            output_dataset, df[output_cols].values, output_cols
        ),
        storage_interface.write_data(metadata_dataset, metadata, metadata_cols),
    )

    shapes = await ModelData(model_id).shapes()
    logger.info(
        f"Successfully reconciled ModelMesh inference {request_id}, "
        f"consisting of {len(df):,} rows from {model_id}."
    )
    logger.debug(
        f"Current storage shapes for {model_id}: "
        f"Inputs={shapes[0]}, "
        f"Outputs={shapes[1]}, "
        f"Metadata={shapes[2]}"
    )

    # Clean up
    await storage_interface.delete_modelmesh_payload(request_id, True)
    await storage_interface.delete_modelmesh_payload(request_id, False)


def reconcile_mismatching_shape_error(shape_tuples, payload_type, payload_id):
    msg = (
        f"Could not reconcile KServe Inference {payload_id}, because {payload_type} shapes were mismatched. "
        f"When using multiple {payload_type}s to describe data columns, all shapes must match."
        f"However, the following tensor shapes were found:"
    )
    for i, (name, shape) in enumerate(shape_tuples):
        msg += f"\n{i}:\t{name}:\t{shape}"
    logger.error(msg)
    raise HTTPException(status_code=400, detail=msg)


def reconcile_mismatching_row_count_error(payload_id, input_shape, output_shape):
    msg = (
        f"Could not reconcile KServe Inference {payload_id}, because the number of "
        f"output rows ({output_shape}) did not match the number of input rows "
        f"({input_shape})."
    )
    logger.error(msg)
    raise HTTPException(status_code=400, detail=msg)


def process_payload(payload, get_data: Callable, enforced_first_shape: int = None):
    if (
        len(get_data(payload)) > 1
    ):  # multi tensor case: we have ncols of data of shape [nrows]
        data = []
        shapes = set()
        shape_tuples = []
        column_names = []
        for kserve_data in get_data(payload):
            data.append(kserve_data.data)
            shapes.add(tuple(kserve_data.data.shape))
            column_names.append(kserve_data.name)
            shape_tuples.append((kserve_data.data.name, kserve_data.data.shape))
        if len(shapes) == 1:
            row_count = list(shapes)[0][0]
            if enforced_first_shape is not None and row_count != enforced_first_shape:
                reconcile_mismatching_row_count_error(
                    payload.id, enforced_first_shape, row_count
                )
            if list_utils.contains_non_numeric(data):
                return np.array(data, dtype="O").T, column_names
            else:
                return np.array(data).T, column_names
        else:
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
                "{}-{}".format(kserve_data.name, i) for i in range(kserve_data.shape[1])
            ]
        else:
            column_names = [kserve_data.name]
        if list_utils.contains_non_numeric(kserve_data.data):
            return np.array(kserve_data.data, dtype="O"), column_names
        else:
            return np.array(kserve_data.data), column_names


async def reconcile(
    input_payload: KServeInferenceRequest, output_payload: KServeInferenceResponse
):
    input_array, input_names = process_payload(input_payload, lambda p: p.inputs)
    output_array, output_names = process_payload(
        output_payload, lambda p: p.outputs, input_array.shape[0]
    )

    metadata_names = ["iso_time", "unix_timestamp", "tags"]
    if (
        input_payload.parameters is not None
        and input_payload.parameters.get(BIAS_IGNORE_PARAM, "false") == "true"
    ):
        tags = [SYNTHETIC_TAG]
    else:
        tags = [UNLABELED_TAG]
    iso_time = datetime.isoformat(datetime.utcnow())
    unix_timestamp = time.time()
    metadata = np.array(
        [[iso_time, unix_timestamp, tags]] * len(input_array), dtype="O"
    )

    input_dataset = output_payload.model_name + INPUT_SUFFIX
    output_dataset = output_payload.model_name + OUTPUT_SUFFIX
    metadata_dataset = output_payload.model_name + METADATA_SUFFIX

    await asyncio.gather(
        storage_interface.write_data(input_dataset, input_array, input_names),
        storage_interface.write_data(output_dataset, output_array, output_names),
        storage_interface.write_data(metadata_dataset, metadata, metadata_names),
    )

    shapes = await ModelData(output_payload.model_name).shapes()
    logger.info(
        f"Successfully reconciled KServe inference {input_payload.id}, "
        f"consisting of {input_array.shape[0]:,} rows from {output_payload.model_name}."
    )
    logger.debug(
        f"Current storage shapes for {output_payload.model_name}: "
        f"Inputs={shapes[0]}, "
        f"Outputs={shapes[1]}, "
        f"Metadata={shapes[2]}"
    )


@router.post("/")
async def consume_cloud_event(
    payload: Union[KServeInferenceRequest, KServeInferenceResponse],
    ce_id: Annotated[str | None, Header()] = None,
):
    # set payload if from cloud event header
    payload.id = ce_id

    if isinstance(payload, KServeInferenceRequest):
        if len(payload.inputs) == 0:
            msg = f"KServe Inference Input {payload.id} received, but data field was empty. Payload will not be saved."
            logger.error(msg)
            raise HTTPException(status_code=400, detail=msg)
        else:
            logger.info(f"KServe Inference Input {payload.id} received.")
            # if a match is found, the payload is auto-deleted from data
            partial_output = await storage_interface.get_partial_payload(
                payload.id, is_input=False
            )
            if partial_output is not None:
                await reconcile(payload, partial_output)
            else:
                await storage_interface.persist_partial_payload(payload, is_input=True)
            return {
                "status": "success",
                "message": f"Input payload {payload.id} processed successfully",
            }

    elif isinstance(payload, KServeInferenceResponse):
        if len(payload.outputs) == 0:
            msg = (
                f"KServe Inference Output {payload.id} received from model={payload.model_name}, "
                f"but data field was empty. Payload will not be saved."
            )
            logger.error(msg)
            raise HTTPException(status_code=400, detail=msg)
        else:
            logger.info(
                f"KServe Inference Output {payload.id} received from model={payload.model_name}."
            )
            partial_input = await storage_interface.get_partial_payload(
                payload.id, is_input=True
            )
            if partial_input is not None:
                await reconcile(partial_input, payload)
            else:
                await storage_interface.persist_partial_payload(payload, is_input=False)

        return {
            "status": "success",
            "message": f"Output payload {payload.id} processed successfully",
        }
