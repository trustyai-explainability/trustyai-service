# endpoints/consumer.py
import asyncio
import time
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Dict, Optional, Literal, List, Union, Callable, Annotated
import logging

from src.service.constants import *
from src.service.data.model_data import ModelData
from src.service.data.storage import get_storage_interface
from src.service.utils import list_utils

router = APIRouter()
logger = logging.getLogger(__name__)

PartialKind = Literal["request", "response"]
storage_inferface = get_storage_interface()
unreconciled_inputs = {}
unreconciled_outputs = {}


class PartialPayloadId(BaseModel):
    pass


class InferencePartialPayload(BaseModel):
    partialPayloadId: Optional[PartialPayloadId] = None
    metadata: Optional[Dict[str, str]] = None
    id: Optional[str] = None
    kind: Optional[PartialKind] = None
    data: Optional[str] = None
    modelid: Optional[str] = None


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


def reconcile_mismatching_shape_error(shape_tuples, payload_type, payload_id):
    msg = (f"Could not reconcile KServe Inference {payload_id}, because {payload_type} shapes were mismatched. "
           f"When using multiple {payload_type}s to describe data columns, all shapes must match."
           f"However, the following tensor shapes were found:")
    for i, (name, shape) in enumerate(shape_tuples):
        msg += f"\n{i}:\t{name}:\t{shape}"
    logger.error(msg)
    raise HTTPException(status_code=400, detail=msg)


def reconcile_mismatching_row_count_error(payload_id, input_shape, output_shape):
    msg = (f"Could not reconcile KServe Inference {payload_id}, because the number of "
           f"output rows ({output_shape}) did not match the number of input rows "
           f"({input_shape}).")
    logger.error(msg)
    raise HTTPException(status_code=400, detail=msg)


def process_payload(payload, get_data: Callable, enforced_first_shape: int = None):
    if len(get_data(payload)) > 1:  # multi tensor case: we have ncols of data of shape [nrows]
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
                reconcile_mismatching_row_count_error(payload.id, enforced_first_shape, row_count)
            if list_utils.contains_non_numeric(data):
                return np.array(data, dtype="O").T, column_names
            else:
                return np.array(data).T, column_names
        else:
            reconcile_mismatching_shape_error(
                shape_tuples,
                "input" if enforced_first_shape is None else "output",
                payload.id
            )
    else:  # single tensor case: we have one tensor of shape [nrows, d1, d2, ...., dN]
        kserve_data: KServeData = get_data(payload)[0]
        if enforced_first_shape is not None and kserve_data.shape[0] != enforced_first_shape:
            reconcile_mismatching_row_count_error(payload.id, enforced_first_shape, kserve_data.shape[0])

        if len(kserve_data.shape) > 1:
            column_names = ["{}-{}".format(kserve_data.name, i) for i in range(kserve_data.shape[1])]
        else:
            column_names = [kserve_data.name]
        if list_utils.contains_non_numeric(kserve_data.data):
            return np.array(kserve_data.data, dtype="O"), column_names
        else:
            return np.array(kserve_data.data), column_names


async def reconcile(input_payload: KServeInferenceRequest, output_payload: KServeInferenceResponse):
    input_array, input_names = process_payload(input_payload, lambda p: p.inputs)
    output_array, output_names = process_payload(output_payload, lambda p: p.outputs, input_array.shape[0])

    metadata_names = ["iso_time", "unix_timestamp", "tags"]
    if input_payload.parameters is not None and input_payload.parameters.get(BIAS_IGNORE_PARAM, "false") == "true":
        tags = [SYNTHETIC_TAG]
    else:
        tags = [UNLABELED_TAG]
    iso_time = datetime.isoformat(datetime.utcnow())
    unix_timestamp = time.time()
    metadata = np.array([[iso_time, unix_timestamp, tags]] * len(input_array), dtype="O")

    input_dataset = output_payload.model_name + INPUT_SUFFIX
    output_dataset = output_payload.model_name + OUTPUT_SUFFIX
    metadata_dataset = output_payload.model_name + METADATA_SUFFIX

    async with asyncio.TaskGroup() as tg:
        tg.create_task(storage_inferface.write_data(input_dataset, input_array, input_names))
        tg.create_task(storage_inferface.write_data(output_dataset, output_array, output_names))
        tg.create_task(storage_inferface.write_data(metadata_dataset, metadata, metadata_names))

    shapes = await (ModelData(output_payload.model_name).shapes())
    logger.info(f"Successfully reconciled KServe inference {input_payload.id}, "
                f"consisting of {input_array.shape[0]:,} rows from {output_payload.model_name}.")
    logger.debug(f"Current storage shapes for {output_payload.model_name}: "
                 f"Inputs={shapes[0]}, "
                 f"Outputs={shapes[1]}, "
                 f"Metadata={shapes[2]}")


@router.post("/")
async def consume_cloud_event(payload: Union[KServeInferenceRequest, KServeInferenceResponse],
                              ce_id: Annotated[str | None, Header()] = None):
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
            partial_output = await storage_inferface.get_partial_payload(payload.id, is_input=False)
            if partial_output is not None:
                await reconcile(payload, partial_output)
            else:
                await storage_inferface.persist_partial_payload(payload, is_input=True)
            return {"status": "success", "message": f"Input payload {payload.id} processed successfully"}

    elif isinstance(payload, KServeInferenceResponse):
        if len(payload.outputs) == 0:
            msg = (f"KServe Inference Output {payload.id} received from model={payload.model_name}, "
                   f"but data field was empty. Payload will not be saved.")
            logger.error(msg)
            raise HTTPException(status_code=400, detail=msg)
        else:
            logger.info(f"KServe Inference Output {payload.id} received from model={payload.model_name}.")
            partial_input = await storage_inferface.get_partial_payload(payload.id, is_input=True)
            if partial_input is not None:
                await reconcile(partial_input, payload)
            else:
                await storage_inferface.persist_partial_payload(payload, is_input=False)

        return {"status": "success", "message": f"Output payload {payload.id} processed successfully"}
