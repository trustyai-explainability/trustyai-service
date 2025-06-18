import copy
import json
import os
import pickle
import shutil
import sys
import tempfile
import uuid

import h5py
import numpy as np
import pytest

TEMP_DIR = tempfile.mkdtemp()
os.environ["STORAGE_DATA_FOLDER"] = TEMP_DIR
from fastapi.testclient import TestClient

from src.main import app
from src.service.constants import (
    INPUT_SUFFIX,
    METADATA_SUFFIX,
    OUTPUT_SUFFIX,
    TRUSTYAI_TAG_PREFIX,
)
from src.service.data.storage import get_storage_interface


def pytest_sessionfinish(session, exitstatus):
    """Clean up the temporary directory after all tests are done."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


pytest.hookimpl(pytest_sessionfinish)
client = TestClient(app)
MODEL_ID = "example1"


def generate_payload(n_rows, n_input_cols, n_output_cols, datatype, tag, input_offset=0, output_offset=0):
    """Generate a test payload with specific dimensions and data types."""
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    input_data = []
    for i in range(n_rows):
        if n_input_cols == 1:
            input_data.append(i + input_offset)
        else:
            row = [i + j + input_offset for j in range(n_input_cols)]
            input_data.append(row)
    output_data = []
    for i in range(n_rows):
        if n_output_cols == 1:
            output_data.append(i * 2 + output_offset)
        else:
            row = [i * 2 + j + output_offset for j in range(n_output_cols)]
            output_data.append(row)
    payload = {
        "model_name": model_name,
        "data_tag": tag,
        "is_ground_truth": False,
        "request": {
            "inputs": [
                {
                    "name": "input",
                    "shape": [n_rows, n_input_cols] if n_input_cols > 1 else [n_rows],
                    "datatype": datatype,
                    "data": input_data,
                }
            ]
        },
        "response": {
            "outputs": [
                {
                    "name": "output",
                    "shape": [n_rows, n_output_cols] if n_output_cols > 1 else [n_rows],
                    "datatype": datatype,
                    "data": output_data,
                }
            ]
        },
    }
    return payload


def generate_multi_input_payload(n_rows, n_input_cols, n_output_cols, datatype, tag):
    """Generate a test payload with multi-dimensional tensors like real data."""
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    input_data = []
    for row_idx in range(n_rows):
        row = [row_idx + col_idx * 10 for col_idx in range(n_input_cols)]
        input_data.append(row)
    output_data = []
    for row_idx in range(n_rows):
        row = [row_idx * 2 + col_idx for col_idx in range(n_output_cols)]
        output_data.append(row)
    payload = {
        "model_name": model_name,
        "data_tag": tag,
        "is_ground_truth": False,
        "request": {
            "inputs": [
                {
                    "name": "multi_input",
                    "shape": [n_rows, n_input_cols],
                    "datatype": datatype,
                    "data": input_data,
                }
            ]
        },
        "response": {
            "outputs": [
                {
                    "name": "multi_output",
                    "shape": [n_rows, n_output_cols],
                    "datatype": datatype,
                    "data": output_data,
                }
            ]
        },
    }
    return payload


def generate_mismatched_shape_no_unique_name_multi_input_payload(n_rows, n_input_cols, n_output_cols, datatype, tag):
    """Generate a payload with mismatched shapes and non-unique names."""
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    input_data_1 = [[row_idx + col_idx * 10 for col_idx in range(n_input_cols)] for row_idx in range(n_rows)]
    mismatched_rows = n_rows - 1 if n_rows > 1 else 1
    input_data_2 = [[row_idx + col_idx * 20 for col_idx in range(n_input_cols)] for row_idx in range(mismatched_rows)]
    output_data = [[row_idx * 2 + col_idx for col_idx in range(n_output_cols)] for row_idx in range(n_rows)]
    payload = {
        "model_name": model_name,
        "data_tag": tag,
        "is_ground_truth": False,
        "request": {
            "inputs": [
                {
                    "name": "same_name",
                    "shape": [n_rows, n_input_cols],
                    "datatype": datatype,
                    "data": input_data_1,
                },
                {
                    "name": "same_name",
                    "shape": [mismatched_rows, n_input_cols],
                    "datatype": datatype,
                    "data": input_data_2,
                },
            ]
        },
        "response": {
            "outputs": [
                {
                    "name": "multi_output",
                    "shape": [n_rows, n_output_cols],
                    "datatype": datatype,
                    "data": output_data,
                }
            ]
        },
    }
    return payload


def get_data_from_storage(model_name, suffix):
    """Get data from storage file."""
    storage = get_storage_interface()
    filename = storage._get_filename(model_name + suffix)
    if not os.path.exists(filename):
        return None
    with h5py.File(filename, "r") as f:
        if model_name + suffix in f:
            data = f[model_name + suffix][:]
            column_names = f[model_name + suffix].attrs.get("column_names", [])
            return {"data": data, "column_names": column_names}


def get_metadata_ids(model_name):
    """Extract actual IDs from metadata storage."""
    storage = get_storage_interface()
    filename = storage._get_filename(model_name + METADATA_SUFFIX)
    if not os.path.exists(filename):
        return []
    ids = []
    with h5py.File(filename, "r") as f:
        if model_name + METADATA_SUFFIX in f:
            metadata = f[model_name + METADATA_SUFFIX][:]
            column_names = f[model_name + METADATA_SUFFIX].attrs.get("column_names", [])
            id_idx = next((i for i, name in enumerate(column_names) if name.lower() == "id"), None)
            if id_idx is not None:
                for row in metadata:
                    try:
                        if hasattr(row, "__getitem__") and len(row) > id_idx:
                            id_val = row[id_idx]
                        else:
                            row_data = pickle.loads(row.tobytes())
                            id_val = row_data[id_idx]
                        if isinstance(id_val, np.ndarray):
                            ids.append(str(id_val))
                        else:
                            ids.append(str(id_val))
                    except Exception as e:
                        print(f"Error processing ID from row {len(ids)}: {e}")
                        continue
    print(f"Successfully extracted {len(ids)} IDs: {ids}")
    return ids


def get_metadata_from_storage(model_name):
    """Get metadata directly from storage file."""
    storage = get_storage_interface()
    filename = storage._get_filename(model_name + METADATA_SUFFIX)
    if not os.path.exists(filename):
        return {"data": [], "column_names": []}
    with h5py.File(filename, "r") as f:
        if model_name + METADATA_SUFFIX in f:
            metadata = f[model_name + METADATA_SUFFIX][:]
            column_names = f[model_name + METADATA_SUFFIX].attrs.get("column_names", [])
            parsed_rows = []
            for row in metadata:
                try:
                    row_data = pickle.loads(row.tobytes())
                    parsed_rows.append(row_data)
                except Exception as e:
                    print(f"Error unpickling metadata row: {e}")

            return {"data": parsed_rows, "column_names": column_names}
    return {"data": [], "column_names": []}


def count_rows_with_tag(model_name, tag):
    """Count rows with a specific tag in metadata."""
    storage = get_storage_interface()
    filename = storage._get_filename(model_name + METADATA_SUFFIX)
    if not os.path.exists(filename):
        return 0
    count = 0
    with h5py.File(filename, "r") as f:
        if model_name + METADATA_SUFFIX in f:
            metadata = f[model_name + METADATA_SUFFIX][:]
            column_names = f[model_name + METADATA_SUFFIX].attrs.get("column_names", [])
            tag_idx = next(
                (i for i, name in enumerate(column_names) if name.lower() == "tag"),
                None,
            )
            if tag_idx is not None:
                for row in metadata:
                    try:
                        row_data = pickle.loads(row.tobytes())
                        if tag_idx < len(row_data) and row_data[tag_idx] == tag:
                            count += 1
                    except Exception as e:
                        print(f"Error unpickling tag: {e}")
    return count


def post_test(payload, expected_status_code, check_msgs):
    """Post a payload and check the response."""
    response = client.post("/data/upload", json=payload)
    if response.status_code != expected_status_code:
        print(f"\n=== DEBUG INFO ===")
        print(f"Expected status: {expected_status_code}")
        print(f"Actual status: {response.status_code}")
        print(f"Response text: {response.text}")
        print(f"Response headers: {dict(response.headers)}")
        if hasattr(response, "json"):
            try:
                print(f"Response JSON: {response.json()}")
            except:
                pass
        print(f"==================")

    assert response.status_code == expected_status_code
    return response


# data upload tests
@pytest.mark.parametrize("n_input_rows", [1, 5, 250])
@pytest.mark.parametrize("n_input_cols", [1, 4])
@pytest.mark.parametrize("n_output_cols", [1, 2])
@pytest.mark.parametrize("datatype", ["INT64", "INT32", "FP32", "FP64", "BOOL"])
def test_upload_data(n_input_rows, n_input_cols, n_output_cols, datatype):
    """Test uploading data with various dimensions and datatypes."""
    data_tag = "TRAINING"
    payload = generate_payload(n_input_rows, n_input_cols, n_output_cols, datatype, data_tag)
    response = post_test(payload, 200, [f"{n_input_rows} datapoints"])
    inputs = get_data_from_storage(payload["model_name"], INPUT_SUFFIX)
    outputs = get_data_from_storage(payload["model_name"], OUTPUT_SUFFIX)
    assert inputs is not None, "Input data not found in storage"
    assert outputs is not None, "Output data not found in storage"
    assert len(inputs["data"]) == n_input_rows, "Incorrect number of input rows"
    assert len(outputs["data"]) == n_input_rows, "Incorrect number of output rows"
    tag_count = count_rows_with_tag(payload["model_name"], data_tag)
    assert tag_count == n_input_rows, "Not all rows have the correct tag"


@pytest.mark.parametrize("n_rows", [1, 3, 5, 250])
@pytest.mark.parametrize("n_input_cols", [2, 6])
@pytest.mark.parametrize("n_output_cols", [4])
@pytest.mark.parametrize("datatype", ["INT64", "INT32", "FP32", "FP64", "BOOL"])
def test_upload_multi_input_data(n_rows, n_input_cols, n_output_cols, datatype):
    """Test uploading data with multiple input tensors."""
    data_tag = "TRAINING"
    payload = generate_multi_input_payload(n_rows, n_input_cols, n_output_cols, datatype, data_tag)
    response = post_test(payload, 200, [f"{n_rows} datapoints"])
    inputs = get_data_from_storage(payload["model_name"], INPUT_SUFFIX)
    outputs = get_data_from_storage(payload["model_name"], OUTPUT_SUFFIX)
    assert inputs is not None, "Input data not found in storage"
    assert outputs is not None, "Output data not found in storage"
    assert len(inputs["data"]) == n_rows, "Incorrect number of input rows"
    assert len(outputs["data"]) == n_rows, "Incorrect number of output rows"
    assert len(inputs["column_names"]) == n_input_cols, "Incorrect number of input columns"
    assert len(outputs["column_names"]) == n_output_cols, "Incorrect number of output columns"
    assert len(inputs["column_names"]) >= 2, "Should have at least 2 input column names"
    tag_count = count_rows_with_tag(payload["model_name"], data_tag)
    assert tag_count == n_rows, "Not all rows have the correct tag"


def test_upload_multi_input_data_no_unique_name():
    """Test error case for non-unique tensor names."""
    payload = generate_mismatched_shape_no_unique_name_multi_input_payload(250, 4, 3, "FP64", "TRAINING")
    response = client.post("/data/upload", json=payload)
    assert response.status_code == 400
    assert "One or more errors" in response.text
    assert "unique names" in response.text
    assert "first dimension" in response.text


def test_upload_multiple_tagging():
    """Test uploading data with multiple tags."""
    n_payload1 = 50
    n_payload2 = 51
    tag1 = "TRAINING"
    tag2 = "NOT TRAINING"
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    payload1 = generate_payload(n_payload1, 10, 1, "INT64", tag1)
    payload1["model_name"] = model_name
    post_test(payload1, 200, [f"{n_payload1} datapoints"])
    payload2 = generate_payload(n_payload2, 10, 1, "INT64", tag2)
    payload2["model_name"] = model_name
    post_test(payload2, 200, [f"{n_payload2} datapoints"])
    tag1_count = count_rows_with_tag(model_name, tag1)
    tag2_count = count_rows_with_tag(model_name, tag2)
    assert tag1_count == n_payload1, f"Expected {n_payload1} rows with tag {tag1}"
    assert tag2_count == n_payload2, f"Expected {n_payload2} rows with tag {tag2}"
    inputs = get_data_from_storage(model_name, INPUT_SUFFIX)
    assert len(inputs["data"]) == n_payload1 + n_payload2, "Incorrect total number of rows"


def test_upload_tag_that_uses_protected_name():
    """Test error when using a protected tag name."""
    invalid_tag = f"{TRUSTYAI_TAG_PREFIX}_something"
    payload = generate_payload(5, 10, 1, "INT64", invalid_tag)
    response = post_test(payload, 400, ["reserved for internal TrustyAI use only"])
    expected_msg = f"The tag prefix '{TRUSTYAI_TAG_PREFIX}' is reserved for internal TrustyAI use only. Provided tag '{invalid_tag}' violates this restriction."
    assert expected_msg in response.text


@pytest.mark.parametrize("n_input_rows", [1, 5, 250])
@pytest.mark.parametrize("n_input_cols", [1, 4])
@pytest.mark.parametrize("n_output_cols", [1, 2])
@pytest.mark.parametrize("datatype", ["INT64", "INT32", "FP32", "FP64", "BOOL"])
def test_upload_data_and_ground_truth(n_input_rows, n_input_cols, n_output_cols, datatype):
    """Test uploading model data and corresponding ground truth data."""
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    payload = generate_payload(n_input_rows, n_input_cols, n_output_cols, datatype, "TRAINING")
    payload["model_name"] = model_name
    payload["is_ground_truth"] = False
    post_test(payload, 200, [f"{n_input_rows} datapoints"])
    ids = get_metadata_ids(model_name)
    payload_gt = generate_payload(n_input_rows, n_input_cols, n_output_cols, datatype, "TRAINING", 0, 1)
    payload_gt["model_name"] = model_name
    payload_gt["is_ground_truth"] = True
    payload_gt["request"] = payload["request"]
    payload_gt["request"]["inputs"][0]["execution_ids"] = ids
    post_test(payload_gt, 200, [f"{n_input_rows} ground truths"])
    original_data = get_data_from_storage(model_name, OUTPUT_SUFFIX)
    gt_data = get_data_from_storage(f"{model_name}_ground_truth", OUTPUT_SUFFIX)
    assert len(original_data["data"]) == len(gt_data["data"]), "Row dimensions don't match"
    assert len(original_data["column_names"]) == len(gt_data["column_names"]), "Column dimensions don't match"
    original_ids = get_metadata_ids(model_name)
    gt_ids = get_metadata_ids(f"{model_name}_ground_truth")
    assert original_ids == gt_ids, "Ground truth IDs don't match original IDs"


def test_upload_mismatch_input_values():
    """Test error when ground truth inputs don't match original data."""
    n_input_rows = 5
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    payload0 = generate_payload(n_input_rows, 10, 1, "INT64", "TRAINING")
    payload0["model_name"] = model_name
    post_test(payload0, 200, [f"{n_input_rows} datapoints"])
    ids = get_metadata_ids(model_name)
    payload1 = generate_payload(n_input_rows, 10, 1, "INT64", "TRAINING", 1, 0)
    payload1["model_name"] = model_name
    payload1["is_ground_truth"] = True
    payload1["request"]["inputs"][0]["execution_ids"] = ids
    response = client.post("/data/upload", json=payload1)
    assert response.status_code == 400
    assert "Found fatal mismatches" in response.text or "inputs are not identical" in response.text


def test_upload_mismatch_input_lengths():
    """Test error when ground truth has different input lengths."""
    n_input_rows = 5
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    payload0 = generate_payload(n_input_rows, 10, 1, "INT64", "TRAINING")
    payload0["model_name"] = model_name
    post_test(payload0, 200, [f"{n_input_rows} datapoints"])
    ids = get_metadata_ids(model_name)
    payload1 = generate_payload(n_input_rows, 11, 1, "INT64", "TRAINING")
    payload1["model_name"] = model_name
    payload1["is_ground_truth"] = True
    payload1["request"]["inputs"][0]["execution_ids"] = ids
    response = client.post("/data/upload", json=payload1)
    assert response.status_code == 400
    assert "Found fatal mismatches" in response.text
    assert (
        "input shapes do not match. Observed inputs have length=10 while uploaded inputs have length=11"
        in response.text
    )


def test_upload_mismatch_input_and_output_types():
    """Test error when ground truth has different data types."""
    n_input_rows = 5
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    payload0 = generate_payload(n_input_rows, 10, 2, "INT64", "TRAINING")
    payload0["model_name"] = model_name
    post_test(payload0, 200, [f"{n_input_rows} datapoints"])
    ids = get_metadata_ids(model_name)
    payload1 = generate_payload(n_input_rows, 10, 2, "FP32", "TRAINING", 0, 1)
    payload1["model_name"] = model_name
    payload1["is_ground_truth"] = True
    payload1["request"]["inputs"][0]["execution_ids"] = ids
    response = client.post("/data/upload", json=payload1)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    assert response.status_code == 400
    assert "Found fatal mismatches" in response.text
    assert "Class=Long != Class=Float" in response.text or "inputs are not identical" in response.text


def test_upload_gaussian_data():
    """Test uploading realistic Gaussian data."""
    payload = {
        "model_name": "gaussian-credit-model",
        "data_tag": "TRAINING",
        "request": {
            "inputs": [
                {
                    "name": "credit_inputs",
                    "shape": [2, 4],
                    "datatype": "FP64",
                    "data": [
                        [
                            47.45380690750797,
                            478.6846214843319,
                            13.462184703540503,
                            20.764525303373535,
                        ],
                        [
                            47.468246185717554,
                            575.6911203538863,
                            10.844143722475575,
                            14.81343667761101,
                        ],
                    ],
                }
            ]
        },
        "response": {
            "model_name": "gaussian-credit-model__isvc-d79a7d395d",
            "model_version": "1",
            "outputs": [
                {
                    "name": "predict",
                    "datatype": "FP32",
                    "shape": [2, 1],
                    "data": [0.19013395683309373, 0.2754730253205645],
                }
            ],
        },
    }
    post_test(payload, 200, ["2 datapoints"])