import uuid
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class DataframeGenerators:
    """Python equivalent of Java DataframeGenerators"""

    @staticmethod
    def generate_random_dataframe(observations: int, feature_diversity: int = 100) -> pd.DataFrame:
        random = np.random.RandomState(0)
        data = {
            "age": [],
            "gender": [],
            "race": [],
            "income": [],
            "trustyai.ID": [],
            "trustyai.MODEL_ID": [],
            "trustyai.TIMESTAMP": [],
            "trustyai.TAG": [],
            "trustyai.INDEX": [],
        }
        for i in range(observations):
            data["age"].append(i % feature_diversity)
            data["gender"].append(1 if random.choice([True, False]) else 0)
            data["race"].append(1 if random.choice([True, False]) else 0)
            data["income"].append(1 if random.choice([True, False]) else 0)
            data["trustyai.ID"].append(str(uuid.uuid4()))
            data["trustyai.MODEL_ID"].append("example1")
            data["trustyai.TIMESTAMP"].append((datetime.now() - timedelta(seconds=i)).isoformat())
            data["trustyai.TAG"].append("")
            data["trustyai.INDEX"].append(i)
        return pd.DataFrame(data)

    @staticmethod
    def generate_random_text_dataframe(observations: int, seed: int = 0) -> pd.DataFrame:
        if seed < 0:
            random = np.random.RandomState(0)
        else:
            random = np.random.RandomState(seed)
        makes = ["Ford", "Chevy", "Dodge", "GMC", "Buick"]
        colors = ["Red", "Blue", "White", "Black", "Purple", "Green", "Yellow"]
        data = {
            "year": [],
            "make": [],
            "color": [],
            "value": [],
            "trustyai.ID": [],
            "trustyai.MODEL_ID": [],
            "trustyai.TIMESTAMP": [],
            "trustyai.TAG": [],
            "trustyai.INDEX": [],
        }
        for i in range(observations):
            data["year"].append(1970 + i % 50)
            data["make"].append(makes[i % len(makes)])
            data["color"].append(colors[i % len(colors)])
            data["value"].append(random.random() * 50)
            data["trustyai.ID"].append(str(uuid.uuid4()))
            data["trustyai.MODEL_ID"].append("example1")
            data["trustyai.TIMESTAMP"].append((datetime.now() - timedelta(seconds=i)).isoformat())
            data["trustyai.TAG"].append("")
            data["trustyai.INDEX"].append(i)
        return pd.DataFrame(data)


# Mock storage for testing
class MockStorage:
    def __init__(self):
        self.data = {}

    async def read_data(self, dataset_name: str):
        if dataset_name.endswith("_outputs"):
            model_id = dataset_name.replace("_outputs", "")
            if model_id not in self.data:
                raise Exception(f"Model {model_id} not found")
            output_data = self.data[model_id].get("output")
            output_cols = self.data[model_id].get("output_cols", [])
            return output_data, output_cols
        elif dataset_name.endswith("_metadata"):
            model_id = dataset_name.replace("_metadata", "")
            if model_id not in self.data:
                raise Exception(f"Model {model_id} not found")
            metadata_data = self.data[model_id].get("metadata")
            metadata_cols = ["ID", "MODEL_ID", "TIMESTAMP", "TAG", "INDEX"]
            return metadata_data, metadata_cols
        elif dataset_name.endswith("_inputs"):
            model_id = dataset_name.replace("_inputs", "")
            if model_id not in self.data:
                raise Exception(f"Model {model_id} not found")
            input_data = self.data[model_id].get("input")
            input_cols = self.data[model_id].get("input_cols", [])
            return input_data, input_cols
        else:
            raise Exception(f"Unknown dataset: {dataset_name}")

    def save_dataframe(self, df: pd.DataFrame, model_id: str):
        input_cols = [col for col in df.columns if not col.startswith("trustyai.") and col not in ["income", "value"]]
        output_cols = [col for col in df.columns if col in ["income", "value"]]
        metadata_cols = [col for col in df.columns if col.startswith("trustyai.")]
        input_data = df[input_cols].values if input_cols else np.array([])
        output_data = df[output_cols].values if output_cols else np.array([])
        metadata_data_cols = ["ID", "MODEL_ID", "TIMESTAMP", "TAG", "INDEX"]
        metadata_values = []
        for _, row in df.iterrows():
            row_data = []
            for col in metadata_data_cols:
                trusty_col = f"trustyai.{col}"
                if trusty_col in df.columns:
                    value = row[trusty_col]
                    if col == "INDEX":
                        row_data.append(int(value))
                    else:
                        row_data.append(str(value))
                else:
                    row_data.append("" if col != "INDEX" else 0)
            metadata_values.append(row_data)
        metadata_data = np.array(metadata_values, dtype=object)
        self.data[model_id] = {
            "dataframe": df,
            "input": input_data,
            "input_cols": input_cols,
            "output": output_data,
            "output_cols": output_cols,
            "metadata": metadata_data,
        }

    def reset(self):
        self.data.clear()


mock_storage = MockStorage()


@pytest.fixture(autouse=True)
def setup_storage():
    """Setup mock storage for all tests"""
    with patch("src.service.utils.download.get_storage_interface", return_value=mock_storage):
        yield


@pytest.fixture(autouse=True)
def reset_storage():
    """Reset storage before each test"""
    mock_storage.reset()
    yield


# Test constants
MODEL_ID = "example1"


def test_download_data():
    """equivalent of Java downloadData() test"""
    dataframe = DataframeGenerators.generate_random_dataframe(1000)
    mock_storage.save_dataframe(dataframe, MODEL_ID)

    payload = {
        "modelId": MODEL_ID,
        "matchAll": [
            {"columnName": "gender", "operation": "EQUALS", "values": [0]},
            {"columnName": "race", "operation": "EQUALS", "values": [0]},
            {"columnName": "income", "operation": "EQUALS", "values": [0]},
        ],
        "matchAny": [
            {"columnName": "age", "operation": "BETWEEN", "values": [5, 10]},
            {"columnName": "age", "operation": "BETWEEN", "values": [50, 70]},
        ],
        "matchNone": [{"columnName": "age", "operation": "BETWEEN", "values": [55, 65]}],
    }
    response = client.post("/data/download", json=payload)
    assert response.status_code == 200
    result = response.json()
    result_df = pd.read_csv(StringIO(result["dataCSV"]))
    assert len(result_df[(result_df["age"] > 55) & (result_df["age"] < 65)]) == 0
    assert len(result_df[result_df["gender"] == 1]) == 0
    assert len(result_df[result_df["race"] == 1]) == 0
    assert len(result_df[result_df["income"] == 1]) == 0
    assert len(result_df[(result_df["age"] >= 10) & (result_df["age"] < 50)]) == 0
    assert len(result_df[result_df["age"] > 70]) == 0


def test_download_text_data():
    """equivalent of Java downloadTextData() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1000)
    mock_storage.save_dataframe(dataframe, MODEL_ID)

    payload = {
        "modelId": MODEL_ID,
        "matchAll": [
            {
                "columnName": "make",
                "operation": "EQUALS",
                "values": ["Chevy", "Ford", "Dodge"],
            },
            {
                "columnName": "year",
                "operation": "BETWEEN",
                "values": [1990, 2050],
            },
        ],
    }

    response = client.post("/data/download", json=payload)
    assert response.status_code == 200
    result = response.json()
    result_df = pd.read_csv(StringIO(result["dataCSV"]))
    assert len(result_df[result_df["year"] < 1990]) == 0
    assert len(result_df[result_df["make"] == "GMC"]) == 0
    assert len(result_df[result_df["make"] == "Buick"]) == 0


def test_download_text_data_between_error():
    """equivalent of Java downloadTextDataBetweenError() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1000)
    mock_storage.save_dataframe(dataframe, MODEL_ID)
    payload = {
        "modelId": MODEL_ID,
        "matchAll": [
            {
                "columnName": "make",
                "operation": "BETWEEN",
                "values": ["Chevy", "Ford", "Dodge"],
            }
        ],
    }
    response = client.post("/data/download", json=payload)
    assert response.status_code == 400
    assert (
        "BETWEEN operation must contain exactly two values, describing the lower and upper bounds of the desired range. Received 3 values"
        in response.text
    )
    assert (
        "BETWEEN operation must only contain numbers, describing the lower and upper bounds of the desired range. Received non-numeric values"
        in response.text
    )


def test_download_text_data_invalid_column_error():
    """equivalent of Java downloadTextDataInvalidColumnError() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1000)
    mock_storage.save_dataframe(dataframe, MODEL_ID)
    payload = {
        "modelId": MODEL_ID,
        "matchAll": [
            {
                "columnName": "mak123e",
                "operation": "EQUALS",
                "values": ["Chevy", "Ford"],
            }
        ],
    }

    response = client.post("/data/download", json=payload)
    assert response.status_code == 400
    assert "No feature or output found with name=" in response.text


def test_download_text_data_invalid_operation_error():
    """equivalent of Java downloadTextDataInvalidOperationError() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1000)
    mock_storage.save_dataframe(dataframe, MODEL_ID)
    payload = {
        "modelId": MODEL_ID,
        "matchAll": [
            {
                "columnName": "mak123e",
                "operation": "DOESNOTEXIST",
                "values": ["Chevy", "Ford"],
            }
        ],
    }
    response = client.post("/data/download", json=payload)
    assert response.status_code == 400
    assert "RowMatch operation must be one of [BETWEEN, EQUALS]" in response.text


def test_download_text_data_internal_column():
    """equivalent of Java downloadTextDataInternalColumn() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1000)
    dataframe.loc[0:499, "trustyai.TAG"] = "TRAINING"
    mock_storage.save_dataframe(dataframe, MODEL_ID)
    payload = {
        "modelId": MODEL_ID,
        "matchAll": [
            {
                "columnName": "trustyai.TAG",
                "operation": "EQUALS",
                "values": ["TRAINING"],
            }
        ],
    }
    response = client.post("/data/download", json=payload)
    assert response.status_code == 200
    result = response.json()
    result_df = pd.read_csv(StringIO(result["dataCSV"]))
    assert len(result_df) == 500


def test_download_text_data_internal_column_index():
    """equivalent of Java downloadTextDataInternalColumnIndex() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1000)
    mock_storage.save_dataframe(dataframe, MODEL_ID)
    expected_rows = dataframe.iloc[0:10].copy()
    payload = {
        "modelId": MODEL_ID,
        "matchAll": [
            {
                "columnName": "trustyai.INDEX",
                "operation": "BETWEEN",
                "values": [0, 10],
            }
        ],
    }
    response = client.post("/data/download", json=payload)
    print(f"Response status: {response.status_code}")
    print(f"Response text: {response.text}")
    assert response.status_code == 200
    result = response.json()
    result_df = pd.read_csv(StringIO(result["dataCSV"]))
    assert len(result_df) == 10
    input_cols = ["year", "make", "color"]
    for i in range(10):
        for col in input_cols:
            assert result_df.iloc[i][col] == expected_rows.iloc[i][col], f"Row {i}, column {col} mismatch"


def test_download_text_data_internal_column_timestamp():
    """equivalent of Java downloadTextDataInternalColumnTimestamp() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1, -1)
    base_time = datetime.now()
    for i in range(100):
        new_row = DataframeGenerators.generate_random_text_dataframe(1, i)
        # Use milliseconds to simulate Thread.sleep(1) and ensure ascending order
        timestamp = (base_time + timedelta(milliseconds=i + 1)).isoformat()
        # Fix this line - change to UPPERCASE
        new_row["trustyai.TIMESTAMP"] = [timestamp]
        dataframe = pd.concat([dataframe, new_row], ignore_index=True)
    mock_storage.save_dataframe(dataframe, MODEL_ID)
    extract_idx = 50
    n_to_get = 10
    expected_rows = dataframe.iloc[extract_idx : extract_idx + n_to_get].copy()
    start_time = dataframe.iloc[extract_idx]["trustyai.TIMESTAMP"]
    end_time = dataframe.iloc[extract_idx + n_to_get]["trustyai.TIMESTAMP"]
    payload = {
        "modelId": MODEL_ID,
        "matchAny": [
            {
                "columnName": "trustyai.TIMESTAMP",
                "operation": "BETWEEN",
                "values": [start_time, end_time],
            }
        ],
    }
    response = client.post("/data/download", json=payload)
    assert response.status_code == 200
    result = response.json()
    result_df = pd.read_csv(StringIO(result["dataCSV"]))
    assert len(result_df) == 10
    input_cols = ["year", "make", "color"]
    for i in range(10):
        for col in input_cols:
            assert result_df.iloc[i][col] == expected_rows.iloc[i][col], f"Row {i}, column {col} mismatch"


def test_download_text_data_internal_column_timestamp_unparseable():
    """equivalent of Java downloadTextDataInternalColumnTimestampUnparseable() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1000)
    mock_storage.save_dataframe(dataframe, MODEL_ID)
    payload = {
        "modelId": MODEL_ID,
        "matchAny": [
            {
                "columnName": "trustyai.TIMESTAMP",
                "operation": "BETWEEN",
                "values": ["not a timestamp", "also not a timestamp"],
            }
        ],
    }
    response = client.post("/data/download", json=payload)
    assert response.status_code == 400
    assert "unparseable as an ISO_LOCAL_DATE_TIME" in response.text


def test_download_text_data_null_request():
    """equivalent of Java downloadTextDataNullRequest() test"""
    dataframe = DataframeGenerators.generate_random_text_dataframe(1000)
    mock_storage.save_dataframe(dataframe, MODEL_ID)
    payload = {"modelId": MODEL_ID}
    response = client.post("/data/download", json=payload)
    assert response.status_code == 200
    result = response.json()
    result_df = pd.read_csv(StringIO(result["dataCSV"]))
    assert len(result_df) == 1000