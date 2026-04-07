import asyncio
import gzip
import itertools
import json
import os
import shutil
import tempfile
import unittest
import uuid

from fastapi.testclient import TestClient

from src.service.constants import (
    TRUSTYAI_TAG_PREFIX,
)
from src.service.data.model_data import ModelData

MODEL_ID = "example1"


def generate_payload(n_rows, n_input_cols, n_output_cols, datatype, tag, input_offset=0, output_offset=0):
    """Generate a test payload with specific dimensions and data types."""
    model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
    input_data = []
    for i in range(n_rows):
        if n_input_cols == 1:
            val = i + input_offset
            # Convert to bool (0 or 1) if datatype is BOOL
            if datatype == "BOOL":
                val = val % 2
            input_data.append(val)
        else:
            row = [
                (i + j + input_offset) % 2 if datatype == "BOOL" else (i + j + input_offset)
                for j in range(n_input_cols)
            ]
            input_data.append(row)
    output_data = []
    for i in range(n_rows):
        if n_output_cols == 1:
            val = i * 2 + output_offset
            # Convert to bool (0 or 1) if datatype is BOOL
            if datatype == "BOOL":
                val = val % 2
            output_data.append(val)
        else:
            row = [
                (i * 2 + j + output_offset) % 2 if datatype == "BOOL" else (i * 2 + j + output_offset)
                for j in range(n_output_cols)
            ]
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
        if datatype == "BOOL":
            row = [(row_idx + col_idx * 10) % 2 for col_idx in range(n_input_cols)]
        else:
            row = [row_idx + col_idx * 10 for col_idx in range(n_input_cols)]
        input_data.append(row)
    output_data = []
    for row_idx in range(n_rows):
        if datatype == "BOOL":
            row = [(row_idx * 2 + col_idx) % 2 for col_idx in range(n_output_cols)]
        else:
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
    if datatype == "BOOL":
        input_data_1 = [[(row_idx + col_idx * 10) % 2 for col_idx in range(n_input_cols)] for row_idx in range(n_rows)]
        mismatched_rows = n_rows - 1 if n_rows > 1 else 1
        input_data_2 = [
            [(row_idx + col_idx * 20) % 2 for col_idx in range(n_input_cols)] for row_idx in range(mismatched_rows)
        ]
        output_data = [[(row_idx * 2 + col_idx) % 2 for col_idx in range(n_output_cols)] for row_idx in range(n_rows)]
    else:
        input_data_1 = [[row_idx + col_idx * 10 for col_idx in range(n_input_cols)] for row_idx in range(n_rows)]
        mismatched_rows = n_rows - 1 if n_rows > 1 else 1
        input_data_2 = [
            [row_idx + col_idx * 20 for col_idx in range(n_input_cols)] for row_idx in range(mismatched_rows)
        ]
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


def count_rows_with_tag(model_name, tag):
    """Count rows with a specific tag in metadata."""
    metadata_df = asyncio.run(ModelData(model_name).get_metadata_as_df())
    return metadata_df["tags"].apply(lambda tags: tag in tags).sum()


def get_metadata_ids(model_name):
    """Count rows with a specific tag in metadata."""
    metadata_df = asyncio.run(ModelData(model_name).get_metadata_as_df())
    return metadata_df["id"].tolist()


class TestUploadEndpointPVC(unittest.TestCase):
    def setUp(self):
        self.TEMP_DIR = tempfile.mkdtemp()
        os.environ["STORAGE_DATA_FOLDER"] = self.TEMP_DIR

        # Force reload of the global storage interface to use the new temp dir
        from src.service.data import storage

        storage.get_global_storage_interface(force_reload=True)

        # Re-create the FastAPI app to ensure it uses the new storage interface
        from importlib import reload

        import src.main

        reload(src.main)
        from src.main import app

        self.client = TestClient(app)

    def tearDown(self):
        if os.path.exists(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)

    def post_test(self, payload, expected_status_code, check_msgs):
        """Post a payload and check the response."""
        response = self.client.post("/data/upload", json=payload)
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
            print("==================")

        self.assertEqual(response.status_code, expected_status_code)
        return response

    # data upload tests
    def test_upload_data(self):
        n_input_rows_options = [1, 5, 250]
        n_input_cols_options = [1, 4]
        n_output_cols_options = [1, 2]
        datatype_options = ["INT64", "INT32", "FP32", "FP64", "BOOL"]

        for idx, (n_input_rows, n_input_cols, n_output_cols, datatype) in enumerate(
            itertools.product(n_input_rows_options, n_input_cols_options, n_output_cols_options, datatype_options)
        ):
            with self.subTest(
                f"subtest-{idx}",
                n_input_rows=n_input_rows,
                n_input_cols=n_input_cols,
                n_output_cols=n_output_cols,
                datatype=datatype,
            ):
                """Test uploading data with various dimensions and datatypes."""
                data_tag = "TRAINING"
                payload = generate_payload(n_input_rows, n_input_cols, n_output_cols, datatype, data_tag)
                response = self.post_test(payload, 200, [f"{n_input_rows} datapoints"])

                inputs, outputs, metadata = asyncio.run(ModelData(payload["model_name"]).data())

                self.assertEqual(response.status_code, 200)
                self.assertIn(str(n_input_rows), response.text)
                self.assertIsNotNone(inputs, "Input data not found in storage")
                self.assertIsNotNone(outputs, "Output data not found in storage")

                self.assertEqual(len(inputs), n_input_rows, "Incorrect number of input rows")
                self.assertEqual(len(outputs), n_input_rows, "Incorrect number of output rows")

                tag_count = count_rows_with_tag(payload["model_name"], data_tag)
                self.assertEqual(tag_count, n_input_rows, "Not all rows have the correct tag")

    def test_upload_multi_input_data(self):
        """Test uploading data with multiple input tensors."""
        n_rows_options = [1, 3, 5, 250]
        n_input_cols_options = [2, 6]
        n_output_cols_options = [4]
        datatype_options = ["INT64", "INT32", "FP32", "FP64", "BOOL"]

        for n_rows, n_input_cols, n_output_cols, datatype in itertools.product(
            n_rows_options, n_input_cols_options, n_output_cols_options, datatype_options
        ):
            with self.subTest(
                n_rows=n_rows,
                n_input_cols=n_input_cols,
                n_output_cols=n_output_cols,
                datatype=datatype,
            ):
                # Arrange
                data_tag = "TRAINING"
                payload = generate_multi_input_payload(n_rows, n_input_cols, n_output_cols, datatype, data_tag)

                # Act
                self.post_test(payload, 200, [f"{n_rows} datapoints"])

                model_data = ModelData(payload["model_name"])
                inputs, outputs, metadata = asyncio.run(model_data.data())
                input_column_names, output_column_names, metadata_column_names = asyncio.run(
                    model_data.original_column_names()
                )

                # Assert
                self.assertIsNotNone(inputs, "Input data not found in storage")
                self.assertIsNotNone(outputs, "Output data not found in storage")
                self.assertEqual(len(inputs), n_rows, "Incorrect number of input rows")
                self.assertEqual(len(outputs), n_rows, "Incorrect number of output rows")
                self.assertEqual(len(input_column_names), n_input_cols, "Incorrect number of input columns")
                self.assertEqual(len(output_column_names), n_output_cols, "Incorrect number of output columns")
                self.assertGreaterEqual(len(input_column_names), 2, "Should have at least 2 input column names")
                tag_count = count_rows_with_tag(payload["model_name"], data_tag)
                self.assertEqual(tag_count, n_rows, "Not all rows have the correct tag")

    def test_upload_multi_input_data_no_unique_name(self):
        """Test error case for non-unique tensor names."""
        payload = generate_mismatched_shape_no_unique_name_multi_input_payload(250, 4, 3, "FP64", "TRAINING")
        response = self.client.post("/data/upload", json=payload)
        self.assertEqual(response.status_code, 400)
        print(response.text)
        self.assertIn("input shapes were mismatched", response.text)
        self.assertIn("[250, 4]", response.text)

    def test_upload_mismatched_row_counts(self):
        """Test error case for mismatched input/output row counts."""
        model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"
        n_input_rows = 10
        n_output_rows = 9

        payload = {
            "model_name": model_name,
            "is_ground_truth": False,
            "request": {
                "inputs": [
                    {
                        "name": "input",
                        "shape": [n_input_rows],
                        "datatype": "FP32",
                        "data": [float(i) for i in range(n_input_rows)],
                    }
                ],
            },
            "response": {
                "outputs": [
                    {
                        "name": "output",
                        "shape": [n_output_rows],
                        "datatype": "FP32",
                        "data": [float(i) for i in range(n_output_rows)],
                    }
                ],
            },
        }

        response = self.client.post("/data/upload", json=payload)
        self.assertEqual(response.status_code, 400)
        response_text = response.text
        self.assertIn("Could not reconcile", response_text)
        self.assertIn("number of", response_text.lower())

    def test_upload_multiple_tagging(self):
        """Test uploading data with multiple tags."""
        n_payload1 = 50
        n_payload2 = 51
        tag1 = "TRAINING"
        tag2 = "NOT TRAINING "
        model_name = f"{MODEL_ID}_{uuid.uuid4().hex[:8]}"

        payload1 = generate_payload(n_payload1, 10, 1, "INT64", tag1)
        payload1["model_name"] = model_name
        self.post_test(payload1, 200, [f"{n_payload1} datapoints"])

        payload2 = generate_payload(n_payload2, 10, 1, "INT64", tag2)
        payload2["model_name"] = model_name
        self.post_test(payload2, 200, [f"{n_payload2} datapoints"])

        tag1_count = count_rows_with_tag(model_name, tag1)
        tag2_count = count_rows_with_tag(model_name, tag2)

        self.assertEqual(tag1_count, n_payload1, f"Expected {n_payload1} rows with tag {tag1}")
        self.assertEqual(tag2_count, n_payload2, f"Expected {n_payload2} rows with tag {tag2}")

        input_rows, _, _ = asyncio.run(ModelData(payload1["model_name"]).row_counts())
        self.assertEqual(input_rows, n_payload1 + n_payload2, "Incorrect total number of rows")

    def test_upload_tag_that_uses_protected_name(self):
        """Test error when using a protected tag name."""
        invalid_tag = f"{TRUSTYAI_TAG_PREFIX}_something"
        payload = generate_payload(5, 10, 1, "INT64", invalid_tag)
        response = self.post_test(payload, 400, ["reserved for internal TrustyAI use only"])
        expected_msg = f"The tag prefix '{TRUSTYAI_TAG_PREFIX}' is reserved for internal TrustyAI use only. Provided tag '{invalid_tag}' violates this restriction."
        self.assertIn(expected_msg, response.text)

    def test_upload_gaussian_data(self):
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
                        "data": [[0.19013395683309373], [0.2754730253205645]],
                    }
                ],
            },
        }
        self.post_test(payload, 200, ["2 datapoints"])

    def test_upload_gzip_compressed_data(self):
        """
        Test uploading gzip-compressed data.

        This simulates the behavior of KServe agent in RawDeployment mode,
        which automatically gzip-compresses CloudEvent payloads using Go's
        http.Transport default compression.

        This test verifies that the GzipRequestMiddleware correctly
        decompresses the request body before it reaches the endpoint handler.
        """
        # Generate a test payload
        n_rows = 5
        payload = generate_payload(n_rows, 2, 1, "INT64", "GZIP_TEST")

        # Serialize payload to JSON
        json_payload = json.dumps(payload)

        # Compress with gzip
        compressed_payload = gzip.compress(json_payload.encode("utf-8"))

        # Send gzip-compressed request
        response = self.client.post(
            "/data/upload",
            content=compressed_payload,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

        # Debug output if test fails
        if response.status_code != 200:
            print("\n=== DEBUG INFO ===")
            print(f"Status: {response.status_code}")
            print(f"Response text: {response.text}")
            print(f"Response headers: {dict(response.headers)}")
            print("==================")

        # Verify successful upload
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json["status"], "success")
        self.assertIn(f"{n_rows} datapoints", response_json["message"])

        # Verify data was stored correctly
        inputs, outputs, _metadata = asyncio.run(ModelData(payload["model_name"]).data())
        self.assertIsNotNone(inputs, "Input data not found in storage")
        self.assertIsNotNone(outputs, "Output data not found in storage")
        self.assertEqual(len(inputs), n_rows, "Incorrect number of input rows")
        self.assertEqual(len(outputs), n_rows, "Incorrect number of output rows")

        # Verify tag was applied
        tag_count = count_rows_with_tag(payload["model_name"], "GZIP_TEST")
        self.assertEqual(tag_count, n_rows, "Not all rows have the correct tag")

    def test_upload_malformed_gzip_compressed_data(self):
        """
        Test that malformed gzip-compressed payloads return a client error (400)
        rather than a server error (500).

        This verifies that the GzipRequestMiddleware properly handles invalid
        gzip data and returns a clear error message to the client.
        """
        # Create invalid gzip payload (not actually gzip-compressed)
        invalid_gzip_payload = b"not-a-valid-gzip-stream"

        # Send request with Content-Encoding: gzip but invalid gzip data
        response = self.client.post(
            "/data/upload",
            content=invalid_gzip_payload,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

        # Verify it returns 400 Bad Request (not 500 Internal Server Error)
        self.assertEqual(response.status_code, 400)

        # Verify error message mentions decompression failure
        self.assertIn("could not be decompressed", response.text.lower())

    def test_gzip_middleware_not_applied_to_other_endpoints(self):
        """
        Test that the gzip middleware only applies to /data/upload endpoint.

        This verifies path scoping to avoid unexpected behavior for other endpoints.
        """
        # Try sending gzip-encoded data to a different endpoint (root)
        # The middleware should NOT decompress it
        response = self.client.get(
            "/",
            headers={
                "Content-Encoding": "gzip",
            },
        )

        # Should succeed because middleware doesn't apply to root endpoint
        self.assertEqual(response.status_code, 200)

    def test_upload_gzip_size_limit_exceeded(self):
        """
        Test that extremely large decompressed payloads are rejected with HTTP 413.

        This verifies decompression bomb protection.
        """
        # Create a decompression bomb: highly repetitive data that compresses well
        # A string of 17MB of zeros will compress to a few KB but exceed the 16MB limit
        repetitive_data = "0" * (17 * 1024 * 1024)  # 17MB of zeros

        # Compress it - will be very small due to repetition
        compressed_payload = gzip.compress(repetitive_data.encode("utf-8"))

        # Verify compression is effective (compressed should be < 1% of original)
        compression_ratio = len(compressed_payload) / len(repetitive_data)
        self.assertLess(compression_ratio, 0.01, "Compression not effective enough for test")

        response = self.client.post(
            "/data/upload",
            content=compressed_payload,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

        # Should return 413 Payload Too Large
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds limit", response.text.lower())

    def test_upload_gzip_with_multiple_encodings(self):
        """
        Test that gzip middleware handles requests with multiple Content-Encoding values.

        Since the middleware only supports gzip decompression (not chained decompression),
        it removes the entire Content-Encoding header after decompressing. This prevents
        the downstream app from attempting to decompress using encodings we don't support.

        The request body is only gzip-compressed (not actually gzip+brotli stacked),
        while the header claims "gzip, br". After middleware processing, the body is
        decompressed and the Content-Encoding header is removed entirely.
        """
        # Generate test payload
        payload = generate_payload(5, 2, 1, "INT64", "MULTI_ENCODING_TEST")
        json_payload = json.dumps(payload)
        compressed_payload = gzip.compress(json_payload.encode("utf-8"))

        # Send with multiple encodings (gzip, br)
        # Note: We're only actually gzip-compressing, testing header handling
        response = self.client.post(
            "/data/upload",
            content=compressed_payload,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip, br",  # Multiple encodings (only gzip actually applied)
            },
        )

        # Should succeed - body is decompressed, Content-Encoding header removed
        self.assertEqual(response.status_code, 200)

    def test_gzip_middleware_skip_conditions(self):
        """Test middleware skips requests without gzip encoding or wrong content-type."""
        payload = generate_payload(3, 2, 1, "INT64", "SKIP_TEST")
        json_payload = json.dumps(payload).encode("utf-8")

        # No Content-Encoding - should skip middleware
        response = self.client.post("/data/upload", content=json_payload, headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 200)

        # Non-gzip encoding - should skip middleware
        response = self.client.post("/data/upload", content=json_payload, headers={"Content-Type": "application/json", "Content-Encoding": "br"})
        self.assertEqual(response.status_code, 200)

        # Missing Content-Type with gzip encoding - should skip middleware
        response = self.client.post("/data/upload", content=json_payload, headers={"Content-Encoding": "gzip"})
        self.assertIn(response.status_code, [200, 400, 422])

        # Wrong content-type - should skip middleware (won't crash)
        response = self.client.post("/data/upload", content=json_payload, headers={"Content-Type": "text/plain", "Content-Encoding": "gzip"})
        self.assertLess(response.status_code, 500)

    def test_upload_empty_gzip_body(self):
        """Test that empty gzip body returns 400 Bad Request."""
        response = self.client.post(
            "/data/upload",
            content=b"",
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.text.lower())

    def test_gzip_middleware_init_validation(self):
        """Test that GzipRequestMiddleware validates max_size parameter."""
        from src.middleware.gzip_middleware import GzipRequestMiddleware

        with self.assertRaises(ValueError):
            GzipRequestMiddleware(app=None, max_size=0)


if __name__ == "__main__":
    unittest.main()
