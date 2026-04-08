"""
Tests for ModelMesh payload reconciliation.
"""

import asyncio
import os
import tempfile
import unittest
import uuid
from unittest import mock

import h5py
import numpy as np
import pandas as pd

from src.service.data.modelmesh_parser import ModelMeshPayloadParser, PartialPayload
from src.service.data.storage.pvc import PVCStorage
from tests.service.data.test_utils import ModelMeshTestData


class TestPayloadReconciliation(unittest.TestCase):
    """
    Test class for ModelMesh payload reconciliation.
    """

    def setUp(self):
        """Set up the test environment."""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage = PVCStorage(self.temp_dir.name)

        self.model_name = "test-model"
        self.request_id = str(uuid.uuid4())

        input_specs = [("input", 5, 10, "INT32", 0, None)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        self.input_payload_dict, self.output_payload_dict, _, _ = ModelMeshTestData.generate_test_payloads(
            self.model_name, input_specs, output_specs
        )

        self.input_payload = PartialPayload(**self.input_payload_dict)
        self.output_payload = PartialPayload(**self.output_payload_dict)

    def tearDown(self):
        """Clean up after tests."""
        print(asyncio.run(self.storage.list_all_datasets()))
        self.temp_dir.cleanup()

    async def _test_persist_input_payload(self):
        """Test persisting an input payload."""
        await self.storage.persist_partial_payload(self.input_payload, payload_id=self.request_id, is_input=True)

        retrieved_payload = await self.storage.get_partial_payload(self.request_id, is_input=True, is_modelmesh=True)

        self.assertIsNotNone(retrieved_payload)
        self.assertEqual(retrieved_payload.data, self.input_payload.data)

        output_payload = await self.storage.get_partial_payload(self.request_id, is_input=False, is_modelmesh=True)
        self.assertIsNone(output_payload)

    async def _test_persist_output_payload(self):
        """Test persisting an output payload."""
        await self.storage.persist_partial_payload(self.output_payload, payload_id=self.request_id, is_input=False)

        retrieved_payload = await self.storage.get_partial_payload(self.request_id, is_input=False, is_modelmesh=True)

        self.assertIsNotNone(retrieved_payload)
        self.assertEqual(retrieved_payload.data, self.output_payload.data)

        input_payload = await self.storage.get_partial_payload(self.request_id, is_input=True, is_modelmesh=True)
        self.assertIsNone(input_payload)

    async def _test_full_reconciliation(self):
        """Test the full payload reconciliation process."""
        await self.storage.persist_partial_payload(self.input_payload, self.request_id, is_input=True)
        await self.storage.persist_partial_payload(self.output_payload, self.request_id, is_input=False)

        input_payload = await self.storage.get_partial_payload(self.request_id, is_input=True, is_modelmesh=True)
        output_payload = await self.storage.get_partial_payload(self.request_id, is_input=False, is_modelmesh=True)

        self.assertIsNotNone(input_payload)
        self.assertIsNotNone(output_payload)

        df = ModelMeshPayloadParser.payloads_to_dataframe(
            input_payload, output_payload, self.request_id, self.model_name
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("input", df.columns)
        self.assertIn("output_output", df.columns)
        self.assertEqual(len(df), 5)  # Based on our test data with 5 rows

        self.assertIn("id", df.columns)
        self.assertEqual(df["id"].iloc[0], self.request_id)

        self.assertIn("model_id", df.columns)
        self.assertEqual(df["model_id"].iloc[0], self.model_name)

        # Clean up
        await self.storage.delete_partial_payload(self.request_id, is_input=True)
        await self.storage.delete_partial_payload(self.request_id, is_input=False)

        input_payload = await self.storage.get_partial_payload(self.request_id, is_input=True, is_modelmesh=True)
        output_payload = await self.storage.get_partial_payload(self.request_id, is_input=False, is_modelmesh=True)

        self.assertIsNone(input_payload)
        self.assertIsNone(output_payload)

    async def _test_reconciliation_with_real_data(self):
        """Test reconciliation with sample b64 encoded data from files."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_data_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "data")

        with open(os.path.join(test_data_dir, "input-sample.b64"), "r") as f:
            sample_input_data = f.read().strip()

        with open(os.path.join(test_data_dir, "output-sample.b64"), "r") as f:
            sample_output_data = f.read().strip()

        input_payload = PartialPayload(data=sample_input_data)
        output_payload = PartialPayload(data=sample_output_data)

        request_id = str(uuid.uuid4())
        model_id = "sample-model"

        await self.storage.persist_partial_payload(input_payload, request_id, is_input=True)
        await self.storage.persist_partial_payload(output_payload, request_id, is_input=False)

        stored_input = await self.storage.get_partial_payload(request_id, is_input=True, is_modelmesh=True)
        stored_output = await self.storage.get_partial_payload(request_id, is_input=False, is_modelmesh=True)

        self.assertIsNotNone(stored_input)
        self.assertIsNotNone(stored_output)
        self.assertEqual(stored_input.data, sample_input_data)
        self.assertEqual(stored_output.data, sample_output_data)

        with mock.patch.object(ModelMeshPayloadParser, "payloads_to_dataframe") as mock_to_df:
            sample_df = pd.DataFrame({
                "input_feature": [1, 2, 3],
                "output_output_feature": [4, 5, 6],
                "id": [request_id] * 3,
                "model_id": [model_id] * 3,
                "synthetic": [False] * 3,
            })
            mock_to_df.return_value = sample_df

            df = ModelMeshPayloadParser.payloads_to_dataframe(stored_input, stored_output, request_id, model_id)

            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 3)

            mock_to_df.assert_called_once_with(stored_input, stored_output, request_id, model_id)

        # Clean up
        await self.storage.delete_partial_payload(request_id, is_input=True)
        await self.storage.delete_partial_payload(request_id, is_input=False)

        self.assertIsNone(await self.storage.get_partial_payload(request_id, is_input=True, is_modelmesh=True))
        self.assertIsNone(await self.storage.get_partial_payload(request_id, is_input=False, is_modelmesh=True))

    async def _test_corrupted_payload_handling(self):
        """Test error handling for corrupted or invalid payloads."""
        request_id = str(uuid.uuid4())

        # Test 1: Corrupted payload data (invalid base64)
        corrupted_payload = PartialPayload(data="!!!INVALID_BASE64_DATA!!!")
        await self.storage.persist_partial_payload(corrupted_payload, request_id, is_input=True)

        retrieved = await self.storage.get_partial_payload(request_id, is_input=True, is_modelmesh=True)
        # Should still retrieve the corrupted payload (storage doesn't validate)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.data, corrupted_payload.data)

        # Test 2: Attempt to parse corrupted payload should handle gracefully
        # (This is where the consumer endpoint would catch parsing errors)
        with self.assertRaises(Exception):
            # Attempting to parse invalid base64 should raise an error
            ModelMeshPayloadParser.parse_input_payload(corrupted_payload)

        # Clean up
        await self.storage.delete_partial_payload(request_id, is_input=True)

        # Test 3: Missing payload (already deleted)
        missing_payload = await self.storage.get_partial_payload(request_id, is_input=True, is_modelmesh=True)
        self.assertIsNone(missing_payload)

    async def _test_void_type_length_exceeds_limit(self):
        """Test that rows exceeding MAX_VOID_TYPE_LENGTH raise ValueError."""
        import random

        from src.service.data.storage.pvc import MAX_VOID_TYPE_LENGTH

        # Use unique dataset name to avoid conflicts
        dataset_name = f"dataset_oversized_void_type_{uuid.uuid4().hex[:8]}"

        # Create a payload that will exceed MAX_VOID_TYPE_LENGTH when serialized with JSON+gzip
        # Use random data that won't compress well (repetitive data compresses to ~50 bytes)
        # Use 2x the limit to ensure it exceeds MAX_VOID_TYPE_LENGTH even after compression
        random.seed(42)
        random_string = "".join(chr(random.randint(32, 126)) for _ in range(MAX_VOID_TYPE_LENGTH * 2))
        mixed_row = [
            random_string,  # large non-compressible payload
            {"key": "value"},  # non-primitive object
            123,  # numeric value
        ]
        dataset = np.array([mixed_row], dtype=object)
        column_names = ["col_str", "col_obj", "col_int"]

        with self.assertRaises(ValueError) as context:
            await self.storage.write_data(dataset_name, dataset, column_names)

        error_msg = str(context.exception)
        self.assertIn("exceeds maximum allowed size", error_msg)
        self.assertIn(str(MAX_VOID_TYPE_LENGTH), error_msg)

    async def _test_void_type_length_within_limit(self):
        """Test that rows under MAX_VOID_TYPE_LENGTH are written successfully."""
        from src.service.data.storage.pvc import MAX_VOID_TYPE_LENGTH

        # Use unique dataset name to avoid conflicts
        dataset_name = f"dataset_under_void_type_limit_{uuid.uuid4().hex[:8]}"

        # Create a payload that is just under the limit
        # Account for pickle overhead by using a smaller string
        allowed_string = "y" * (MAX_VOID_TYPE_LENGTH - 200)
        mixed_row = [
            allowed_string,
            {"key": "value"},
            456,
        ]
        original_dataset = np.array([mixed_row], dtype=object)
        column_names = ["col_str", "col_obj", "col_int"]

        await self.storage.write_data(dataset_name, original_dataset, column_names)
        retrieved_dataset = await self.storage.read_data(dataset_name)

        # Ensure we got exactly one row with the expected values back
        self.assertEqual(1, retrieved_dataset.shape[0])
        self.assertEqual(3, retrieved_dataset.shape[1])
        retrieved_row = list(retrieved_dataset[0])

        self.assertEqual(allowed_string, retrieved_row[0])
        self.assertEqual({"key": "value"}, retrieved_row[1])
        self.assertEqual(456, retrieved_row[2])

    async def _test_deserialize_corrupted_payload(self):
        """Test that corrupted payload data raises DeserializationError."""
        from src.service.data.storage.exceptions import DeserializationError

        request_id = str(uuid.uuid4())

        # First, manually store corrupted data
        # We'll directly write corrupted bytes to HDF5 storage
        from src.service.data.storage.pvc import PARTIAL_INPUT_NAME

        dataset_name = PARTIAL_INPUT_NAME
        corrupted_data = b"\x1f\x8b\x08\x00\xff\xff\xff\xff"  # Invalid gzip

        async with self.storage.get_lock(dataset_name):
            file_path = self.storage._get_filepath(dataset_name)
            with h5py.File(file_path, "a") as db:
                if dataset_name not in db:
                    db.create_dataset(dataset_name, data=np.array([]), maxshape=(None,))

                dataset = db[dataset_name]
                dataset.attrs[request_id] = np.void(corrupted_data)

        # Now try to retrieve it - should raise DeserializationError
        with self.assertRaises(DeserializationError) as context:
            await self.storage.get_partial_payload(request_id, is_input=True, is_modelmesh=True)

        # Verify error contains useful information
        self.assertIn(request_id, str(context.exception))
        self.assertIn("Failed to deserialize", str(context.exception))

        # Clean up
        await self.storage.delete_partial_payload(request_id, is_input=True)

    async def _test_deserialize_invalid_format(self):
        """Test that invalid serialization format raises DeserializationError."""
        from src.service.data.storage.exceptions import DeserializationError

        request_id = str(uuid.uuid4())

        # Store non-gzip, non-JSON data
        from src.service.data.storage.pvc import PARTIAL_INPUT_NAME

        dataset_name = PARTIAL_INPUT_NAME
        invalid_data = b"this is not a valid format"

        async with self.storage.get_lock(dataset_name):
            file_path = self.storage._get_filepath(dataset_name)
            with h5py.File(file_path, "a") as db:
                if dataset_name not in db:
                    db.create_dataset(dataset_name, data=np.array([]), maxshape=(None,))

                dataset = db[dataset_name]
                dataset.attrs[request_id] = np.void(invalid_data)

        # Try to retrieve - should raise DeserializationError
        with self.assertRaises(DeserializationError) as context:
            await self.storage.get_partial_payload(request_id, is_input=True, is_modelmesh=True)

        self.assertIn(request_id, str(context.exception))

        # Clean up
        await self.storage.delete_partial_payload(request_id, is_input=True)

    async def _test_not_found_returns_none(self):
        """Test that missing payload returns None (not an exception)."""
        nonexistent_id = str(uuid.uuid4())

        # Should return None, not raise exception
        result = await self.storage.get_partial_payload(nonexistent_id, is_input=True, is_modelmesh=True)
        self.assertIsNone(result)


def run_async_test(coro):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


TestPayloadReconciliation.test_persist_input_payload = lambda self: run_async_test(self._test_persist_input_payload())  # type: ignore[attr-defined]
TestPayloadReconciliation.test_persist_output_payload = lambda self: run_async_test(self._test_persist_output_payload())  # type: ignore[attr-defined]
TestPayloadReconciliation.test_full_reconciliation = lambda self: run_async_test(self._test_full_reconciliation())  # type: ignore[attr-defined]
TestPayloadReconciliation.test_reconciliation_with_real_data = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_reconciliation_with_real_data()
)
TestPayloadReconciliation.test_void_type_length_exceeds_limit = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_void_type_length_exceeds_limit()
)
TestPayloadReconciliation.test_void_type_length_within_limit = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_void_type_length_within_limit()
)
TestPayloadReconciliation.test_corrupted_payload_handling = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_corrupted_payload_handling()
)
TestPayloadReconciliation.test_deserialize_corrupted_payload = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_deserialize_corrupted_payload()
)
TestPayloadReconciliation.test_deserialize_invalid_format = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_deserialize_invalid_format()
)
TestPayloadReconciliation.test_not_found_returns_none = lambda self: run_async_test(  # type: ignore[attr-defined]
    self._test_not_found_returns_none()
)


if __name__ == "__main__":
    unittest.main()
