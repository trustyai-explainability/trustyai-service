"""
Tests for ModelMesh payload reconciliation.
"""

import asyncio
import unittest
import tempfile
import os
from unittest import mock
import uuid

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
        print(self.storage.list_all_datasets())
        self.temp_dir.cleanup()

    async def _test_persist_input_payload(self):
        """Test persisting an input payload."""
        await self.storage.persist_modelmesh_payload(self.input_payload, self.request_id, is_input=True)

        retrieved_payload = await self.storage.get_modelmesh_payload(self.request_id, is_input=True)

        self.assertIsNotNone(retrieved_payload)
        self.assertEqual(retrieved_payload.data, self.input_payload.data)

        output_payload = await self.storage.get_modelmesh_payload(self.request_id, is_input=False)
        self.assertIsNone(output_payload)

    async def _test_persist_output_payload(self):
        """Test persisting an output payload."""
        await self.storage.persist_modelmesh_payload(self.output_payload, self.request_id, is_input=False)

        retrieved_payload = await self.storage.get_modelmesh_payload(self.request_id, is_input=False)

        self.assertIsNotNone(retrieved_payload)
        self.assertEqual(retrieved_payload.data, self.output_payload.data)

        input_payload = await self.storage.get_modelmesh_payload(self.request_id, is_input=True)
        self.assertIsNone(input_payload)

    async def _test_full_reconciliation(self):
        """Test the full payload reconciliation process."""
        await self.storage.persist_modelmesh_payload(self.input_payload, self.request_id, is_input=True)
        await self.storage.persist_modelmesh_payload(self.output_payload, self.request_id, is_input=False)

        input_payload = await self.storage.get_modelmesh_payload(self.request_id, is_input=True)
        output_payload = await self.storage.get_modelmesh_payload(self.request_id, is_input=False)

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
        await self.storage.delete_modelmesh_payload(self.request_id, is_input=True)
        await self.storage.delete_modelmesh_payload(self.request_id, is_input=False)

        input_payload = await self.storage.get_modelmesh_payload(self.request_id, is_input=True)
        output_payload = await self.storage.get_modelmesh_payload(self.request_id, is_input=False)

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

        await self.storage.persist_modelmesh_payload(input_payload, request_id, is_input=True)
        await self.storage.persist_modelmesh_payload(output_payload, request_id, is_input=False)

        stored_input = await self.storage.get_modelmesh_payload(request_id, is_input=True)
        stored_output = await self.storage.get_modelmesh_payload(request_id, is_input=False)

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
        await self.storage.delete_modelmesh_payload(request_id, is_input=True)
        await self.storage.delete_modelmesh_payload(request_id, is_input=False)

        self.assertIsNone(await self.storage.get_modelmesh_payload(request_id, is_input=True))
        self.assertIsNone(await self.storage.get_modelmesh_payload(request_id, is_input=False))


def run_async_test(coro):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


TestPayloadReconciliation.test_persist_input_payload = lambda self: run_async_test(self._test_persist_input_payload())
TestPayloadReconciliation.test_persist_output_payload = lambda self: run_async_test(self._test_persist_output_payload())
TestPayloadReconciliation.test_full_reconciliation = lambda self: run_async_test(self._test_full_reconciliation())
TestPayloadReconciliation.test_reconciliation_with_real_data = lambda self: run_async_test(
    self._test_reconciliation_with_real_data()
)


if __name__ == "__main__":
    unittest.main()
