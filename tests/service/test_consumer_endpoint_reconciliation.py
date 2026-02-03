"""
Tests for ModelMesh payload reconciliation through the consumer endpoint.
"""

import asyncio
import unittest
import tempfile
import uuid
from unittest import mock

from fastapi.testclient import TestClient

from src.service.data.modelmesh_parser import ModelMeshPayloadParser, PartialPayload
from src.endpoints.consumer.consumer_endpoint import router as consumer_router
from tests.service.data.test_utils import ModelMeshTestData


class TestConsumerEndpointReconciliation(unittest.TestCase):
    """
    Test class for ModelMesh payload reconciliation through the consumer endpoint.
    """

    def setUp(self):
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()

        self.storage_patch = mock.patch(
            "src.endpoints.consumer.consumer_endpoint.get_global_storage_interface"
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage

        self.parser_patch = mock.patch.object(ModelMeshPayloadParser, "parse_input_payload")
        self.mock_parse_input = self.parser_patch.start()

        self.parser_output_patch = mock.patch.object(ModelMeshPayloadParser, "parse_output_payload")
        self.mock_parse_output = self.parser_output_patch.start()

        self.parser_dataframe_patch = mock.patch.object(ModelMeshPayloadParser, "payloads_to_dataframe")
        self.mock_to_dataframe = self.parser_dataframe_patch.start()

        self.model_data_patch = mock.patch("src.endpoints.consumer.consumer_endpoint.ModelData")
        self.mock_model_data = self.model_data_patch.start()
        self.mock_model_data.return_value.shapes.return_value = [
            (5, 10),
            (5, 1),
            (5, 3),
        ]

        from fastapi import FastAPI

        self.app = FastAPI()
        self.app.include_router(consumer_router)
        self.client = TestClient(self.app)

        self.model_name = "test-model"
        self.request_id = str(uuid.uuid4())

        input_specs = [("input", 5, 10, "INT32", 0, None)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        self.input_payload_dict, self.output_payload_dict, _, _ = ModelMeshTestData.generate_test_payloads(
            self.model_name, input_specs, output_specs
        )

        self.input_payload = PartialPayload(**self.input_payload_dict)
        self.output_payload = PartialPayload(**self.output_payload_dict)

        self.mock_df = mock.MagicMock()
        self.mock_df.columns = ["input", "output_output", "id", "model_id", "synthetic"]
        self.mock_df.__len__.return_value = 5
        self.mock_df.__getitem__.return_value.values = mock.MagicMock()

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
        self.storage_patch.stop()
        self.parser_patch.stop()
        self.parser_output_patch.stop()
        self.parser_dataframe_patch.stop()
        self.model_data_patch.stop()

    async def _test_consume_input_payload(self):
        """Test consuming an input payload."""
        self.mock_storage.persist_partial_payload = mock.AsyncMock()
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_parse_input.return_value = True
        self.mock_parse_output.side_effect = ValueError("Not an output payload")

        inference_payload = {
            "data": self.input_payload.data,
            "modelid": self.model_name,
            "partialPayloadId": {"prediction_id": self.request_id, "kind": "request"},
        }

        response = self.client.post("/consumer/kserve/v2", json=inference_payload)
        print(response.text)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "success",
                "message": f"Payload for {self.request_id} processed successfully",
            },
        )

        self.mock_storage.persist_partial_payload.assert_called_once()
        call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]
        self.assertEqual(call_kwargs["payload_id"], self.request_id)
        self.assertTrue(call_kwargs["is_input"])  # is_input=True

    async def _test_consume_output_payload(self):
        """Test consuming an output payload."""
        self.mock_storage.persist_partial_payload = mock.AsyncMock()
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_parse_input.side_effect = ValueError("Not an input payload")
        self.mock_parse_output.return_value = True

        inference_payload = {
            "data": self.output_payload.data,
            "modelid": self.model_name,
            "partialPayloadId": {"prediction_id": self.request_id, "kind": "response"},
        }

        response = self.client.post("/consumer/kserve/v2", json=inference_payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "success",
                "message": f"Payload for {self.request_id} processed successfully",
            },
        )

        self.mock_storage.persist_partial_payload.assert_called_once()
        call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]
        self.assertEqual(call_kwargs["payload_id"], self.request_id)
        self.assertFalse(call_kwargs["is_input"])  # is_input=True

    async def _test_reconcile_payloads(self):
        """Test reconciling both input and output payloads."""
        # Setup mocks for correct interactions
        self.mock_storage.get_partial_payload = mock.AsyncMock()
        self.mock_storage.get_partial_payload.side_effect = [
            None,
            self.input_payload,
        ]

        self.mock_storage.persist_partial_payload = mock.AsyncMock()
        self.mock_storage.write_data = mock.AsyncMock()
        self.mock_storage.delete_partial_payload = mock.AsyncMock()

        with (
            mock.patch(
                "src.endpoints.consumer.consumer_endpoint.ModelMeshPayloadParser.parse_input_payload",
                side_effect=lambda x: True,
            ) as mock_parse_input,
            mock.patch(
                "src.endpoints.consumer.consumer_endpoint.ModelMeshPayloadParser.parse_output_payload",
                side_effect=ValueError("Not an output"),
            ) as mock_parse_output,
            mock.patch(
                "src.endpoints.consumer.consumer_endpoint.ModelMeshPayloadParser.payloads_to_dataframe",
                return_value=self.mock_df,
            ) as mock_df,
            mock.patch(
                "src.endpoints.consumer.consumer_endpoint.asyncio.gather",
            ) as mock_gather,
            mock.patch(
                "src.endpoints.consumer.consumer_endpoint.reconcile_modelmesh_payloads",
                new=mock.AsyncMock(),
            ) as mock_reconcile,
        ):

            async def mock_gather_impl(*args, **kwargs):
                results = []
                for coro in args:
                    if hasattr(coro, "__await__"):
                        results.append(None)
                return results

            mock_gather.side_effect = mock_gather_impl

            input_inference_payload = {
                "data": self.input_payload.data,
                "modelid": self.model_name,
                "partialPayloadId": {
                    "prediction_id": self.request_id,
                    "kind": "request",
                },
            }

            response_input = self.client.post("/consumer/kserve/v2", json=input_inference_payload)

            self.assertEqual(response_input.status_code, 200)
            self.assertEqual(
                response_input.json(),
                {
                    "status": "success",
                    "message": f"Payload for {self.request_id} processed successfully",
                },
            )

            self.mock_storage.persist_partial_payload.assert_called_once()
            call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]

            self.assertEqual(call_kwargs['payload_id'], self.request_id)
            self.assertTrue(call_kwargs['is_input'])  # is_input=True

            self.mock_storage.persist_partial_payload.reset_mock()

            mock_parse_input.side_effect = ValueError("Not an input")
            mock_parse_output.side_effect = lambda x: True

            output_inference_payload = {
                "data": self.output_payload.data,
                "modelid": self.model_name,
                "partialPayloadId": {
                    "prediction_id": self.request_id,
                    "kind": "response",
                },
            }

            response_output = self.client.post("/consumer/kserve/v2", json=output_inference_payload)

            self.assertEqual(response_output.status_code, 200)
            self.assertEqual(
                response_output.json(),
                {
                    "status": "success",
                    "message": f"Payload for {self.request_id} processed successfully",
                },
            )

            call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]
            self.assertEqual(call_kwargs["payload_id"], self.request_id)
            self.assertFalse(call_kwargs["is_input"])  # is_input=False

            mock_reconcile.assert_called_once()
            reconcile_args = mock_reconcile.call_args[0]
            self.assertEqual(reconcile_args[2], self.request_id)  # request_id
            self.assertEqual(reconcile_args[3], self.model_name)  # model_id


def run_async_test(coro):
    """Helper function to run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


TestConsumerEndpointReconciliation.test_consume_input_payload = lambda self: run_async_test(
    self._test_consume_input_payload()
)
TestConsumerEndpointReconciliation.test_consume_output_payload = lambda self: run_async_test(
    self._test_consume_output_payload()
)
TestConsumerEndpointReconciliation.test_reconcile_payloads = lambda self: run_async_test(
    self._test_reconcile_payloads()
)


if __name__ == "__main__":
    unittest.main()
