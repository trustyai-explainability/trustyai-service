"""Tests for ModelMesh payload reconciliation through the consumer endpoint."""

import asyncio
import tempfile
import unittest
import uuid
from collections.abc import Coroutine
from http import HTTPStatus
from typing import Any
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.service.data.test_utils import ModelMeshTestData
from trustyai_service.endpoints.consumer.consumer_endpoint import (
    router as consumer_router,
)
from trustyai_service.service.data.modelmesh_parser import (
    ModelMeshPayloadParser,
    PartialPayload,
)


class TestConsumerEndpointReconciliation(unittest.TestCase):
    """Test class for ModelMesh payload reconciliation through the consumer endpoint."""

    def setUp(self) -> None:
        """Set up the test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()

        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage

        self.parser_patch = mock.patch.object(
            ModelMeshPayloadParser,
            "parse_input_payload",
        )
        self.mock_parse_input = self.parser_patch.start()

        self.parser_output_patch = mock.patch.object(
            ModelMeshPayloadParser,
            "parse_output_payload",
        )
        self.mock_parse_output = self.parser_output_patch.start()

        self.parser_dataframe_patch = mock.patch.object(
            ModelMeshPayloadParser,
            "payloads_to_dataframe",
        )
        self.mock_to_dataframe = self.parser_dataframe_patch.start()

        self.model_data_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.ModelData",
        )
        self.mock_model_data = self.model_data_patch.start()
        self.mock_model_data.return_value.shapes.return_value = [
            (5, 10),
            (5, 1),
            (5, 3),
        ]

        self.app = FastAPI()
        self.app.include_router(consumer_router)
        self.client = TestClient(self.app)

        self.model_name = "test-model"
        self.request_id = str(uuid.uuid4())

        input_specs = [("input", 5, 10, "INT32", 0, None)]
        output_specs = [("output", 5, 1, "INT32", 0)]

        self.input_payload_dict, self.output_payload_dict, _, _ = (
            ModelMeshTestData.generate_test_payloads(
                self.model_name,
                input_specs,
                output_specs,
            )
        )

        self.input_payload = PartialPayload(**self.input_payload_dict)
        self.output_payload = PartialPayload(**self.output_payload_dict)

        self.mock_df = mock.MagicMock()
        self.mock_df.columns = ["input", "output_output", "id", "model_id", "synthetic"]
        self.mock_df.__len__.return_value = 5
        self.mock_df.__getitem__.return_value.values = mock.MagicMock()

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.temp_dir.cleanup()
        self.storage_patch.stop()
        self.parser_patch.stop()
        self.parser_output_patch.stop()
        self.parser_dataframe_patch.stop()
        self.model_data_patch.stop()

    async def _test_consume_input_payload(self) -> None:
        """Test consuming an input payload."""
        self.mock_storage.persist_partial_payload = mock.AsyncMock()
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_parse_input.return_value = True
        self.mock_parse_output.side_effect = ValueError("Not an output payload")

        inference_payload = {
            "data": self.input_payload.data,
            "modelid": self.model_name,
            "id": self.request_id,
            "kind": "request",
        }

        response = self.client.post("/consumer/kserve/v2", json=inference_payload)

        assert response.status_code == HTTPStatus.OK
        assert response.json() == {
            "status": "success",
            "message": f"Payload for {self.request_id} processed successfully",
        }

        self.mock_storage.persist_partial_payload.assert_called_once()
        call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]
        assert call_kwargs["payload_id"] == self.request_id
        assert call_kwargs["is_input"]  # is_input=True

    async def _test_consume_output_payload(self) -> None:
        """Test consuming an output payload."""
        self.mock_storage.persist_partial_payload = mock.AsyncMock()
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_parse_input.side_effect = ValueError("Not an input payload")
        self.mock_parse_output.return_value = True

        inference_payload = {
            "data": self.output_payload.data,
            "modelid": self.model_name,
            "id": self.request_id,
            "kind": "response",
        }

        response = self.client.post("/consumer/kserve/v2", json=inference_payload)

        assert response.status_code == HTTPStatus.OK
        assert response.json() == {
            "status": "success",
            "message": f"Payload for {self.request_id} processed successfully",
        }

        self.mock_storage.persist_partial_payload.assert_called_once()
        call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]
        assert call_kwargs["payload_id"] == self.request_id
        assert not call_kwargs["is_input"]  # is_input=True

    async def _test_reconcile_payloads(self) -> None:
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
                "trustyai_service.endpoints.consumer.consumer_endpoint.ModelMeshPayloadParser.parse_input_payload",
                side_effect=lambda _x: True,
            ) as mock_parse_input,
            mock.patch(
                "trustyai_service.endpoints.consumer.consumer_endpoint.ModelMeshPayloadParser.parse_output_payload",
                side_effect=ValueError("Not an output"),
            ) as mock_parse_output,
            mock.patch(
                "trustyai_service.endpoints.consumer.consumer_endpoint.ModelMeshPayloadParser.payloads_to_dataframe",
                return_value=self.mock_df,
            ) as _mock_df,
            mock.patch(
                "trustyai_service.endpoints.consumer.consumer_endpoint.asyncio.gather",
            ) as mock_gather,
            mock.patch(
                "trustyai_service.endpoints.consumer.consumer_endpoint.reconcile_modelmesh_payloads",
                new=mock.AsyncMock(),
            ) as mock_reconcile,
        ):

            async def mock_gather_impl(*args: object, **_kwargs: object) -> list[None]:
                return [None for coro in args if hasattr(coro, "__await__")]

            mock_gather.side_effect = mock_gather_impl

            input_inference_payload = {
                "data": self.input_payload.data,
                "modelid": self.model_name,
                "id": self.request_id,
                "kind": "request",
            }

            response_input = self.client.post(
                "/consumer/kserve/v2",
                json=input_inference_payload,
            )

            assert response_input.status_code == HTTPStatus.OK
            assert response_input.json() == {
                "status": "success",
                "message": f"Payload for {self.request_id} processed successfully",
            }

            self.mock_storage.persist_partial_payload.assert_called_once()
            call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]

            assert call_kwargs["payload_id"] == self.request_id
            assert call_kwargs["is_input"]  # is_input=True

            self.mock_storage.persist_partial_payload.reset_mock()

            mock_parse_input.side_effect = ValueError("Not an input")
            mock_parse_output.side_effect = lambda _x: True

            output_inference_payload = {
                "data": self.output_payload.data,
                "modelid": self.model_name,
                "id": self.request_id,
                "kind": "response",
            }

            response_output = self.client.post(
                "/consumer/kserve/v2",
                json=output_inference_payload,
            )

            assert response_output.status_code == HTTPStatus.OK
            assert response_output.json() == {
                "status": "success",
                "message": f"Payload for {self.request_id} processed successfully",
            }

            call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]
            assert call_kwargs["payload_id"] == self.request_id
            assert not call_kwargs["is_input"]  # is_input=False

            mock_reconcile.assert_called_once()
            reconcile_args = mock_reconcile.call_args[0]
            assert reconcile_args[2] == self.request_id  # request_id
            assert reconcile_args[3] == self.model_name  # model_id


def run_async_test(coro: Coroutine[Any, Any, None]) -> None:
    """Run async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


TestConsumerEndpointReconciliation.test_consume_input_payload = lambda self: (  # type: ignore[attr-defined]
    run_async_test(self._test_consume_input_payload())
)
TestConsumerEndpointReconciliation.test_consume_output_payload = lambda self: (  # type: ignore[attr-defined]
    run_async_test(self._test_consume_output_payload())
)
TestConsumerEndpointReconciliation.test_reconcile_payloads = lambda self: (  # type: ignore[attr-defined]
    run_async_test(self._test_reconcile_payloads())
)


class TestConsumerEndpointValidation(unittest.TestCase):
    """Test validation of required fields and invalid values on the consumer endpoint."""

    def setUp(self) -> None:
        """Set up test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_get_storage.return_value = mock.AsyncMock()

        self.app = FastAPI()
        self.app.include_router(consumer_router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

        self.valid_payload = {
            "id": "test-id",
            "kind": "request",
            "modelid": "test-model",
            "data": "dGVzdA==",
        }

    def tearDown(self) -> None:
        """Clean up patches."""
        self.storage_patch.stop()

    def test_missing_id_returns_400(self) -> None:
        """Payload without 'id' field is rejected with 400."""
        payload = {**self.valid_payload}
        del payload["id"]
        response = self.client.post("/consumer/kserve/v2", json=payload)
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "id" in response.json()["detail"]

    def test_missing_kind_returns_400(self) -> None:
        """Payload without 'kind' field is rejected with 400."""
        payload = {**self.valid_payload}
        del payload["kind"]
        response = self.client.post("/consumer/kserve/v2", json=payload)
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "kind" in response.json()["detail"]

    def test_missing_modelid_returns_400(self) -> None:
        """Payload without 'modelid' field is rejected with 400."""
        payload = {**self.valid_payload}
        del payload["modelid"]
        response = self.client.post("/consumer/kserve/v2", json=payload)
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "modelid" in response.json()["detail"]

    def test_missing_data_returns_400(self) -> None:
        """Payload without 'data' field is rejected with 400."""
        payload = {**self.valid_payload}
        del payload["data"]
        response = self.client.post("/consumer/kserve/v2", json=payload)
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "data" in response.json()["detail"]

    def test_invalid_kind_returns_422(self) -> None:
        """Invalid kind value is rejected by Pydantic with 422."""
        payload = {**self.valid_payload, "kind": "prediction"}
        response = self.client.post("/consumer/kserve/v2", json=payload)
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY

    def test_nested_format_returns_400(self) -> None:
        """Old nested format should fail — partialPayloadId is ignored, id is null."""
        payload = {
            "partialPayloadId": {"prediction_id": "test-id", "kind": "request"},
            "modelid": "test-model",
            "data": "dGVzdA==",
        }
        response = self.client.post("/consumer/kserve/v2", json=payload)
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "id" in response.json()["detail"]


if __name__ == "__main__":
    unittest.main()
