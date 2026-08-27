"""Tests for KServe inference logger CloudEvent consumption.

The KServe inference logger POSTs CloudEvents to ``/`` on the health port
(8080).  These tests verify that the endpoint correctly:

- Discriminates request vs response payloads using the ``ce-type`` header
- Extracts the model name from the ``Inferenceservicename`` header
- Correlates request/response pairs via the ``ce-id`` header
- Decompresses gzip bodies (Knative strips Content-Encoding)
- Falls back to discriminated-union parsing when ``ce-type`` is absent
"""

import gzip
import json
import unittest
from http import HTTPStatus
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints.consumer import (
    KServeData,
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.endpoints.consumer.consumer_endpoint import (
    CE_TYPE_REQUEST,
    CE_TYPE_RESPONSE,
    consume_cloud_event,
)


def _make_request_dict(
    *,
    id_: str | None = "req-1",
    n_rows: int = 3,
) -> dict:
    """Build a minimal KServe V2 inference request dict."""
    payload: dict = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [n_rows],
                "datatype": "FP32",
                "data": [float(i) for i in range(n_rows)],
            },
        ],
    }
    if id_ is not None:
        payload["id"] = id_
    return payload


def _make_response_dict(
    *,
    id_: str | None = "req-1",
    model_name: str | None = "test-model",
    n_rows: int = 3,
) -> dict:
    """Build a minimal KServe V2 inference response dict."""
    payload: dict = {
        "outputs": [
            {
                "name": "output-0",
                "shape": [n_rows],
                "datatype": "FP32",
                "data": [float(i) for i in range(n_rows)],
            },
        ],
    }
    if id_ is not None:
        payload["id"] = id_
    if model_name is not None:
        payload["model_name"] = model_name
    return payload


def _health_app() -> FastAPI:
    """Build a minimal FastAPI app mirroring the health_app registration."""
    app = FastAPI()
    app.post("/")(consume_cloud_event)
    return app


class TestCeTypeDiscrimination(unittest.TestCase):
    """Verify that the ce-type header controls payload parsing."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_storage.persist_partial_payload = mock.AsyncMock()

        self.client = TestClient(_health_app(), raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    def test_ce_type_request_parses_as_request(self) -> None:
        """Body is parsed as KServeInferenceRequest when ce-type is request."""
        payload = _make_request_dict()
        resp = self.client.post(
            "/",
            json=payload,
            headers={
                "ce-type": CE_TYPE_REQUEST,
                "ce-id": "corr-1",
            },
        )
        assert resp.status_code == HTTPStatus.OK
        assert "input" in resp.json()["message"].lower()

        call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]
        assert call_kwargs["is_input"] is True

    def test_ce_type_response_parses_as_response(self) -> None:
        """Body is parsed as KServeInferenceResponse when ce-type is response."""
        payload = _make_response_dict()
        resp = self.client.post(
            "/",
            json=payload,
            headers={
                "ce-type": CE_TYPE_RESPONSE,
                "ce-id": "corr-1",
            },
        )
        assert resp.status_code == HTTPStatus.OK
        assert "output" in resp.json()["message"].lower()

        call_kwargs = self.mock_storage.persist_partial_payload.call_args[1]
        assert call_kwargs["is_input"] is False

    def test_no_ce_type_falls_back_to_discriminated_parsing(self) -> None:
        """Without ce-type, TypeAdapter discriminates by field presence."""
        payload = _make_request_dict()
        resp = self.client.post(
            "/",
            json=payload,
            headers={"ce-id": "corr-2"},
        )
        assert resp.status_code == HTTPStatus.OK

    def test_unknown_ce_type_falls_back_to_discriminated_parsing(self) -> None:
        """An unrecognised ce-type falls back to discriminated parsing."""
        payload = _make_response_dict()
        resp = self.client.post(
            "/",
            json=payload,
            headers={
                "ce-type": "com.example.unknown",
                "ce-id": "corr-3",
            },
        )
        assert resp.status_code == HTTPStatus.OK


class TestInferenceServiceNameHeader(unittest.TestCase):
    """Verify Inferenceservicename header sets model_name on responses."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_storage.persist_partial_payload = mock.AsyncMock()

        self.client = TestClient(_health_app(), raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    def test_model_name_set_from_header_when_missing_in_body(self) -> None:
        """Inferenceservicename header fills in model_name when body omits it."""
        payload = _make_response_dict(model_name=None)
        resp = self.client.post(
            "/",
            json=payload,
            headers={
                "ce-type": CE_TYPE_RESPONSE,
                "ce-id": "corr-4",
                "Inferenceservicename": "my-model",
            },
        )
        assert resp.status_code == HTTPStatus.OK

        persisted_payload = self.mock_storage.persist_partial_payload.call_args[0][0]
        assert isinstance(persisted_payload, KServeInferenceResponse)
        assert persisted_payload.model_name == "my-model"

    def test_model_name_overridden_by_header(self) -> None:
        """Inferenceservicename header overrides the body model_name."""
        payload = _make_response_dict(model_name="body-model")
        resp = self.client.post(
            "/",
            json=payload,
            headers={
                "ce-type": CE_TYPE_RESPONSE,
                "ce-id": "corr-5",
                "Inferenceservicename": "header-model",
            },
        )
        assert resp.status_code == HTTPStatus.OK

        persisted_payload = self.mock_storage.persist_partial_payload.call_args[0][0]
        assert persisted_payload.model_name == "header-model"

    def test_header_does_not_affect_request_payloads(self) -> None:
        """Inferenceservicename header is ignored for request payloads."""
        payload = _make_request_dict()
        resp = self.client.post(
            "/",
            json=payload,
            headers={
                "ce-type": CE_TYPE_REQUEST,
                "ce-id": "corr-6",
                "Inferenceservicename": "my-model",
            },
        )
        assert resp.status_code == HTTPStatus.OK

        persisted_payload = self.mock_storage.persist_partial_payload.call_args[0][0]
        assert isinstance(persisted_payload, KServeInferenceRequest)


class TestCloudEventCorrelation(unittest.TestCase):
    """Verify ce-id is used for request/response correlation."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage

        self.client = TestClient(_health_app(), raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    def test_response_triggers_reconciliation_when_request_stored(self) -> None:
        """Response with matching stored request triggers reconciliation."""
        stored_input = KServeInferenceRequest(
            id="corr-7",
            inputs=[
                KServeData(
                    name="input-0",
                    shape=[3],
                    datatype="FP32",
                    data=[1.0, 2.0, 3.0],
                ),
            ],
        )
        self.mock_storage.get_partial_payload = mock.AsyncMock(
            return_value=stored_input
        )

        with mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.reconcile_kserve",
            new=mock.AsyncMock(),
        ) as mock_reconcile:
            payload = _make_response_dict(id_=None, n_rows=3)
            resp = self.client.post(
                "/",
                json=payload,
                headers={
                    "ce-type": CE_TYPE_RESPONSE,
                    "ce-id": "corr-7",
                    "Inferenceservicename": "my-model",
                },
            )
            assert resp.status_code == HTTPStatus.OK
            mock_reconcile.assert_called_once()

            call_args = mock_reconcile.call_args[0]
            assert call_args[0] is stored_input
            assert isinstance(call_args[1], KServeInferenceResponse)
            assert call_args[1].model_name == "my-model"

    def test_request_triggers_reconciliation_when_response_stored(self) -> None:
        """Request with matching stored response triggers reconciliation."""
        stored_output = KServeInferenceResponse(
            id="corr-8",
            model_name="my-model",
            outputs=[
                KServeData(
                    name="output-0",
                    shape=[3],
                    datatype="FP32",
                    data=[1.0, 2.0, 3.0],
                ),
            ],
        )
        self.mock_storage.get_partial_payload = mock.AsyncMock(
            return_value=stored_output
        )

        with mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.reconcile_kserve",
            new=mock.AsyncMock(),
        ) as mock_reconcile:
            payload = _make_request_dict(id_=None, n_rows=3)
            resp = self.client.post(
                "/",
                json=payload,
                headers={
                    "ce-type": CE_TYPE_REQUEST,
                    "ce-id": "corr-8",
                },
            )
            assert resp.status_code == HTTPStatus.OK
            mock_reconcile.assert_called_once()


class TestCloudEventGzipHandling(unittest.TestCase):
    """Verify gzip-compressed CloudEvent bodies are decompressed."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_storage.persist_partial_payload = mock.AsyncMock()

        self.client = TestClient(_health_app(), raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    def test_gzip_compressed_request_is_decompressed(self) -> None:
        """Gzip body is decompressed before parsing as inference request."""
        payload = _make_request_dict()
        compressed = gzip.compress(json.dumps(payload).encode())

        resp = self.client.post(
            "/",
            content=compressed,
            headers={
                "ce-type": CE_TYPE_REQUEST,
                "ce-id": "gz-1",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == HTTPStatus.OK

    def test_gzip_compressed_response_is_decompressed(self) -> None:
        """Gzip body is decompressed before parsing as inference response."""
        payload = _make_response_dict()
        compressed = gzip.compress(json.dumps(payload).encode())

        resp = self.client.post(
            "/",
            content=compressed,
            headers={
                "ce-type": CE_TYPE_RESPONSE,
                "ce-id": "gz-2",
                "Inferenceservicename": "my-model",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == HTTPStatus.OK


class TestCloudEventValidationErrors(unittest.TestCase):
    """Verify proper error responses for invalid CloudEvent payloads."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage

        self.client = TestClient(_health_app(), raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    def test_invalid_json_returns_400(self) -> None:
        """Non-JSON body with ce-type returns 400."""
        resp = self.client.post(
            "/",
            content=b"not json",
            headers={
                "ce-type": CE_TYPE_REQUEST,
                "ce-id": "bad-1",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == HTTPStatus.BAD_REQUEST

    def test_wrong_ce_type_for_body_returns_400(self) -> None:
        """ce-type says request but body is a response format -> 400."""
        payload = _make_response_dict()  # has 'outputs', not 'inputs'
        resp = self.client.post(
            "/",
            json=payload,
            headers={
                "ce-type": CE_TYPE_REQUEST,
                "ce-id": "bad-2",
            },
        )
        assert resp.status_code == HTTPStatus.BAD_REQUEST

    def test_missing_ce_id_and_payload_id_returns_400(self) -> None:
        """No ce-id header and no id in body returns 400."""
        payload = _make_request_dict(id_=None)
        resp = self.client.post(
            "/",
            json=payload,
            headers={
                "ce-type": CE_TYPE_REQUEST,
            },
        )
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "id" in resp.json()["detail"].lower()


if __name__ == "__main__":
    unittest.main()
