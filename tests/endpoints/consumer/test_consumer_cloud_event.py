"""Tests for the consume_cloud_event endpoint (POST /).

Covers error paths, reconciliation edge cases, tag assignment, and
storage interaction for the KServe cloud-event consumer.
"""

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
    router as consumer_router,
)
from trustyai_service.exceptions import ReconciliationError


def _make_request(
    *,
    id_: str | None = "req-1",
    n_rows: int = 3,
    n_cols: int = 2,
    parameters: dict[str, str] | None = None,
) -> dict:
    """Build a minimal KServeInferenceRequest dict."""
    if n_cols > 1:
        data = [[float(r * n_cols + c) for c in range(n_cols)] for r in range(n_rows)]
        shape = [n_rows, n_cols]
    else:
        data = [float(i) for i in range(n_rows)]
        shape = [n_rows]

    payload: dict = {
        "inputs": [
            {
                "name": "input",
                "shape": shape,
                "datatype": "FP32",
                "data": data,
            },
        ],
    }
    if id_ is not None:
        payload["id"] = id_
    if parameters is not None:
        payload["parameters"] = parameters
    return payload


def _make_response(
    *,
    id_: str | None = "req-1",
    model_name: str = "test-model",
    n_rows: int = 3,
    n_cols: int = 1,
) -> dict:
    """Build a minimal KServeInferenceResponse dict."""
    if n_cols > 1:
        data = [[float(r * n_cols + c) for c in range(n_cols)] for r in range(n_rows)]
        shape = [n_rows, n_cols]
    else:
        data = [float(i) for i in range(n_rows)]
        shape = [n_rows]

    payload: dict = {
        "model_name": model_name,
        "outputs": [
            {
                "name": "output",
                "shape": shape,
                "datatype": "FP32",
                "data": data,
            },
        ],
    }
    if id_ is not None:
        payload["id"] = id_
    return payload


class TestCloudEventValidation(unittest.TestCase):
    """Validation and error-path tests for POST /."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage

        self.app = FastAPI()
        self.app.include_router(consumer_router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    # -- missing-ID paths --

    def test_request_without_id_or_header_returns_400(self) -> None:
        """Request payload with no id and no ce-id header is rejected."""
        payload = _make_request(id_=None)
        resp = self.client.post("/", json=payload)
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "id" in resp.json()["detail"].lower()

    def test_response_without_id_or_header_returns_400(self) -> None:
        """Response payload with no id and no ce-id header is rejected."""
        payload = _make_response(id_=None)
        resp = self.client.post("/", json=payload)
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "id" in resp.json()["detail"].lower()

    # -- ce-id header override --

    def test_ce_id_header_overrides_payload_id(self) -> None:
        """The ce-id header should override the payload id field."""
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_storage.persist_partial_payload = mock.AsyncMock()

        payload = _make_request(id_="original-id")
        resp = self.client.post("/", json=payload, headers={"ce-id": "overridden-id"})
        assert resp.status_code == HTTPStatus.OK
        assert "overridden-id" in resp.json()["message"]

    def test_ce_id_header_provides_id_when_missing(self) -> None:
        """When payload has no id, ce-id header fills it in."""
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_storage.persist_partial_payload = mock.AsyncMock()

        payload = _make_request(id_=None)
        resp = self.client.post("/", json=payload, headers={"ce-id": "from-header"})
        assert resp.status_code == HTTPStatus.OK
        assert "from-header" in resp.json()["message"]

    # -- empty data fields --

    def test_request_with_empty_inputs_returns_400(self) -> None:
        """Request with empty inputs list is rejected."""
        payload: dict = {"id": "req-1", "inputs": []}
        resp = self.client.post("/", json=payload)
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "empty" in resp.json()["detail"].lower()

    def test_response_with_empty_outputs_returns_400(self) -> None:
        """Response with empty outputs list is rejected."""
        payload: dict = {
            "id": "req-1",
            "model_name": "test-model",
            "outputs": [],
        }
        resp = self.client.post("/", json=payload)
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "empty" in resp.json()["detail"].lower()


class TestCloudEventInputStorageFlow(unittest.TestCase):
    """Test that input payloads are stored or reconciled correctly."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage

        self.app = FastAPI()
        self.app.include_router(consumer_router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    def test_input_stored_when_no_matching_output(self) -> None:
        """Input payload is persisted when no matching output exists."""
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_storage.persist_partial_payload = mock.AsyncMock()

        payload = _make_request(id_="inp-1")
        resp = self.client.post("/", json=payload)

        assert resp.status_code == HTTPStatus.OK
        self.mock_storage.persist_partial_payload.assert_called_once()
        call_kwargs = self.mock_storage.persist_partial_payload.call_args
        assert call_kwargs[1]["payload_id"] == "inp-1"
        assert call_kwargs[1]["is_input"] is True

    def test_input_triggers_reconciliation_when_output_exists(self) -> None:
        """When a matching output exists, reconciliation is triggered."""
        stored_output = KServeInferenceResponse(
            id="inp-2",
            model_name="test-model",
            outputs=[
                KServeData(
                    name="output",
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
            payload = _make_request(id_="inp-2", n_rows=3, n_cols=1)
            resp = self.client.post("/", json=payload)

            assert resp.status_code == HTTPStatus.OK
            mock_reconcile.assert_called_once()
            # First arg is the input, second is the stored output
            call_args = mock_reconcile.call_args[0]
            assert isinstance(call_args[0], KServeInferenceRequest)
            assert call_args[1] is stored_output

    def test_input_invalid_stored_type_returns_500(self) -> None:
        """If storage returns wrong type for output, 500 is raised."""
        # Return a string instead of KServeInferenceResponse
        self.mock_storage.get_partial_payload = mock.AsyncMock(
            return_value="not-a-response"
        )
        payload = _make_request(id_="bad-type-1")
        resp = self.client.post("/", json=payload)
        assert resp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert "invalid payload type" in resp.json()["detail"].lower()


class TestCloudEventOutputStorageFlow(unittest.TestCase):
    """Test that output payloads are stored or reconciled correctly."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage

        self.app = FastAPI()
        self.app.include_router(consumer_router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    def test_output_stored_when_no_matching_input(self) -> None:
        """Output payload is persisted when no matching input exists."""
        self.mock_storage.get_partial_payload = mock.AsyncMock(return_value=None)
        self.mock_storage.persist_partial_payload = mock.AsyncMock()

        payload = _make_response(id_="out-1")
        resp = self.client.post("/", json=payload)

        assert resp.status_code == HTTPStatus.OK
        self.mock_storage.persist_partial_payload.assert_called_once()
        call_kwargs = self.mock_storage.persist_partial_payload.call_args
        assert call_kwargs[1]["payload_id"] == "out-1"
        assert call_kwargs[1]["is_input"] is False

    def test_output_triggers_reconciliation_when_input_exists(self) -> None:
        """When a matching input exists, reconciliation is triggered."""
        stored_input = KServeInferenceRequest(
            id="out-2",
            inputs=[
                KServeData(
                    name="input",
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
            payload = _make_response(id_="out-2", n_rows=3, n_cols=1)
            resp = self.client.post("/", json=payload)

            assert resp.status_code == HTTPStatus.OK
            mock_reconcile.assert_called_once()
            call_args = mock_reconcile.call_args[0]
            assert call_args[0] is stored_input
            assert isinstance(call_args[1], KServeInferenceResponse)

    def test_output_invalid_stored_type_returns_500(self) -> None:
        """If storage returns wrong type for input, 500 is raised."""
        self.mock_storage.get_partial_payload = mock.AsyncMock(
            return_value=42  # not a KServeInferenceRequest
        )
        payload = _make_response(id_="bad-type-2")
        resp = self.client.post("/", json=payload)
        assert resp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert "invalid payload type" in resp.json()["detail"].lower()


class TestCloudEventReconciliationError(unittest.TestCase):
    """ReconciliationError during cloud-event processing is returned as 400."""

    def setUp(self) -> None:
        """Set up FastAPI test client with mocked storage."""
        self.storage_patch = mock.patch(
            "trustyai_service.endpoints.consumer.consumer_endpoint.get_global_storage_interface",
        )
        self.mock_get_storage = self.storage_patch.start()
        self.mock_storage = mock.AsyncMock()
        self.mock_get_storage.return_value = self.mock_storage

        self.app = FastAPI()
        self.app.include_router(consumer_router)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self) -> None:
        """Stop storage mock patch."""
        self.storage_patch.stop()

    def test_reconciliation_error_returns_400(self) -> None:
        """ReconciliationError during reconcile_kserve becomes a 400."""
        stored_output = KServeInferenceResponse(
            id="recon-err",
            model_name="test-model",
            outputs=[
                KServeData(
                    name="output",
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
            new=mock.AsyncMock(
                side_effect=ReconciliationError(
                    "shape mismatch", payload_id="recon-err"
                ),
            ),
        ):
            payload = _make_request(id_="recon-err", n_rows=3, n_cols=1)
            resp = self.client.post("/", json=payload)

            assert resp.status_code == HTTPStatus.BAD_REQUEST
            assert "shape mismatch" in resp.json()["detail"]


if __name__ == "__main__":
    unittest.main()
