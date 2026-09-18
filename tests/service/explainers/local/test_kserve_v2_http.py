"""Protocol-level tests for the KServe V2 provider without network access."""

from typing import Any

import numpy as np
import pytest

from trustyai_service.service.explainers.local import kserve_v2_http as provider_module
from trustyai_service.service.explainers.local.kserve_v2_http import (
    KServeV2HttpPredictionProvider,
)
from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    KServeModelSpec,
    PredictionMetadata,
    ProviderConfigurationError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnsupportedModelError,
)
from trustyai_service.service.explainers.local.types import TaskType


class _Response:
    status_code = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _Client:
    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.timeouts: list[object] = []

    def post(self, _url: str, **kwargs: object) -> _Response:
        body = kwargs["json"]
        assert isinstance(body, dict)
        self.posts.append(body)
        self.timeouts.append(kwargs["timeout"])
        rows = body["inputs"][0]["shape"][0]
        return _Response(
            {
                "model_name": "m",
                "outputs": [
                    {
                        "name": "output",
                        "datatype": "FP32",
                        "shape": [rows, 1],
                        "data": [float(i) for i in range(rows)],
                    }
                ],
            }
        )

    def close(self) -> None:
        return None


class _FlatScalarClient(_Client):
    def post(self, _url: str, **kwargs: object) -> _Response:
        body = kwargs["json"]
        assert isinstance(body, dict)
        self.posts.append(body)
        rows = body["inputs"][0]["shape"][0]
        return _Response(
            {
                "model_name": "m",
                "outputs": [
                    {
                        "name": "output",
                        "datatype": "FP32",
                        "shape": [rows],
                        "data": [float(i) for i in range(rows)],
                    }
                ],
            }
        )


def _provider(client: _Client) -> KServeV2HttpPredictionProvider:
    spec = KServeModelSpec(
        "http://model.example", "m", None, "input", "output", TaskType.REGRESSION
    )
    return KServeV2HttpPredictionProvider(
        client,
        metadata=PredictionMetadata(
            "input", "output", "FP32", "FP32", (-1, 2), (-1, 1)
        ),
        spec=spec,
        max_batch_size=2,
    )


def test_provider_batches_and_preserves_response_rows() -> None:
    """Split oversized requests while preserving every response row."""
    client = _Client()
    provider = _provider(client)
    result = provider.predict(np.ones((5, 2), dtype=float))
    assert result.shape == (5, 1)
    assert len(client.posts) == 3
    assert [post["inputs"][0]["shape"][0] for post in client.posts] == [2, 2, 1]
    assert all(post["inputs"][0]["data"] for post in client.posts)
    assert all(post["outputs"] == [{"name": "output"}] for post in client.posts)
    assert all(
        isinstance(timeout, float) and timeout > 0 for timeout in client.timeouts
    )


def test_provider_normalizes_flat_scalar_response_shape() -> None:
    """Normalize a KServe scalar output tensor to a column matrix."""
    provider = _provider(_FlatScalarClient())
    result = provider.predict(np.ones((2, 2), dtype=float))
    assert result.shape == (2, 1)


def test_provider_rejects_redirect_response() -> None:
    """Treat an unexpected redirect as an upstream contract failure."""

    class RedirectClient:
        def post(self, _url: str, **_kwargs: object) -> object:
            return type("Response", (), {"status_code": 302})()

        def close(self) -> None:
            return None

    provider = _provider(RedirectClient())
    with pytest.raises(ProviderInvalidResponseError):
        provider.predict(np.ones((1, 2), dtype=float))


def test_provider_rejects_non_finite_output() -> None:
    """Reject NaN model outputs before they reach an explainer."""

    class NonFiniteClient(_Client):
        def post(self, _url: str, **kwargs: object) -> _Response:
            body = kwargs["json"]
            assert isinstance(body, dict)
            rows = body["inputs"][0]["shape"][0]
            return _Response(
                {
                    "model_name": "m",
                    "outputs": [
                        {
                            "name": "output",
                            "datatype": "FP32",
                            "shape": [rows, 1],
                            "data": [float("nan")] * rows,
                        }
                    ],
                }
            )

    provider = _provider(NonFiniteClient())
    with pytest.raises(ProviderInvalidResponseError):
        provider.predict(np.ones((1, 2), dtype=float))


def test_transport_config_requires_positive_batch_size() -> None:
    """Reject a non-positive inference batch size."""
    with pytest.raises(ValueError, match="max_batch_size"):
        HttpTransportConfig(headers={}, max_batch_size=0)


def test_unlisted_host_is_a_request_error_before_client_creation() -> None:
    """Reject an endpoint outside the configured deployment host allowlist."""
    spec = KServeModelSpec(
        "http://model.example", "m", None, None, None, TaskType.REGRESSION
    )
    with pytest.raises(ProviderInvalidRequestError):
        KServeV2HttpPredictionProvider.connect(
            spec,
            1,
            HttpTransportConfig(headers={}, allowed_hosts=frozenset({"other.example"})),
        )


def test_model_metadata_requires_an_output_tensor() -> None:
    """Classify omitted output metadata as an unsupported upstream model."""
    payload = {
        "name": "m",
        "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 2]}],
    }
    spec = KServeModelSpec(
        "http://model.example", "m", None, None, None, TaskType.REGRESSION
    )
    with pytest.raises(ProviderUnsupportedModelError):
        provider_module._metadata_from_payload(payload, spec)


def test_model_metadata_rejects_malformed_version_list() -> None:
    """Classify malformed version metadata as an unsupported model contract."""
    payload = {
        "name": "m",
        "versions": "v1",
        "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 2]}],
        "outputs": [{"name": "output", "datatype": "FP32", "shape": [-1, 1]}],
    }
    spec = KServeModelSpec(
        "http://model.example", "m", "v1", None, None, TaskType.REGRESSION
    )
    with pytest.raises(ProviderUnsupportedModelError):
        provider_module._metadata_from_payload(payload, spec)


def test_missing_explicit_output_selector_is_a_request_error() -> None:
    """Keep a caller-selected but absent output distinct from upstream ambiguity."""
    payload = {
        "name": "m",
        "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 2]}],
        "outputs": [{"name": "output", "datatype": "FP32", "shape": [-1, 1]}],
    }
    spec = KServeModelSpec(
        "http://model.example", "m", None, None, "missing", TaskType.REGRESSION
    )
    with pytest.raises(ProviderInvalidRequestError):
        provider_module._metadata_from_payload(payload, spec)


def test_inference_response_must_match_requested_model_version() -> None:
    """Reject an inference response served by a different requested version."""
    response = _Response(
        {
            "model_name": "m",
            "model_version": "v2",
            "outputs": [
                {
                    "name": "output",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [1.0],
                }
            ],
        }
    )
    metadata = PredictionMetadata("input", "output", "FP32", "FP32", (-1, 2), (-1, 1))
    spec = KServeModelSpec(
        "http://model.example", "m", "v1", None, None, TaskType.REGRESSION
    )
    with pytest.raises(ProviderInvalidResponseError):
        provider_module._decode_inference_response(response, metadata, spec, 1)


def test_invalid_http_client_configuration_is_not_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map malformed TLS/client construction settings to configuration errors."""

    class InvalidClient:
        def __init__(self, **_kwargs: object) -> None:
            msg = "invalid TLS settings"
            raise ValueError(msg)

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: InvalidClient)
    spec = KServeModelSpec(
        "http://model.example", "m", None, None, None, TaskType.REGRESSION
    )
    with pytest.raises(ProviderConfigurationError):
        KServeV2HttpPredictionProvider.connect(
            spec,
            1,
            HttpTransportConfig(headers={}, allowed_hosts=frozenset({"model.example"})),
        )


def test_provider_rejects_oversized_request_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce the request element cap before creating a request body."""
    client = _Client()
    provider = _provider(client)
    monkeypatch.setattr(provider_module, "_MAX_REQUEST_ELEMENTS", 1)
    with pytest.raises(ProviderInvalidRequestError):
        provider.predict(np.ones((1, 2), dtype=float))
    assert client.posts == []


def test_bounded_stream_rejects_an_oversized_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce the response byte cap while bytes are still streamed."""

    class StreamResponse:
        status_code = 200

        def iter_bytes(self) -> list[bytes]:
            return [b"{}"]

    class StreamContext:
        def __enter__(self) -> StreamResponse:
            return StreamResponse()

        def __exit__(self, *_args: object) -> None:
            return None

    class StreamClient:
        def stream(self, _method: str, _url: str, **_kwargs: object) -> StreamContext:
            return StreamContext()

        def close(self) -> None:
            return None

    monkeypatch.setattr(provider_module, "_MAX_RESPONSE_BYTES", 1)
    with pytest.raises(ProviderInvalidResponseError, match="too large"):
        provider_module._request_bounded(StreamClient(), "GET", "http://m", 1)


@pytest.mark.parametrize("status_code", [302, 400, 422])
def test_metadata_contract_statuses_are_upstream_response_errors(
    status_code: int,
) -> None:
    """Map redirects and request-contract 4xx responses to upstream errors."""

    class Client:
        def get(self, _url: str, **_kwargs: object) -> object:
            return type("Response", (), {"status_code": status_code})()

    spec = KServeModelSpec(
        "http://model.example", "m", None, None, None, TaskType.REGRESSION
    )
    with pytest.raises(ProviderInvalidResponseError):
        KServeV2HttpPredictionProvider._from_metadata(
            Client(), spec, 2, "http://model.example", 1
        )


def test_integer_inputs_must_be_integral_and_in_range() -> None:
    """Reject integer tensors containing fractional or out-of-range values."""
    client = _Client()
    spec = KServeModelSpec(
        "http://model.example", "m", None, None, None, TaskType.REGRESSION
    )
    provider = KServeV2HttpPredictionProvider(
        client,
        PredictionMetadata("input", "output", "UINT8", "FP32", (-1, 2), (-1, 1)),
        spec,
        2,
    )
    with pytest.raises(ProviderInvalidRequestError):
        provider.predict(np.array([[1.5, 2.0]]))
    with pytest.raises(ProviderInvalidRequestError):
        provider.predict(np.array([[-1.0, 2.0]]))
