"""Protocol-level tests for the KServe V2 provider without network access."""

from typing import Any

import numpy as np
import pytest

from trustyai_service.service.explainers.local.kserve_v2_http import (
    KServeV2HttpPredictionProvider,
)
from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    KServeModelSpec,
    PredictionMetadata,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
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
        "http://model.example", "m", None, None, None, TaskType.REGRESSION
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
