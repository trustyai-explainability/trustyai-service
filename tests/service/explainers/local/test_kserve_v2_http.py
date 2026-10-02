"""Protocol-level tests for the KServe V2 codec and HTTP provider."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import pytest

from trustyai_service.service.explainers.local import kserve_v2_codec as codec
from trustyai_service.service.explainers.local import kserve_v2_http as provider_module
from trustyai_service.service.explainers.local import (
    transport_config as transport_config_module,
)
from trustyai_service.service.explainers.local.kserve_v2_codec import (
    decode_response,
    encode_request,
    parse_metadata,
)
from trustyai_service.service.explainers.local.kserve_v2_http import (
    KServeModelSpec,
    KServeV2HttpPredictionProvider,
)
from trustyai_service.service.explainers.local.model_provider import (
    DependencyUnavailableError,
    HttpTransportConfig,
    PredictionMetadata,
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Response:
    """Small bounded-response double with an HTTP-like JSON boundary."""

    status_code = 200

    def __init__(self, payload: object, *, content: bytes | None = None) -> None:
        self._payload = payload
        self.content = (
            json.dumps(payload, allow_nan=True).encode() if content is None else content
        )

    def json(self) -> object:
        """Return the already-decoded response payload."""
        return self._payload


class _MalformedJsonResponse(_Response):
    """Response whose JSON parser fails at the codec boundary."""

    def json(self) -> object:
        """Raise the same parser error produced by malformed JSON."""
        message = "malformed"
        raise json.JSONDecodeError(message, "{", 0)


class _ProviderResponse:
    """HTTP response double used by provider tests."""

    def __init__(
        self, payload: object, *, status_code: int = 200, content: bytes | None = None
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = (
            json.dumps(payload, allow_nan=True).encode() if content is None else content
        )

    def json(self) -> object:
        """Return the response payload."""
        return self._payload


class _StreamingResponse:
    """Response double that exposes only an incrementally read body."""

    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {} if headers is None else headers
        self._content = content
        self.content_accessed = False
        self.closed = False
        self.chunks_read = 0

    @property
    def content(self) -> bytes:
        self.content_accessed = True
        return self._content

    def json(self) -> object:
        return json.loads(self._content)

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        del chunk_size
        for offset in range(0, len(self._content), 2):
            self.chunks_read += 1
            yield self._content[offset : offset + 2]


class _StreamContext:
    """Context manager double that records response cleanup."""

    def __init__(self, response: _StreamingResponse) -> None:
        self.response = response

    def __enter__(self) -> _StreamingResponse:
        return self.response

    def __exit__(self, *_args: object) -> None:
        self.response.closed = True


def _metadata_payload(
    *,
    name: object = "model",
    versions: object = ("v1", "v2"),
    inputs: object = None,
    outputs: object = None,
) -> dict[str, object]:
    """Build a hand-written KServe metadata fixture."""
    return {
        "name": name,
        "versions": list(versions) if isinstance(versions, tuple) else versions,
        "inputs": (
            [{"name": "features", "datatype": "FP32", "shape": [-1, 3]}]
            if inputs is None
            else inputs
        ),
        "outputs": (
            [{"name": "prediction", "datatype": "FP32", "shape": [-1, 1]}]
            if outputs is None
            else outputs
        ),
    }


def _metadata(
    *,
    input_shape: tuple[int, ...] = (-1, 3),
    output_shape: tuple[int, ...] = (-1, 1),
    input_datatype: str = "FP32",
    output_datatype: str = "FP32",
) -> PredictionMetadata:
    """Build selected metadata independently from the codec implementation."""
    return PredictionMetadata(
        input_name="features",
        output_name="prediction",
        input_datatype=input_datatype,
        output_datatype=output_datatype,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def _output_payload(  # noqa: PLR0913
    *,
    shape: list[int],
    data: object,
    name: str = "prediction",
    datatype: str = "FP32",
    model_name: object = "model",
    model_version: object = None,
) -> dict[str, object]:
    """Build a hand-written KServe inference response fixture."""
    payload: dict[str, object] = {
        "model_name": model_name,
        "outputs": [{"name": name, "datatype": datatype, "shape": shape, "data": data}],
    }
    if model_version is not None:
        payload["model_version"] = model_version
    return payload


def test_parse_metadata_selects_tensors_and_accepts_advertised_version() -> None:
    """Select the requested tensors while retaining the advertised V2 contract."""
    payload = _metadata_payload(
        inputs=[{"name": "features", "datatype": "FP32", "shape": [-1, 3]}],
        outputs=[
            {"name": "unused", "datatype": "FP32", "shape": [-1, 1]},
            {"name": "prediction", "datatype": "FP32", "shape": [-1, 1]},
        ],
    )

    result = parse_metadata(
        payload,
        "model",
        "v2",
        input_name="features",
        output_name="prediction",
    )

    assert result == PredictionMetadata(
        "features", "prediction", "FP32", "FP32", (-1, 3), (-1, 1)
    )


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        (_metadata_payload(name=None), ProviderUnsupportedModelError),
        (_metadata_payload(name="other"), ProviderUnsupportedModelError),
        (_metadata_payload(versions=["v1"]), ProviderInvalidRequestError),
        (
            _metadata_payload(
                inputs=[
                    {"name": "a", "datatype": "FP32", "shape": [-1, 3]},
                    {"name": "b", "datatype": "FP32", "shape": [-1, 2]},
                ]
            ),
            ProviderUnsupportedModelError,
        ),
        (
            _metadata_payload(
                outputs=[
                    {"name": "a", "datatype": "FP32", "shape": [-1, 1]},
                    {"name": "b", "datatype": "FP32", "shape": [-1, 1]},
                ]
            ),
            ProviderUnsupportedModelError,
        ),
        (
            _metadata_payload(
                inputs=[{"name": "features", "datatype": "BYTES", "shape": [-1, 3]}]
            ),
            ProviderUnsupportedModelError,
        ),
        (
            _metadata_payload(
                inputs=[{"name": "features", "datatype": "FP32", "shape": [-1, 3, 1]}]
            ),
            ProviderUnsupportedModelError,
        ),
        (
            _metadata_payload(
                outputs=[{"name": "prediction", "datatype": "FP32", "shape": [-1, -1]}]
            ),
            ProviderUnsupportedModelError,
        ),
        (
            _metadata_payload(
                inputs=[{"name": "features", "datatype": "FP32", "shape": [2, 3]}]
            ),
            ProviderUnsupportedModelError,
        ),
    ],
)
def test_parse_metadata_rejects_unsupported_or_mismatched_contracts(
    payload: dict[str, object], error_type: type[Exception]
) -> None:
    """Classify malformed, ambiguous, and unsupported metadata at the codec boundary."""
    with pytest.raises(error_type):
        parse_metadata(payload, "model", "v2")


@pytest.mark.parametrize("selector", ["missing", "also-missing"])
def test_parse_metadata_rejects_absent_explicit_tensor_selectors(selector: str) -> None:
    """Keep a caller-selected but absent input or output distinct from ambiguity."""
    kwargs = (
        {"input_name": selector} if selector == "missing" else {"output_name": selector}
    )

    with pytest.raises(ProviderInvalidRequestError):
        parse_metadata(_metadata_payload(), "model", "v2", **kwargs)


def test_encode_request_uses_flattened_row_major_data_and_selected_output() -> None:
    """Encode a two-dimensional batch with the advertised V2 shape and order."""
    result = encode_request(
        _metadata(input_shape=(-1, 2)),
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
    )

    assert result == {
        "inputs": [
            {
                "name": "features",
                "shape": [2, 2],
                "datatype": "FP32",
                "data": [1.0, 2.0, 3.0, 4.0],
            }
        ],
        "outputs": [{"name": "prediction"}],
    }


def test_encode_request_casts_integral_values_to_python_integers() -> None:
    """Encode integral JSON values as integers for an integer tensor."""
    result = encode_request(
        _metadata(input_shape=(-1, 2), input_datatype="INT32"),
        np.asarray([[1.0, -2.0]]),
    )

    tensor = cast("dict[str, object]", cast("list[object]", result["inputs"])[0])
    data = cast("list[object]", tensor["data"])
    assert data == [1, -2]
    assert all(type(value) is int for value in data)


def test_encode_request_casts_zero_one_values_to_python_booleans() -> None:
    """Encode zero/one JSON values as booleans for a BOOL tensor."""
    result = encode_request(
        _metadata(input_shape=(-1, 3), input_datatype="BOOL"),
        np.asarray([[0.0, 1.0, 0.0]]),
    )

    tensor = cast("dict[str, object]", cast("list[object]", result["inputs"])[0])
    data = cast("list[object]", tensor["data"])
    assert data == [False, True, False]
    assert all(type(value) is bool for value in data)


def test_encode_request_rejects_unmatched_output_selector() -> None:
    """Do not let a caller override the negotiated output tensor name."""
    with pytest.raises(ProviderInvalidRequestError, match="output"):
        encode_request(
            _metadata(input_shape=(-1, 2)),
            np.asarray([[1.0, 2.0]]),
            output_name="other",
        )


@pytest.mark.parametrize(
    ("input_shape", "values", "expected_shape"),
    [
        ((-1,), [[1.0], [2.0]], [2]),
        ((3,), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [2, 3]),
        ((-1, 3), [[1.0, 2.0, 3.0]], [1, 3]),
        ((1, 3), [[1.0, 2.0, 3.0]], [1, 3]),
    ],
)
def test_encode_request_supports_rank_one_and_rank_two_layouts(
    input_shape: tuple[int, ...], values: list[list[float]], expected_shape: list[int]
) -> None:
    """Prepend dynamic batches while treating rank-one scalar metadata specially."""
    result = encode_request(_metadata(input_shape=input_shape), np.asarray(values))

    assert result["inputs"][0]["shape"] == expected_shape  # type: ignore[index]


def test_encode_request_rejects_width_and_fixed_batch_mismatches() -> None:
    """Reject requests that cannot be represented by the negotiated input tensor."""
    with pytest.raises(ProviderInvalidRequestError):
        encode_request(_metadata(input_shape=(-1, 3)), np.ones((2, 2)))

    with pytest.raises(ProviderInvalidRequestError):
        encode_request(_metadata(input_shape=(1, 3)), np.ones((2, 3)))


def test_encode_request_rejects_non_finite_values_and_boolean_range() -> None:
    """Reject unsafe numeric inputs before serializing a request body."""
    with pytest.raises(ProviderInvalidRequestError):
        encode_request(_metadata(), np.asarray([[1.0, np.nan, 3.0]]))

    with pytest.raises(ProviderInvalidRequestError):
        encode_request(
            _metadata(input_shape=(-1, 2), input_datatype="BOOL"),
            np.asarray([[0.0, 2.0]]),
        )


@pytest.mark.parametrize("datatype", ["FP32", "INT32"])
def test_encode_request_rejects_boolean_arrays_for_non_bool_datatypes(
    datatype: str,
) -> None:
    """Do not coerce NumPy booleans into non-BOOL model inputs."""
    with pytest.raises(ProviderInvalidRequestError, match="Boolean"):
        encode_request(
            _metadata(input_shape=(-1, 2), input_datatype=datatype),
            np.asarray([[True, False]], dtype=np.bool_),
        )


def test_encode_request_rejects_excessive_element_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce the request element cap before scanning values for finiteness."""
    monkeypatch.setattr(codec, "_MAX_REQUEST_ELEMENTS", 1)

    with pytest.raises(ProviderInvalidRequestError, match="too large"):
        encode_request(
            _metadata(input_shape=(-1, 2)),
            np.asarray([[np.nan, 1.0]]),
        )


def test_encode_request_rejects_unvalidated_metadata_contracts() -> None:
    """Keep direct codec callers from bypassing negotiated shape and dtype checks."""
    with pytest.raises(ProviderUnsupportedModelError):
        encode_request(_metadata(input_shape=(-1, 2, 1)), np.ones((1, 2)))

    with pytest.raises(ProviderInvalidRequestError):
        encode_request(
            _metadata(input_datatype="BYTES"),
            np.ones((1, 3)),
        )


@pytest.mark.parametrize(
    ("shape", "data"),
    [
        ([2, 2], [[1.0, 2.0], [3.0, 4.0]]),
        ([2, 2], [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test_decode_response_accepts_nested_or_flattened_row_major_data(
    shape: list[int], data: object
) -> None:
    """Decode both JSON tensor data representations without changing row order."""
    result = decode_response(
        _Response(_output_payload(shape=shape, data=data)),
        _metadata(output_shape=(-1, 2)),
        "model",
        None,
        2,
    )

    np.testing.assert_allclose(result, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    ("shape", "data"),
    [([2], [1.0, 2.0]), ([2, 1], [[1.0], [2.0]]), ([2, 1], [1.0, 2.0])],
)
def test_decode_response_normalizes_scalar_outputs_to_a_column(
    shape: list[int], data: object
) -> None:
    """Normalize scalar output tensors, regardless of their valid wire shape."""
    result = decode_response(
        _Response(_output_payload(shape=shape, data=data)),
        _metadata(output_shape=(-1, 1)),
        "model",
        None,
        2,
    )

    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, [[1.0], [2.0]])


def test_decode_response_preserves_wider_matrix_outputs() -> None:
    """Retain one output column per class or regression target."""
    result = decode_response(
        _Response(_output_payload(shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        _metadata(output_shape=(-1, 3)),
        "model",
        None,
        2,
    )

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        (
            _output_payload(shape=[2, 1], data=[1.0, 2.0], name="other"),
            ProviderInvalidResponseError,
        ),
        (
            {
                "model_name": "model",
                "outputs": [
                    {
                        "name": "prediction",
                        "datatype": "FP32",
                        "shape": [2, 1],
                        "data": [1.0],
                    },
                    {
                        "name": "prediction",
                        "datatype": "FP32",
                        "shape": [2, 1],
                        "data": [2.0],
                    },
                ],
            },
            ProviderInvalidResponseError,
        ),
        (
            _output_payload(shape=[2, 1], data=[1.0, 2.0], datatype="INT32"),
            ProviderInvalidResponseError,
        ),
        (
            _output_payload(shape=[2, 1], data=[float("nan"), 2.0]),
            ProviderInvalidResponseError,
        ),
        (
            _output_payload(shape=[4], data=[1.0, 2.0, 3.0, 4.0]),
            ProviderInvalidResponseError,
        ),
    ],
)
def test_decode_response_rejects_invalid_selected_output_contracts(
    payload: dict[str, object], error_type: type[Exception]
) -> None:
    """Reject missing, duplicate, mismatched, non-finite, and flattened shapes."""
    with pytest.raises(error_type):
        decode_response(_Response(payload), _metadata(), "model", None, 2)


def test_decode_response_rejects_wrong_model_identity_and_version() -> None:
    """Prevent a response from a different model or requested version being consumed."""
    with pytest.raises(ProviderInvalidResponseError):
        decode_response(
            _Response(_output_payload(shape=[1, 1], data=[1.0], model_name="other")),
            _metadata(),
            "model",
            None,
            1,
        )


def test_decode_response_enforces_fixed_singleton_batches_and_metadata_shape() -> None:
    """Do not accept a response batch that contradicts fixed or higher-rank metadata."""
    with pytest.raises(ProviderInvalidResponseError):
        decode_response(
            _Response(_output_payload(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])),
            _metadata(output_shape=(1, 2)),
            "model",
            None,
            2,
        )

    with pytest.raises(ProviderUnsupportedModelError):
        decode_response(
            _Response(_output_payload(shape=[1, 2], data=[1.0, 2.0])),
            _metadata(output_shape=(-1, 2, 1)),
            "model",
            None,
            1,
        )

    with pytest.raises(ProviderInvalidResponseError):
        decode_response(
            _Response(_output_payload(shape=[1, 1], data=[1.0], model_version="v2")),
            _metadata(),
            "model",
            "v1",
            1,
        )


def test_decode_response_rejects_excessive_elements_before_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the response element cap before NumPy conversion or reshaping."""
    monkeypatch.setattr(codec, "_MAX_RESPONSE_ELEMENTS", 1)

    with pytest.raises(ProviderInvalidResponseError, match="too large"):
        decode_response(
            _Response(_output_payload(shape=[2, 1], data=[1.0, 2.0])),
            _metadata(),
            "model",
            None,
            2,
        )


def test_decode_response_rejects_oversized_body_before_json_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the byte cap before invoking the response JSON parser."""
    monkeypatch.setattr(codec, "_MAX_RESPONSE_BYTES", 1)
    response = _MalformedJsonResponse({}, content=b"{}")

    with pytest.raises(ProviderInvalidResponseError, match="too large"):
        decode_response(response, _metadata(), "model", None, 1)


def test_decode_response_maps_malformed_json_to_provider_error() -> None:
    """Do not expose JSON parser exceptions from the codec boundary."""
    with pytest.raises(ProviderInvalidResponseError):
        decode_response(_MalformedJsonResponse({}), _metadata(), "model", None, 1)


def test_decode_response_uses_direction_specific_integer_validation() -> None:
    """Reject fractional or out-of-range integer values as invalid responses."""
    with pytest.raises(ProviderInvalidResponseError):
        decode_response(
            _Response(_output_payload(shape=[1, 1], data=[1.5], datatype="UINT8")),
            _metadata(output_datatype="UINT8"),
            "model",
            None,
            1,
        )

    with pytest.raises(ProviderInvalidResponseError):
        decode_response(
            _Response(_output_payload(shape=[1, 1], data=[-1.0], datatype="UINT8")),
            _metadata(output_datatype="UINT8"),
            "model",
            None,
            1,
        )


@pytest.mark.parametrize("datatype", ["FP32", "INT32"])
def test_decode_response_rejects_boolean_values_for_non_bool_datatypes(
    datatype: str,
) -> None:
    """Do not coerce JSON booleans into non-BOOL model outputs."""
    with pytest.raises(ProviderInvalidResponseError, match="Boolean"):
        decode_response(
            _Response(
                _output_payload(
                    shape=[1, 1],
                    data=[True],
                    datatype=datatype,
                )
            ),
            _metadata(output_datatype=datatype),
            "model",
            None,
            1,
        )


def test_codec_import_does_not_load_http_or_endpoint_integrations() -> None:
    """Keep the codec safe to import without optional HTTP or endpoint packages."""
    repository_root = Path(__file__).parents[4]
    script = """
import builtins
import sys

sys.path.insert(0, "src")
forbidden = (
    "httpx2",
    "fastapi",
    "lime",
    "shap",
    "trustyai_service.endpoints",
    "trustyai_service.service.data.storage",
)
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == module or name.startswith(module + ".") for module in forbidden):
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import trustyai_service.service.explainers.local.kserve_v2_codec  # noqa: E402
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_provider_import_does_not_load_optional_http_dependency() -> None:
    """Import the provider safely when the optional HTTP package is absent."""
    repository_root = Path(__file__).parents[4]
    script = """
import builtins
import sys

sys.path.insert(0, "src")
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "httpx2" or name.startswith("httpx2."):
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import trustyai_service.service.explainers.local.kserve_v2_http  # noqa: E402
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _provider_spec(
    *,
    base_url: str = "http://model.example/prefix/",
    model_name: str = "model",
    model_version: str | None = None,
    input_name: str | None = None,
    output_name: str | None = None,
) -> KServeModelSpec:
    """Build a provider specification for unit tests."""
    return KServeModelSpec(
        base_url,
        model_name,
        model_version,
        input_name,
        output_name,
    )


def _provider_transport(
    *,
    headers: dict[str, str] | None = None,
    allowed_hosts: frozenset[str] | None = None,
    max_batch_size: int = 1024,
) -> HttpTransportConfig:
    """Build deployment-owned transport settings for provider tests."""
    return HttpTransportConfig(
        headers={} if headers is None else headers,
        allowed_hosts=(
            frozenset({"model.example"}) if allowed_hosts is None else allowed_hosts
        ),
        max_batch_size=max_batch_size,
    )


def test_connect_loads_one_client_negotiates_metadata_and_closes_idempotently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negotiate typed metadata once and close the owned client only once."""

    class Client:
        instances: ClassVar[list[Client]] = []

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.get_calls: list[tuple[str, dict[str, object]]] = []
            self.close_calls = 0
            type(self).instances.append(self)

        def get(self, url: str, **kwargs: object) -> _ProviderResponse:
            self.get_calls.append((url, kwargs))
            return _ProviderResponse(_metadata_payload())

        def close(self) -> None:
            self.close_calls += 1

    Client.instances = []
    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(),
        _provider_transport(headers={"X-Deployment": "yes"}),
        timeout_seconds=2.0,
    )

    client = Client.instances[0]
    assert provider.metadata == PredictionMetadata(
        "features", "prediction", "FP32", "FP32", (-1, 3), (-1, 1)
    )
    assert client.kwargs == {
        "headers": {"X-Deployment": "yes"},
        "verify": True,
        "cert": None,
        "follow_redirects": False,
        "trust_env": False,
        "timeout": 2.0,
    }
    assert client.get_calls[0][0] == "http://model.example/prefix/v2/models/model"
    assert 0 < client.get_calls[0][1]["timeout"] <= 2.0

    provider.close()
    provider.close()
    assert client.close_calls == 1


def test_connect_passes_resolved_address_to_httpx2_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the production client while retaining the model hostname in its URL."""

    class Client:
        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def close(self) -> None:
            return None

    Client.__module__ = "httpx2"
    captured: dict[str, object] = {}

    def create_client(
        client_type: type,
        transport: HttpTransportConfig,
        timeout_seconds: float,
        *,
        pinned_address: str | tuple[str, ...] | None = None,
    ) -> Client:
        captured.update(
            {
                "client_type": client_type,
                "transport": transport,
                "timeout": timeout_seconds,
                "pinned_address": pinned_address,
            }
        )
        return Client()

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    monkeypatch.setattr(
        provider_module,
        "resolve_outbound_addresses",
        lambda _url, **_kwargs: ("93.184.216.34",),
    )
    monkeypatch.setattr(provider_module, "_create_client", create_client)

    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(),
        _provider_transport(),
        timeout_seconds=2.0,
    )
    provider.close()

    assert captured["client_type"] is Client
    assert captured["pinned_address"] == ("93.184.216.34",)


def test_httpx2_transport_uses_direct_connections_and_deadline_backend() -> None:
    """Keep the direct httpx2/httpcore2 integration seam explicit."""
    httpx2 = pytest.importorskip("httpx2")
    client = provider_module._create_client(
        httpx2.Client,
        _provider_transport(),
        timeout_seconds=2.0,
    )

    try:
        transport = getattr(client, "_" + "transport")
        assert isinstance(transport, provider_module._DirectHTTPTransport)
        assert isinstance(
            getattr(transport, "_" + "network_backend"),
            provider_module._DeadlineNetworkBackend,
        )
        assert not hasattr(transport, "_" + "pool")
    finally:
        client.close()


@pytest.mark.parametrize(
    ("family", "sockaddr"),
    [
        (provider_module.socket.AF_INET, ("10.0.0.4", 8080)),
        (provider_module.socket.AF_INET, ("169.254.169.254", 8080)),
        (provider_module.socket.AF_INET6, ("::1", 8080, 0, 0)),
        (provider_module.socket.AF_INET6, ("fc00::4", 8080, 0, 0)),
    ],
)
def test_hostname_resolution_rejects_restricted_addresses(
    monkeypatch: pytest.MonkeyPatch,
    family: int,
    sockaddr: tuple[object, ...],
) -> None:
    """Reject private, metadata, loopback, and unique-local DNS results."""
    monkeypatch.setattr(
        provider_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                family,
                provider_module.socket.SOCK_STREAM,
                6,
                "",
                sockaddr,
            )
        ],
    )

    with pytest.raises(ProviderInvalidRequestError, match="restricted"):
        provider_module.resolve_outbound_addresses("http://model.example:8080")


def test_hostname_resolution_pins_a_public_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a hostname once so later connections use the same public address."""
    address = "93.184.216.34"
    monkeypatch.setattr(
        provider_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                provider_module.socket.AF_INET,
                provider_module.socket.SOCK_STREAM,
                6,
                "",
                (address, 8080),
            )
        ],
    )

    assert provider_module.resolve_outbound_addresses("http://model.example:8080") == (
        address,
    )


def test_hostname_resolution_returns_all_safe_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain all safe addresses so connection setup can fail over."""
    addresses = ("93.184.216.34", "93.184.216.35")
    monkeypatch.setattr(
        provider_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                provider_module.socket.AF_INET,
                provider_module.socket.SOCK_STREAM,
                6,
                "",
                (address, 8443),
            )
            for address in addresses
        ],
    )

    assert (
        provider_module.resolve_outbound_addresses("https://model.example:8443")
        == addresses
    )


def test_hostname_resolution_allows_explicit_private_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow an explicitly configured in-cluster host to use private addresses."""
    monkeypatch.setattr(
        provider_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                provider_module.socket.AF_INET,
                provider_module.socket.SOCK_STREAM,
                6,
                "",
                ("10.0.0.4", 8443),
            )
        ],
    )

    assert provider_module.resolve_outbound_addresses(
        "https://model.namespace.svc.cluster.local:8443",
        allow_private_addresses=True,
    ) == ("10.0.0.4",)


def test_hostname_resolution_honors_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a deadline error without retaining the local-explainer worker."""
    release = threading.Event()

    def blocked_resolution(*_args: object, **_kwargs: object) -> list[object]:
        release.wait(timeout=1)
        return []

    monkeypatch.setattr(provider_module.socket, "getaddrinfo", blocked_resolution)
    try:
        with pytest.raises(ProviderDeadlineError):
            provider_module.resolve_outbound_addresses(
                "https://model.example:8443",
                timeout_seconds=0.01,
            )
    finally:
        release.set()


def test_hostname_resolution_rejects_when_resolver_capacity_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not enqueue unlimited DNS work behind stalled resolver threads."""

    class NoCapacity:
        def acquire(self, **_kwargs: object) -> bool:
            return False

    monkeypatch.setattr(transport_config_module, "_DNS_RESOLVER_SLOTS", NoCapacity())

    with pytest.raises(ProviderUnavailableError, match="DNS resolution capacity"):
        provider_module.resolve_outbound_addresses("https://model.example:8443")


def test_deadline_backend_connects_to_pinned_address() -> None:
    """Keep the original hostname for HTTP while pinning the TCP destination."""

    class Backend:
        address: str | None = None

        def connect_tcp(
            self,
            host: str,
            _port: int,
            **_kwargs: object,
        ) -> object:
            self.address = host
            return object()

    backend = Backend()
    wrapped = provider_module._DeadlineNetworkBackend(
        backend,
        pinned_address="93.184.216.34",
    )

    wrapped.connect_tcp("model.example", 8080)

    assert backend.address == "93.184.216.34"


def test_deadline_backend_falls_back_to_next_pinned_address() -> None:
    """Try the next validated address when the first connection fails."""

    class Backend:
        def __init__(self) -> None:
            self.addresses: list[str] = []

        def connect_tcp(
            self,
            host: str,
            _port: int,
            **_kwargs: object,
        ) -> object:
            self.addresses.append(host)
            if host == "93.184.216.34":
                raise OSError from None
            return object()

    backend = Backend()
    wrapped = provider_module._DeadlineNetworkBackend(
        backend,
        pinned_address=("93.184.216.34", "93.184.216.35"),
    )

    wrapped.connect_tcp("model.example", 8443)

    assert backend.addresses == ["93.184.216.34", "93.184.216.35"]


def test_direct_transport_releases_connection_if_request_construction_fails() -> None:
    """Close and forget a connection when httpcore request construction fails."""
    httpx2 = pytest.importorskip("httpx2")
    httpcore2 = pytest.importorskip("httpcore2")

    class Connection:
        instances: ClassVar[list[Connection]] = []

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            type(self).instances.append(self)

        def handle_request(self, _request: object) -> object:
            pytest.fail("request dispatch must not start")

        def close(self) -> None:
            self.close_calls += 1

    class CoreRequest:
        def __init__(self, **_kwargs: object) -> None:
            message = "synthetic request construction failure"
            raise ValueError(message)

    class Httpcore:
        URL = httpcore2.URL
        HTTPConnection = Connection
        Request = CoreRequest

    transport = provider_module._DirectHTTPTransport(
        httpx2,
        Httpcore,
        ssl_context=object(),
        network_backend=cast("provider_module._NetworkBackend", object()),
    )

    with pytest.raises(ValueError, match="request construction"):
        transport.handle_request(httpx2.Request("GET", "http://model.example"))

    assert len(Connection.instances) == 1
    assert Connection.instances[0].close_calls == 1
    assert transport._connections == set()


def test_connect_rejects_oversized_streamed_metadata_before_materializing(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop reading an unknown-length response after the bounded prefix."""
    monkeypatch.setattr(provider_module, "_MAX_RESPONSE_BYTES", 4)

    class Response:
        status_code = 200

        def __init__(self) -> None:
            self.chunks_seen = 0
            self.full_body_materialized = False
            self.closed = False

        @property
        def content(self) -> bytes:
            pytest.fail("the streamed body must not be materialized")

        def json(self) -> object:
            pytest.fail("oversized body must be rejected before JSON parsing")

        def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            self.chunks_seen += 1
            yield b"abcd"
            self.chunks_seen += 1
            yield b"e"
            self.full_body_materialized = True
            yield b"the rest of the hostile response"

    class Context:
        def __init__(self, response: Response) -> None:
            self.response = response

        def __enter__(self) -> Response:
            return self.response

        def __exit__(self, *_args: object) -> None:
            self.response.closed = True

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.response = Response()
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> Context:
            return Context(self.response)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderInvalidResponseError, match="too large"):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert Client.instance.response.chunks_seen == 2
    assert not Client.instance.response.full_body_materialized
    assert not Client.instance.response.closed
    assert Client.instance.close_calls == 1


def test_connect_rejects_declared_oversized_response_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a declared oversized body before consuming any response chunks."""
    monkeypatch.setattr(provider_module, "_MAX_RESPONSE_BYTES", 4)

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.response = _StreamingResponse(
                b"body",
                headers={"Content-Length": "5"},
            )
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> _StreamContext:
            return _StreamContext(self.response)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderInvalidResponseError, match="too large"):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert Client.instance.response.chunks_read == 0
    assert not Client.instance.response.closed
    assert Client.instance.close_calls == 1


def test_connect_preserves_metadata_body_error_when_stream_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep invalid metadata primary when its response cleanup raises."""

    class Response(_StreamingResponse):
        def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            yield cast("bytes", "not bytes")

    class Context(_StreamContext):
        def __exit__(self, *_args: object) -> None:
            raise ConnectionError from None

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.response = Response(b"ignored")
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> Context:
            return Context(self.response)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderInvalidResponseError, match="invalid content"):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


def test_streamed_httpx2_decoding_error_maps_to_invalid_response() -> None:
    """Classify a decoder failure while constructing the body iterator."""
    httpx2 = pytest.importorskip("httpx2")

    class Response(_StreamingResponse):
        def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            message = "invalid gzip response"
            raise httpx2.DecodingError(message)

    response = Response(
        b"ignored",
        headers={"Content-Encoding": "gzip"},
    )

    with pytest.raises(ProviderInvalidResponseError, match="invalid content"):
        provider_module._read_streamed_response(
            response,
            deadline=time.monotonic() + 1.0,
        )


@pytest.mark.parametrize("exception_type", [ValueError, TypeError])
def test_streamed_iterator_construction_errors_map_to_invalid_response(
    exception_type: type[Exception],
) -> None:
    """Classify ordinary iterator-construction failures as invalid content."""

    class Response(_StreamingResponse):
        def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            message = "iterator construction failed"
            raise exception_type(message)

    with pytest.raises(ProviderInvalidResponseError, match="invalid content"):
        provider_module._read_streamed_response(
            Response(b"ignored"),
            deadline=time.monotonic() + 1.0,
        )


def test_streamed_primary_error_skips_blocking_response_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not invoke response cleanup that can block after a body error."""

    class Response:
        status_code = 200
        headers: ClassVar[dict[str, str]] = {}

        def __init__(self) -> None:
            self.close_started = threading.Event()
            self.reclaimed_by_client = False

        def close(self) -> None:
            self.close_started.set()
            threading.Event().wait()

        def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            yield cast("bytes", "not bytes")

    class Context:
        def __init__(self, response: Response) -> None:
            self.response = response

        def __enter__(self) -> Response:
            return self.response

        def __exit__(self, *_args: object) -> None:
            self.response.close()

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.response = Response()
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> Context:
            return Context(self.response)

        def close(self) -> None:
            self.close_calls += 1
            self.response.reclaimed_by_client = True

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    started = time.monotonic()

    with pytest.raises(ProviderInvalidResponseError, match="invalid content"):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=0.05
        )

    assert time.monotonic() - started < 0.15
    assert Client.instance is not None
    assert not Client.instance.response.close_started.wait(timeout=0.1)
    assert Client.instance.response.reclaimed_by_client
    assert Client.instance.close_calls == 1


def test_fallback_nonblocking_cleanup_capability_releases_after_body_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use an explicit fallback cleanup capability without invoking context exit."""

    class Response(_StreamingResponse):
        def __init__(self) -> None:
            super().__init__(b"ignored")
            self.nonblocking_close_calls = 0
            self.context_exit_calls = 0

        def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            yield cast("bytes", "not bytes")

    class Context(_StreamContext):
        def __init__(self, response: Response) -> None:
            super().__init__(response)
            self.response = response

        def __exit__(self, *_args: object) -> None:
            self.response.context_exit_calls += 1

        def close_nonblocking(self) -> None:
            self.response.nonblocking_close_calls += 1

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.response = Response()
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> Context:
            return Context(self.response)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderInvalidResponseError, match="invalid content"):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert Client.instance.response.nonblocking_close_calls == 1
    assert Client.instance.response.context_exit_calls == 0
    assert Client.instance.close_calls == 1


@pytest.mark.parametrize("failure", ["body", "status", "decoding"])
def test_predict_retires_fallback_client_after_inference_failure(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Retire generic streamed clients without waiting on response cleanup."""
    decoding_error: BaseException | None = None
    if failure == "decoding":
        httpx2 = pytest.importorskip("httpx2")
        decoding_error = httpx2.DecodingError("invalid compressed response")
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    client_closed = threading.Event()
    metadata_body = json.dumps(_metadata_payload()).encode()

    class InferenceResponse(_StreamingResponse):
        def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            if failure == "body":
                yield cast("bytes", "not bytes")
            elif failure == "decoding":
                assert decoding_error is not None
                raise decoding_error
            else:
                yield b"{}"

    class InferenceContext(_StreamContext):
        def __exit__(self, *_args: object) -> None:
            cleanup_started.set()
            release_cleanup.wait(timeout=1.0)
            self.response.closed = True

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            type(self).instance = self

        def stream(self, method: str, _url: str, **_kwargs: object) -> _StreamContext:
            if method == "GET":
                return _StreamContext(_StreamingResponse(metadata_body))
            status_code = 400 if failure == "status" else 200
            return InferenceContext(InferenceResponse(b"{}", status_code=status_code))

        def close(self) -> None:
            self.close_calls += 1
            client_closed.set()

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )
    errors: list[BaseException] = []

    def run_prediction() -> None:
        try:
            provider.predict(np.ones((1, 3)), timeout_seconds=1.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=run_prediction, daemon=True)
    started = time.monotonic()
    try:
        worker.start()
        assert client_closed.wait(timeout=0.2)
        worker.join(timeout=0.2)
        assert not worker.is_alive()
        assert time.monotonic() - started < 0.4
        assert cleanup_started.is_set() is False
        assert len(errors) == 1
        assert isinstance(errors[0], ProviderInvalidResponseError)
        assert Client.instance is not None
        assert Client.instance.close_calls == 1
        provider.close()
        assert Client.instance.close_calls == 1
    finally:
        release_cleanup.set()
        worker.join(timeout=1.0)


def test_connect_maps_metadata_cleanup_failure_without_primary_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map a streamed cleanup-only connection failure as unavailable."""
    body = json.dumps(_metadata_payload()).encode()

    class Context(_StreamContext):
        def __exit__(self, *_args: object) -> None:
            raise ConnectionError from None

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.response = _StreamingResponse(body)
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> Context:
            return Context(self.response)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderUnavailableError):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


def test_predict_preserves_inference_deadline_when_stream_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an inference deadline primary when response cleanup raises."""
    metadata_body = json.dumps(_metadata_payload()).encode()

    class InferenceResponse(_StreamingResponse):
        def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            raise TimeoutError from None
            yield b"unreachable"

    class InferenceContext(_StreamContext):
        def __exit__(self, *_args: object) -> None:
            raise ConnectionError from None

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            self.metadata_response = _StreamingResponse(metadata_body)
            self.inference_response = InferenceResponse(b"ignored")
            type(self).instance = self

        def stream(
            self, method: str, _url: str, **_kwargs: object
        ) -> _StreamContext | InferenceContext:
            if method == "GET":
                return _StreamContext(self.metadata_response)
            return InferenceContext(self.inference_response)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )

    with pytest.raises(ProviderDeadlineError, match="deadline"):
        provider.predict(np.ones((1, 3)))
    provider.close()

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


def test_connect_maps_overflow_content_length_to_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify an arbitrarily long decimal Content-Length as invalid input."""

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.response = _StreamingResponse(
                b"{}",
                headers={"Content-Length": "1" * 5000},
            )
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> _StreamContext:
            return _StreamContext(self.response)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderInvalidResponseError, match="content length"):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert not Client.instance.response.closed
    assert Client.instance.close_calls == 1


def test_deadline_network_stream_bounds_each_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use decreasing per-operation timeouts without any shared socket abort."""
    clock = [100.0]
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])

    class Stream:
        def __init__(self) -> None:
            self.read_timeouts: list[float | None] = []
            self.write_timeouts: list[float | None] = []
            self.close_calls = 0

        def read(self, _max_bytes: int, timeout: float | None = None) -> bytes:
            self.read_timeouts.append(timeout)
            return b""

        def write(self, _buffer: bytes, timeout: float | None = None) -> None:
            self.write_timeouts.append(timeout)

        def close(self) -> None:
            self.close_calls += 1

        def start_tls(
            self,
            _ssl_context: object,
            server_hostname: str | None = None,
            timeout: float | None = None,
        ) -> Stream:
            del server_hostname, timeout
            return self

        def get_extra_info(self, _info: str) -> None:
            return None

    stream = Stream()
    wrapped = provider_module._DeadlineNetworkStream(stream)
    token = provider_module._REQUEST_DEADLINE.set(101.0)
    try:
        wrapped.read(1, timeout=5.0)
        wrapped.write(b"x", timeout=5.0)
        assert stream.read_timeouts == [1.0]
        assert stream.write_timeouts == [1.0]
        clock[0] = 101.0
        with pytest.raises(TimeoutError):
            wrapped.read(1, timeout=5.0)
    finally:
        provider_module._REQUEST_DEADLINE.reset(token)

    wrapped.close()
    assert stream.close_calls == 1


def test_streamed_metadata_and_inference_responses_are_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode normal metadata and inference bodies without materializing raw responses."""
    metadata_body = json.dumps(_metadata_payload()).encode()
    inference_body = json.dumps(_output_payload(shape=[1, 1], data=[2.5])).encode()

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.responses: list[_StreamingResponse] = []
            self.calls: list[tuple[str, dict[str, object]]] = []
            self.close_calls = 0
            type(self).instance = self

        def stream(self, method: str, _url: str, **kwargs: object) -> _StreamContext:
            self.calls.append((method, kwargs))
            body = metadata_body if method == "GET" else inference_body
            response = _StreamingResponse(
                body,
                headers={"Content-Length": str(len(body))},
            )
            self.responses.append(response)
            return _StreamContext(response)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )

    try:
        result = provider.predict(np.ones((1, 3)))
    finally:
        provider.close()

    assert Client.instance is not None
    assert [method for method, _kwargs in Client.instance.calls] == ["GET", "POST"]
    assert all(response.closed for response in Client.instance.responses)
    assert all(not response.content_accessed for response in Client.instance.responses)
    np.testing.assert_allclose(result, [[2.5]])
    assert Client.instance.close_calls == 1


@pytest.mark.parametrize(
    ("exception", "error_type"),
    [
        (TimeoutError("slow"), ProviderDeadlineError),
        (ConnectionError("refused"), ProviderUnavailableError),
    ],
)
def test_stream_metadata_transport_failures_map_to_typed_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    error_type: type[Exception],
) -> None:
    """Map failures raised while opening a streamed metadata response."""

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> _StreamContext:
            raise exception

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(error_type):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


@pytest.mark.parametrize(
    ("exception", "error_type"),
    [
        (TimeoutError("slow"), ProviderDeadlineError),
        (ConnectionError("refused"), ProviderUnavailableError),
    ],
)
def test_stream_inference_transport_failures_map_to_typed_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    error_type: type[Exception],
) -> None:
    """Map failures raised while opening a streamed inference response."""

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            type(self).instance = self

        def stream(self, method: str, _url: str, **_kwargs: object) -> _StreamContext:
            if method == "POST":
                raise exception
            body = json.dumps(_metadata_payload()).encode()
            return _StreamContext(_StreamingResponse(body))

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )
    with pytest.raises(error_type):
        provider.predict(np.ones((1, 3)))
    provider.close()

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


def test_missing_http_dependency_maps_to_dependency_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classify an absent optional client dependency at connect time."""

    def missing_dependency() -> type:
        message = "httpx2"
        raise ModuleNotFoundError(message)

    monkeypatch.setattr(provider_module, "_load_http_client", missing_dependency)

    with pytest.raises(DependencyUnavailableError) as raised:
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert raised.value.code == "dependency_unavailable"


def test_connect_preserves_prefix_and_encodes_model_and_version_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the ingress prefix while quoting model and version path segments."""

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.urls: list[str] = []
            type(self).instance = self

        def get(self, url: str, **_kwargs: object) -> _ProviderResponse:
            self.urls.append(url)
            return _ProviderResponse(
                _metadata_payload(name="model name", versions=["version 1"])
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(
            base_url="http://model.example/ingress/root/",
            model_name="model name",
            model_version="version 1",
        ),
        _provider_transport(),
        timeout_seconds=2.0,
    )
    provider.close()

    assert Client.instance is not None
    assert Client.instance.urls == [
        "http://model.example/ingress/root/v2/models/model%20name/versions/version%201"
    ]


@pytest.mark.parametrize(
    ("model_name", "model_version"),
    [
        ("", None),
        (".", None),
        ("..", None),
        ("model/name", None),
        ("model\\name", None),
        ("model%name", None),
        ("model", ""),
        ("model", "."),
        ("model", "../version"),
        ("model", "version\\one"),
        ("model", "version%one"),
    ],
)
def test_connect_rejects_unsafe_model_identity_segments(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    model_version: str | None,
) -> None:
    """Reject model identities that cannot be safely represented as path segments."""

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("client created for an unsafe model identity")

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderInvalidRequestError):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(
                model_name=model_name,
                model_version=model_version,
            ),
            _provider_transport(),
            timeout_seconds=2.0,
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "model.example:8080",
        "http://user:password@model.example",  # pragma: allowlist secret
        "http://model.example:",
        "http://model.example:0",
        "http://model.example:65536",
        "http://model.example:bad-port",
        "http://model.example?secret=value",
        "http://model.example?",
        "http://model.example#fragment",
        "http://model.example#",
        "http://model.example/%2e%2e/other",
        "http://model.example/path/../other",
        "http://model.example/path\\other",
        "http://model%.example",
    ],
)
def test_connect_rejects_unsafe_base_url_authorities_and_paths(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    """Reject authorities and prefixes that could escape the configured endpoint."""
    client_created = False

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal client_created
            client_created = True

        def close(self) -> None:
            return None

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderInvalidRequestError):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(base_url=base_url),
            _provider_transport(),
            timeout_seconds=2.0,
        )

    assert not client_created


def test_connect_enforces_nonempty_host_allowlist_before_client_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed for missing policy and reject unlisted authorities."""
    client_created = False

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal client_created
            client_created = True

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderConfigurationError, match="allowlist"):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(),
            HttpTransportConfig(headers={}),
            timeout_seconds=2.0,
        )

    with pytest.raises(ProviderInvalidRequestError, match="allowlist"):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(),
            _provider_transport(allowed_hosts=frozenset({"other.example"})),
            timeout_seconds=2.0,
        )

    assert not client_created


@pytest.mark.parametrize("header_name", ["Host", "Content-Length"])
def test_transport_rejects_caller_controlled_headers(header_name: str) -> None:
    """Keep transport-owned authority and framing headers out of overrides."""
    with pytest.raises(ProviderConfigurationError, match="transport-owned"):
        _provider_transport(headers={header_name: "attacker-controlled"})


def test_predict_batches_in_order_with_deadlines_headers_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch rows without retaining inputs and pass a bounded timeout per call."""

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.posts: list[tuple[str, dict[str, object]]] = []
            self.timeouts: list[object] = []
            self.close_calls = 0
            type(self).instance = self

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def post(self, url: str, **kwargs: object) -> _ProviderResponse:
            self.posts.append((url, kwargs))
            self.timeouts.append(kwargs["timeout"])
            body = kwargs["json"]
            assert isinstance(body, dict)
            tensor = body["inputs"][0]  # type: ignore[index]
            assert isinstance(tensor, dict)
            values = tensor["data"]
            assert isinstance(values, list)
            rows = tensor["shape"][0]
            assert isinstance(rows, int)
            return _ProviderResponse(
                _output_payload(
                    shape=[rows, 1],
                    data=values[::3],
                )
            )

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(),
        _provider_transport(headers={"X-Deployment": "yes"}, max_batch_size=2),
        timeout_seconds=5.0,
    )

    values = np.arange(15, dtype=float).reshape(5, 3)
    result = provider.predict(values)

    assert Client.instance is not None
    assert len(Client.instance.posts) == 3
    shapes: list[object] = []
    for _url, kwargs in Client.instance.posts:
        body = kwargs["json"]
        assert isinstance(body, dict)
        inputs = body["inputs"]
        assert isinstance(inputs, list)
        tensor = inputs[0]
        assert isinstance(tensor, dict)
        shapes.append(tensor["shape"])
    assert shapes == [
        [2, 3],
        [2, 3],
        [1, 3],
    ]
    assert [kwargs["headers"] for _url, kwargs in Client.instance.posts] == [
        {"Content-Type": "application/json"}
    ] * 3
    assert all(
        isinstance(timeout, float) and 0 < timeout <= 5.0
        for timeout in Client.instance.timeouts
    )
    np.testing.assert_allclose(result.reshape(-1), [0.0, 3.0, 6.0, 9.0, 12.0])
    assert provider.inference_batch_count == 3
    assert provider.metadata_latency_seconds >= 0.0
    assert provider.inference_latency_seconds >= 0.0
    assert provider.provider_latency >= provider.inference_latency_seconds

    provider.close()


def test_connect_passes_remaining_metadata_deadline_and_rejects_late_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the remaining connect deadline and reject metadata returned too late."""
    clock = [100.0]
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.metadata_timeout: object = None
            self.close_calls = 0
            type(self).instance = self
            clock[0] += 0.25

        def get(self, _url: str, **kwargs: object) -> _ProviderResponse:
            self.metadata_timeout = kwargs["timeout"]
            clock[0] += 0.9
            return _ProviderResponse(_metadata_payload())

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderDeadlineError):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=1.0
        )

    assert Client.instance is not None
    assert Client.instance.metadata_timeout == pytest.approx(0.75)
    assert Client.instance.close_calls == 1


@pytest.mark.parametrize("status_code", [400, 503])
def test_connect_maps_status_after_acquisition_deadline_to_deadline(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    """Check the absolute deadline before classifying a late HTTP status."""
    clock = [100.0]
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])

    class Context:
        def __enter__(self) -> _ProviderResponse:
            clock[0] += 1.1
            return _ProviderResponse({}, status_code=status_code)

        def __exit__(self, *_args: object) -> None:
            return

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            type(self).instance = self

        def stream(self, _method: str, _url: str, **_kwargs: object) -> Context:
            return Context()

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderDeadlineError):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=1.0
        )

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


@pytest.mark.parametrize(
    ("request_path", "status_code"),
    [
        ("streamed", 400),
        ("streamed", 503),
        ("materialized", 400),
        ("materialized", 503),
    ],
)
def test_status_error_after_checker_deadline_maps_to_deadline(
    monkeypatch: pytest.MonkeyPatch,
    request_path: str,
    status_code: int,
) -> None:
    """Map a status error raised after classification starts past the deadline."""
    clock = [100.0]
    deadline = 101.0
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])

    def delayed_status_checker(response: object) -> None:
        clock[0] += 1.1
        provider_module._check_metadata_status(
            cast("provider_module._HttpResponse", response)
        )

    if request_path == "streamed":
        response = _StreamingResponse(b"{}", status_code=status_code)

        class Client:
            def stream(
                self, _method: str, _url: str, **_kwargs: object
            ) -> _StreamContext:
                return _StreamContext(response)

    else:
        response = _ProviderResponse({}, status_code=status_code)

        class Client:
            def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
                return response

    with pytest.raises(ProviderDeadlineError):
        provider_module._request_response(
            Client(),
            "GET",
            "http://model.example",
            deadline=deadline,
            status_checker=delayed_status_checker,
        )


@pytest.mark.parametrize("request_path", ["streamed", "materialized"])
def test_status_checker_is_not_called_after_response_acquisition_deadline(
    monkeypatch: pytest.MonkeyPatch,
    request_path: str,
) -> None:
    """Reject a response acquired after the deadline before classifying status."""
    clock = [100.0]
    deadline = 101.0
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])
    status_checks = 0

    def status_checker(_response: object) -> None:
        nonlocal status_checks
        status_checks += 1

    if request_path == "streamed":
        response = _StreamingResponse(b"{}", status_code=400)

        class Client:
            def stream(
                self, _method: str, _url: str, **_kwargs: object
            ) -> _StreamContext:
                clock[0] += 1.1
                return _StreamContext(response)

    else:
        response = _ProviderResponse({}, status_code=400)

        class Client:
            def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
                clock[0] += 1.1
                return response

    with pytest.raises(ProviderDeadlineError):
        provider_module._request_response(
            Client(),
            "GET",
            "http://model.example",
            deadline=deadline,
            status_checker=status_checker,
        )

    assert status_checks == 0


def test_connect_closes_client_when_final_deadline_check_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the client when metadata assembly expires at the final check."""

    class Clock:
        def __init__(self) -> None:
            self.monotonic_values = iter(
                (100.0, 100.0, 100.5, 100.9, 100.95, 100.99, 101.1, 101.1)
            )

        def monotonic(self) -> float:
            return next(self.monotonic_values)

    monkeypatch.setattr(provider_module, "time", Clock())

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            type(self).instance = self

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderDeadlineError):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=1.0
        )

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


def test_predict_passes_decreasing_remaining_deadlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recompute one monotonic deadline before every inference batch."""
    clock = [100.0]
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.timeouts: list[float] = []
            type(self).instance = self

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def post(self, _url: str, **kwargs: object) -> _ProviderResponse:
            timeout = kwargs["timeout"]
            assert isinstance(timeout, float)
            self.timeouts.append(timeout)
            clock[0] += 0.1
            return _ProviderResponse(_output_payload(shape=[1, 1], data=[1.0]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(max_batch_size=1), timeout_seconds=2.0
    )

    try:
        provider.predict(np.ones((3, 3)), timeout_seconds=1.0)
    finally:
        provider.close()

    assert Client.instance is not None
    assert len(Client.instance.timeouts) == 3
    assert Client.instance.timeouts[0] > Client.instance.timeouts[1]
    assert Client.instance.timeouts[1] > Client.instance.timeouts[2] > 0


def test_predict_rejects_a_late_inference_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map a response that arrives after the monotonic deadline to a deadline error."""
    clock = [100.0]
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def post(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            clock[0] += 1.1
            return _ProviderResponse(_output_payload(shape=[1, 1], data=[1.0]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )

    try:
        with pytest.raises(ProviderDeadlineError):
            provider.predict(np.ones((1, 3)), timeout_seconds=1.0)
    finally:
        provider.close()


def test_predict_checks_deadline_before_final_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject deadline exhaustion during final result assembly."""
    clock = [100.0]
    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])
    concatenate = np.concatenate

    def late_concatenate(*args: object, **kwargs: object) -> np.ndarray:
        clock[0] += 1.1
        return concatenate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(provider_module.np, "concatenate", late_concatenate)

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def post(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_output_payload(shape=[1, 1], data=[1.0]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )

    try:
        with pytest.raises(ProviderDeadlineError):
            provider.predict(np.ones((1, 3)), timeout_seconds=1.0)
    finally:
        provider.close()


def test_predict_uses_connect_timeout_when_call_timeout_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep direct provider calls bounded by the connect-time timeout."""

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.timeouts: list[object] = []
            type(self).instance = self

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def post(self, _url: str, **kwargs: object) -> _ProviderResponse:
            self.timeouts.append(kwargs["timeout"])
            return _ProviderResponse(_output_payload(shape=[1, 1], data=[1.0]))

        def close(self) -> None:
            return None

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=1.5
    )
    provider.predict(np.ones((1, 3)))

    assert Client.instance is not None
    assert len(Client.instance.timeouts) == 1
    assert 0 < Client.instance.timeouts[0] <= 1.5
    provider.close()


@pytest.mark.parametrize("status_code", [401, 403, 404, 408, 429, 500, 503])
def test_metadata_unavailable_statuses_map_to_provider_unavailable(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    """Map known upstream availability statuses without parsing their bodies."""

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            type(self).instance = self

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse({}, status_code=status_code)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderUnavailableError):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


@pytest.mark.parametrize("status_code", [301, 302, 400, 422])
def test_metadata_contract_statuses_map_to_invalid_response(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    """Map redirects and non-availability HTTP failures to invalid responses."""

    class Client:
        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse({}, status_code=status_code)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(ProviderInvalidResponseError):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )


@pytest.mark.parametrize(
    ("exception", "error_type"),
    [
        (TimeoutError("slow"), ProviderDeadlineError),
        (ConnectionError("refused"), ProviderUnavailableError),
    ],
)
def test_metadata_transport_failures_map_to_typed_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
    exception: Exception,
    error_type: type[Exception],
) -> None:
    """Map metadata connection failures without leaking transport exceptions."""

    class Client:
        instance: ClassVar[Client | None] = None

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0
            type(self).instance = self

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            raise exception

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)

    with pytest.raises(error_type):
        KServeV2HttpPredictionProvider.connect(
            _provider_spec(), _provider_transport(), timeout_seconds=2.0
        )

    assert Client.instance is not None
    assert Client.instance.close_calls == 1


def test_inference_transport_failures_map_to_typed_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map timeout and connection failures from infer calls consistently."""

    class Client:
        failure: ClassVar[Exception] = TimeoutError("slow")

        def __init__(self, **_kwargs: object) -> None:
            self.close_calls = 0

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def post(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            raise type(self).failure

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )
    with pytest.raises(ProviderDeadlineError):
        provider.predict(np.ones((1, 3)))
    provider.close()

    Client.failure = ConnectionError("refused")
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )
    with pytest.raises(ProviderUnavailableError):
        provider.predict(np.ones((1, 3)))
    provider.close()


def test_inference_contract_statuses_are_mapped_by_the_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the shared codec for selected-output and HTTP status validation."""

    class Client:
        status_code: ClassVar[int] = 400

        def __init__(self, **_kwargs: object) -> None:
            return None

        def get(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse(_metadata_payload())

        def post(self, _url: str, **_kwargs: object) -> _ProviderResponse:
            return _ProviderResponse({}, status_code=type(self).status_code)

        def close(self) -> None:
            return None

    monkeypatch.setattr(provider_module, "_load_http_client", lambda: Client)
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )
    with pytest.raises(ProviderInvalidResponseError):
        provider.predict(np.ones((1, 3)))
    provider.close()

    Client.status_code = 503
    provider = KServeV2HttpPredictionProvider.connect(
        _provider_spec(), _provider_transport(), timeout_seconds=2.0
    )
    with pytest.raises(ProviderUnavailableError):
        provider.predict(np.ones((1, 3)))
    provider.close()
