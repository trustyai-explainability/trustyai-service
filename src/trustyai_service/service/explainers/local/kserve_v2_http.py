"""Blocking KServe V2 HTTP/JSON prediction provider."""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import Callable, Iterator, Mapping
from numbers import Integral
from typing import NoReturn, Protocol, cast
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np

from .model_provider import (
    DependencyUnavailableError,
    HttpTransportConfig,
    KServeModelSpec,
    PredictionMetadata,
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
    ProviderUnsupportedModelError,
)

_NUMERIC = frozenset(
    {
        "BOOL",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "FP16",
        "FP32",
        "FP64",
    }
)
_MAX_RESPONSE_ELEMENTS = 10_000_000
_MAX_REQUEST_ELEMENTS = 10_000_000
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_PORT = 65_535
_CONTROL_CHAR_LIMIT = 32
_DEL_CHAR = 127
_MATRIX_RANK = 2
_HTTP_BAD_REQUEST = 400
_HTTP_REDIRECT = 300
_HTTP_SERVER_ERROR = 500


class _HttpResponse(Protocol):
    status_code: int
    content: bytes

    def json(self) -> object: ...


class _HttpClient(Protocol):
    def get(self, url: str, **kwargs: object) -> _HttpResponse: ...

    def post(self, url: str, **kwargs: object) -> _HttpResponse: ...

    def close(self) -> None: ...


class _StreamResponse(Protocol):
    status_code: int

    def iter_bytes(self) -> Iterator[bytes]: ...


class _StreamContext(Protocol):
    def __enter__(self) -> _StreamResponse: ...

    def __exit__(self, *args: object) -> None: ...


class _BufferedResponse:
    """Small response implementation used after bounded streaming reads."""

    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content

    def json(self) -> object:
        """Decode the bounded response body as JSON."""
        return json.loads(self.content)


def _bounded_stream_content(response: _StreamResponse) -> bytes:
    """Read a streamed response without exceeding the provider byte limit."""
    content = bytearray()
    size = 0
    for chunk in response.iter_bytes():
        if not isinstance(chunk, bytes):
            msg = "Model response contains invalid bytes"
            raise ProviderInvalidResponseError(msg)
        size += len(chunk)
        if size > _MAX_RESPONSE_BYTES:
            msg = "Model response is too large"
            raise ProviderInvalidResponseError(msg)
        content.extend(chunk)
    return bytes(content)


def _check_materialized_response(response: _HttpResponse) -> None:
    """Enforce the response limit for test doubles and non-streaming clients."""
    content = getattr(response, "content", None)
    if content is not None and len(content) > _MAX_RESPONSE_BYTES:
        msg = "Model response is too large"
        raise ProviderInvalidResponseError(msg)


def _request_bounded(
    client: _HttpClient,
    method: str,
    url: str,
    timeout_seconds: float,
    **kwargs: object,
) -> _HttpResponse:
    """Issue one request and bound the response before parsing JSON."""
    stream = cast(
        "Callable[..., _StreamContext] | None", getattr(client, "stream", None)
    )
    if callable(stream):
        with stream(method, url, timeout=timeout_seconds, **kwargs) as response:
            status_code = int(response.status_code)
            if status_code >= _HTTP_REDIRECT:
                return _BufferedResponse(status_code, b"")
            return _BufferedResponse(status_code, _bounded_stream_content(response))
    if method == "GET":
        response = client.get(url, timeout=timeout_seconds, **kwargs)
    else:
        response = client.post(url, timeout=timeout_seconds, **kwargs)
    _check_materialized_response(response)
    return response


def _invalid_response(message: str) -> NoReturn:
    raise ValueError(message)


def _load_http_client() -> type:
    """Load the optional HTTP client only when MODEL execution is requested."""
    module = importlib.import_module("httpx2")
    return cast("type", module.Client)


def _validate_input_datatype(values: np.ndarray, datatype: str) -> None:
    if datatype == "BOOL":
        if not np.all(np.isin(values, [0, 1])):
            msg = "Boolean model inputs must be zero or one"
            raise ProviderInvalidRequestError(msg)
        return
    if datatype.startswith(("INT", "UINT")):
        if not np.equal(values, np.floor(values)).all():
            msg = "Integer model inputs must be integral"
            raise ProviderInvalidRequestError(msg)
        bits = int(datatype.removeprefix("UINT").removeprefix("INT"))
        lower = 0 if datatype.startswith("UINT") else -(2 ** (bits - 1))
        upper = 2**bits - 1 if datatype.startswith("UINT") else 2 ** (bits - 1) - 1
        if np.any(values < lower) or np.any(values > upper):
            msg = "Integer model inputs are out of range"
            raise ProviderInvalidRequestError(msg)


def _validate_output_datatype(values: np.ndarray, datatype: str) -> None:
    if datatype == "BOOL":
        if not np.all(np.isin(values, [0, 1])):
            msg = "Boolean model outputs are invalid"
            _invalid_response(msg)
        return
    if datatype.startswith(("INT", "UINT")):
        if not np.equal(values, np.floor(values)).all():
            msg = "Integer model outputs are not integral"
            _invalid_response(msg)
        bits = int(datatype.removeprefix("UINT").removeprefix("INT"))
        lower = 0 if datatype.startswith("UINT") else -(2 ** (bits - 1))
        upper = 2**bits - 1 if datatype.startswith("UINT") else 2 ** (bits - 1) - 1
        if np.any(values < lower) or np.any(values > upper):
            msg = "Integer model outputs are out of range"
            _invalid_response(msg)


def _timeout_error(error: Exception) -> bool:
    return isinstance(error, TimeoutError) or error.__class__.__name__.endswith(
        "TimeoutException"
    )


def normalize_base_url(value: str) -> str:
    """Validate and normalize a KServe HTTP server root."""
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        msg = "base_url must be an HTTP(S) URL"
        raise ProviderInvalidRequestError(msg) from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        msg = "base_url must be an HTTP(S) URL"
        raise ProviderInvalidRequestError(msg)
    try:
        _ = parsed.port
    except ValueError as exc:
        msg = "base_url has an invalid port"
        raise ProviderInvalidRequestError(msg) from exc
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        msg = "base_url must not contain credentials, query, or fragment"
        raise ProviderInvalidRequestError(msg)
    if any(
        ord(char) < _CONTROL_CHAR_LIMIT or ord(char) == _DEL_CHAR or char in "\\%"
        for char in parsed.netloc
    ):
        msg = "base_url contains an unsafe authority"
        raise ProviderInvalidRequestError(msg)
    if (
        any(
            ord(char) < _CONTROL_CHAR_LIMIT or ord(char) == _DEL_CHAR
            for char in parsed.path
        )
        or "%" in parsed.path
    ):
        msg = "base_url contains an unsafe path"
        raise ProviderInvalidRequestError(msg)
    if any(part in {".", ".."} for part in parsed.path.split("/")):
        msg = "base_url contains path traversal"
        raise ProviderInvalidRequestError(msg)
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def _segment(value: str, label: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or any(char in value for char in "/\\%")
        or any(
            ord(char) < _CONTROL_CHAR_LIMIT or ord(char) == _DEL_CHAR for char in value
        )
    ):
        msg = f"Invalid {label}"
        raise ProviderInvalidRequestError(msg)
    return quote(value, safe="-_.~")


def _allowlist_entry(value: str) -> str:
    """Normalize one host[:port] allowlist entry for exact authority checks."""
    raw = str(value).strip().lower().rstrip(".")
    if not raw:
        msg = "empty allowlist entry"
        _invalid_response(msg)
    if any(
        ord(char) < _CONTROL_CHAR_LIMIT or ord(char) == _DEL_CHAR or char in "\\%"
        for char in raw
    ):
        msg = "allowlist entry contains unsafe characters"
        _invalid_response(msg)
    parsed = urlsplit(f"//{raw}")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        msg = "allowlist entry must be a host and optional port"
        _invalid_response(msg)
    host = parsed.hostname
    if not host or parsed.username or parsed.password:
        msg = "invalid allowlist host"
        _invalid_response(msg)
    port = parsed.port
    if port is not None and not 1 <= port <= _MAX_PORT:
        msg = "invalid allowlist port"
        _invalid_response(msg)
    normalized_host = f"[{host}]" if ":" in host else host
    return f"{normalized_host}:{port}" if port is not None else normalized_host


def _validate_model_endpoint(
    spec: KServeModelSpec, transport: HttpTransportConfig
) -> str:
    """Validate the model endpoint against deployment transport policy."""
    base = normalize_base_url(spec.base_url)
    parsed_base = urlsplit(base)
    host = (parsed_base.hostname or "").lower().rstrip(".")
    try:
        effective_port = parsed_base.port or (
            443 if parsed_base.scheme == "https" else 80
        )
    except ValueError as exc:
        msg = "base_url has an invalid port"
        raise ProviderInvalidRequestError(msg) from exc
    if not transport.allowed_hosts:
        msg = "Outbound model host allowlist is required"
        raise ProviderConfigurationError(msg)
    if transport.follow_redirects or transport.trust_env:
        msg = "Redirects and ambient proxy settings must remain disabled"
        raise ProviderConfigurationError(msg)
    try:
        entries = {_allowlist_entry(entry) for entry in transport.allowed_hosts}
    except (TypeError, ValueError) as exc:
        msg = "Invalid outbound host allowlist"
        raise ProviderConfigurationError(msg) from exc
    normalized_host = f"[{host}]" if ":" in host else host
    host_ok = normalized_host in entries or host in entries
    host_ok = host_ok or f"{normalized_host}:{effective_port}" in entries
    if not host_ok:
        msg = "Model host is not in the outbound allowlist"
        raise ProviderInvalidRequestError(msg)
    return base


def _create_http_client(
    client_type: type, transport: HttpTransportConfig, timeout_seconds: float
) -> object:
    """Construct the optional client and classify configuration failures."""
    try:
        return client_type(
            headers=dict(transport.headers),
            verify=transport.verify,
            cert=transport.cert,
            follow_redirects=transport.follow_redirects,
            trust_env=transport.trust_env,
            timeout=timeout_seconds,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ProviderConfigurationError from exc
    except Exception as exc:
        raise ProviderUnavailableError from exc


def _shape(value: object) -> tuple[int, ...]:
    """Parse a KServe shape without silently coercing malformed dimensions."""
    if not isinstance(value, list):
        msg = "tensor shape must be a JSON list"
        raise TypeError(msg)
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, Integral)
        for dimension in value
    ):
        msg = "tensor dimensions must be integers"
        raise TypeError(msg)
    return tuple(int(dimension) for dimension in value)


def _mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _invalid_response(message)
    return value


def _tensor_list(
    payload: Mapping[str, object], name: str
) -> list[Mapping[str, object]]:
    value = payload.get(name) or []
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) for item in value
    ):
        _invalid_response(f"model metadata {name} must be a list of tensors")
    return cast("list[Mapping[str, object]]", value)


def _validate_model_version(
    payload: Mapping[str, object], spec: KServeModelSpec
) -> None:
    if "versions" not in payload:
        return
    versions = payload["versions"]
    if not isinstance(versions, list) or any(
        not isinstance(version, str) or not version for version in versions
    ):
        msg = "Model metadata versions must be a list"
        raise ProviderUnsupportedModelError(msg)
    if spec.model_version is not None and spec.model_version not in versions:
        msg = "Requested model version was not found"
        raise ProviderInvalidRequestError(msg)


def _select_tensor(
    tensors: list[Mapping[str, object]],
    selector: str | None,
    label: str,
    *,
    require_single: bool = False,
) -> Mapping[str, object]:
    selected = [
        tensor
        for tensor in tensors
        if selector is None or tensor.get("name") == selector
    ]
    if not selected:
        if selector is None:
            msg = f"Model {label} metadata is missing"
            raise ProviderUnsupportedModelError(msg)
        msg = f"Requested model {label} tensor was not found"
        raise ProviderInvalidRequestError(msg)
    if len(selected) != 1:
        msg = f"Model {label} metadata is ambiguous"
        if require_single:
            msg = (
                f"Model {label} metadata is ambiguous or has multiple required tensors"
            )
        raise ProviderUnsupportedModelError(msg)
    return selected[0]


def _validate_tensor_shapes(
    input_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> None:
    if (
        not input_shape
        or not output_shape
        or len(input_shape) > _MATRIX_RANK
        or len(output_shape) > _MATRIX_RANK
    ):
        msg = "Only flat rank-one or rank-two tensors are supported"
        raise ProviderUnsupportedModelError(msg)
    if any(
        dimension == 0 or dimension < -1 for dimension in input_shape + output_shape
    ):
        msg = "Tensor dimensions must be positive or -1"
        raise ProviderUnsupportedModelError(msg)
    if any(dimension == -1 for dimension in input_shape[1:]) or any(
        dimension == -1 for dimension in output_shape[1:]
    ):
        msg = "Only a leading dynamic batch dimension is supported"
        raise ProviderUnsupportedModelError(msg)
    if len(input_shape) == _MATRIX_RANK and input_shape[0] not in {-1, 1}:
        msg = "Fixed model batches other than one are unsupported"
        raise ProviderUnsupportedModelError(msg)
    if len(output_shape) == _MATRIX_RANK and output_shape[0] not in {-1, 1}:
        msg = "Fixed model batches other than one are unsupported"
        raise ProviderUnsupportedModelError(msg)


def _metadata_from_payload(
    payload_value: object, spec: KServeModelSpec
) -> PredictionMetadata:
    payload = _mapping(payload_value, "model metadata must be a JSON object")
    if payload.get("name") != spec.model_name:
        msg = "Model metadata name does not match the request"
        raise ProviderUnsupportedModelError(msg)
    _validate_model_version(payload, spec)
    inputs = _tensor_list(payload, "inputs")
    outputs = _tensor_list(payload, "outputs")
    if len(inputs) != 1:
        msg = "Model input metadata is ambiguous or has multiple required inputs"
        raise ProviderUnsupportedModelError(msg)
    input_tensor = _select_tensor(inputs, spec.input_name, "input", require_single=True)
    output_tensor = _select_tensor(outputs, spec.output_name, "output")
    input_datatype = input_tensor.get("datatype")
    output_datatype = output_tensor.get("datatype")
    if input_datatype not in _NUMERIC or output_datatype not in _NUMERIC:
        msg = "Only numeric tensors are supported"
        raise ProviderUnsupportedModelError(msg)
    input_name = input_tensor.get("name")
    output_name = output_tensor.get("name")
    if not isinstance(input_name, str) or not isinstance(output_name, str):
        _invalid_response("model tensor names must be strings")
    input_shape = _shape(input_tensor.get("shape", []))
    output_shape = _shape(output_tensor.get("shape", []))
    _validate_tensor_shapes(input_shape, output_shape)
    return PredictionMetadata(
        input_name,
        output_name,
        cast("str", input_datatype),
        cast("str", output_datatype),
        input_shape,
        output_shape,
    )


def _metadata_response(
    client: _HttpClient, url: str, timeout_seconds: float
) -> _HttpResponse:
    try:
        response = _request_bounded(client, "GET", url, max(0.001, timeout_seconds))
    except ProviderError:
        raise
    except Exception as exc:
        if _timeout_error(exc):
            raise ProviderDeadlineError from exc
        raise ProviderUnavailableError from exc
    if (
        response.status_code in {401, 403, 404, 408, 429}
        or response.status_code >= _HTTP_SERVER_ERROR
    ):
        raise ProviderUnavailableError
    if response.status_code >= _HTTP_BAD_REQUEST:
        msg = "Model metadata request was rejected"
        raise ProviderInvalidResponseError(msg)
    if response.status_code >= _HTTP_REDIRECT:
        raise ProviderInvalidResponseError
    return response


def _prediction_layout(
    metadata: PredictionMetadata, values: np.ndarray, max_batch_size: int
) -> tuple[tuple[int, ...], int]:
    row_shape = (
        metadata.input_shape[1:]
        if len(metadata.input_shape) == _MATRIX_RANK
        else (() if metadata.input_shape == (-1,) else metadata.input_shape)
    )
    if row_shape and row_shape[0] > 0 and values.shape[1] != row_shape[-1]:
        msg = "Input feature width does not match model metadata"
        raise ProviderInvalidRequestError(msg)
    if metadata.input_shape == (-1,) and values.shape[1] != 1:
        msg = "Dynamic scalar input metadata requires one feature"
        raise ProviderInvalidRequestError(msg)
    fixed_batch = (
        len(metadata.input_shape) == _MATRIX_RANK and metadata.input_shape[0] == 1
    ) or (len(metadata.output_shape) == _MATRIX_RANK and metadata.output_shape[0] == 1)
    return row_shape, 1 if fixed_batch else max_batch_size


def _inference_body(
    metadata: PredictionMetadata,
    chunk: np.ndarray,
    row_shape: tuple[int, ...],
    *,
    output_name: str | None = None,
) -> dict[str, object]:
    shape = [len(chunk), *row_shape] if row_shape else [len(chunk)]
    body: dict[str, object] = {
        "inputs": [
            {
                "name": metadata.input_name,
                "shape": shape,
                "datatype": metadata.input_datatype,
                "data": chunk.reshape(-1).tolist(),
            }
        ]
    }
    if output_name is not None:
        body["outputs"] = [{"name": output_name}]
    return body


def _validate_inference_status(response: _HttpResponse) -> None:
    if response.status_code >= _HTTP_SERVER_ERROR:
        raise ProviderUnavailableError
    if response.status_code in {401, 403, 404, 408, 429}:
        raise ProviderUnavailableError
    if response.status_code >= _HTTP_REDIRECT:
        msg = "Model inference request was rejected"
        raise ProviderInvalidResponseError(msg)


def _output_shape_is_valid(
    data: np.ndarray,
    out_shape: tuple[int, ...],
    metadata_shape: tuple[int, ...],
    chunk_size: int,
) -> bool:
    if not out_shape or len(out_shape) not in {1, _MATRIX_RANK}:
        return False
    expected = int(np.prod(out_shape))
    if expected > _MAX_RESPONSE_ELEMENTS or data.size != expected:
        return False
    if out_shape[0] != chunk_size:
        return False
    if len(metadata_shape) == 1 and metadata_shape[0] == -1:
        valid_shape = out_shape == (chunk_size,)
    elif len(metadata_shape) == 1:
        valid_shape = out_shape == (chunk_size, metadata_shape[0]) or (
            metadata_shape[0] == 1 and out_shape == (chunk_size,)
        )
    else:
        valid_shape = out_shape == (chunk_size, *metadata_shape[1:]) or (
            metadata_shape[1:] == (1,) and out_shape == (chunk_size,)
        )
    scalar_output = (len(metadata_shape) == 1 and metadata_shape[0] in {-1, 1}) or (
        len(metadata_shape) == _MATRIX_RANK and metadata_shape[1] == 1
    )
    return valid_shape and (
        data.shape == out_shape or (scalar_output and data.ndim == 1)
    )


def _decode_output_tensor(
    selected: Mapping[str, object],
    metadata: PredictionMetadata,
    chunk_size: int,
) -> np.ndarray:
    if selected.get("datatype") != metadata.output_datatype:
        _invalid_response("response datatype mismatch")
    out_shape = _shape(selected["shape"])
    raw_data = selected.get("data")
    if isinstance(raw_data, (list, tuple)) and len(raw_data) > _MAX_RESPONSE_ELEMENTS:
        _invalid_response("model output is too large")
    data = np.asarray(raw_data)
    if data.size > _MAX_RESPONSE_ELEMENTS:
        _invalid_response("model output is too large")
    if data.dtype.kind not in "bfiu":
        _invalid_response("model output must be numeric")
    data = data.astype(float)
    if not np.isfinite(data).all():
        _invalid_response("model output must be finite")
    if not _output_shape_is_valid(data, out_shape, metadata.output_shape, chunk_size):
        _invalid_response("output shape mismatch")
    _validate_output_datatype(data, metadata.output_datatype)
    scalar_output = (
        len(metadata.output_shape) == 1 and metadata.output_shape[0] in {-1, 1}
    ) or (len(metadata.output_shape) == _MATRIX_RANK and metadata.output_shape[1] == 1)
    return data.reshape(chunk_size, 1) if scalar_output else data.reshape(out_shape)


def _decode_inference_response(
    response: _HttpResponse,
    metadata: PredictionMetadata,
    spec: KServeModelSpec,
    chunk_size: int,
) -> np.ndarray:
    _validate_inference_status(response)
    try:
        content = getattr(response, "content", None)
        if content is not None and len(content) > _MAX_RESPONSE_BYTES:
            _invalid_response("model response is too large")
        payload = _mapping(
            response.json(), "model inference response must be a JSON object"
        )
        if payload.get("model_name") != spec.model_name:
            _invalid_response("response model name mismatch")
        if (
            spec.model_version is not None
            and "model_version" in payload
            and payload["model_version"] != spec.model_version
        ):
            _invalid_response("response model version mismatch")
        outputs = _tensor_list(payload, "outputs")
        selected_outputs = [
            item for item in outputs if item.get("name") == metadata.output_name
        ]
        if len(selected_outputs) != 1:
            _invalid_response("selected output is missing or duplicated")
        return _decode_output_tensor(selected_outputs[0], metadata, chunk_size)
    except ProviderError:
        raise
    except (
        AttributeError,
        StopIteration,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise ProviderInvalidResponseError from exc


class KServeV2HttpPredictionProvider:
    """One metadata-negotiated KServe V2 client for one explanation."""

    def __init__(
        self,
        client: object,
        metadata: PredictionMetadata,
        spec: KServeModelSpec,
        max_batch_size: int,
        default_timeout: float = 30.0,
    ) -> None:
        """Initialize a negotiated provider and its request accounting."""
        self._client = client
        self._metadata = metadata
        self._spec = spec
        self._max_batch_size = max_batch_size
        self._default_timeout = default_timeout
        self._closed = False
        self._inference_batch_count = 0
        self._provider_latency = 0.0

    @property
    def metadata(self) -> PredictionMetadata:
        """Return metadata negotiated from the model server."""
        return self._metadata

    @property
    def inference_batch_count(self) -> int:
        """Return the number of inference batches sent to the model."""
        return self._inference_batch_count

    @property
    def provider_latency(self) -> float:
        """Return accumulated metadata and inference latency in seconds."""
        return self._provider_latency

    def record_metadata_latency(self, elapsed: float) -> None:
        """Add metadata negotiation time to the provider latency total."""
        self._provider_latency += elapsed

    @classmethod
    def connect(
        cls,
        spec: KServeModelSpec,
        timeout_seconds: float,
        transport: HttpTransportConfig,
    ) -> KServeV2HttpPredictionProvider:
        """Connect, negotiate metadata, and construct a bounded provider."""
        base = _validate_model_endpoint(spec, transport)
        try:
            client_type = _load_http_client()
        except ImportError as exc:
            raise DependencyUnavailableError from exc
        client = _create_http_client(client_type, transport, timeout_seconds)
        provider = None
        metadata_started = time.monotonic()
        try:
            provider = cls._from_metadata(
                client, spec, transport.max_batch_size, base, timeout_seconds
            )
            provider.record_metadata_latency(time.monotonic() - metadata_started)
        except Exception:
            client.close()
            raise
        else:
            return provider

    @classmethod
    def _from_metadata(
        cls,
        client: object,
        spec: KServeModelSpec,
        max_batch_size: int,
        base: str,
        timeout_seconds: float,
    ) -> KServeV2HttpPredictionProvider:
        http_client = cast("_HttpClient", client)
        model = _segment(spec.model_name, "model name")
        version = (
            f"/versions/{_segment(spec.model_version, 'model version')}"
            if spec.model_version
            else ""
        )
        url = f"{base}/v2/models/{model}{version}"
        response = _metadata_response(http_client, url, timeout_seconds)
        try:
            metadata = _metadata_from_payload(response.json(), spec)
            return cls(
                client,
                metadata,
                spec,
                max_batch_size,
                timeout_seconds,
            )
        except ProviderError:
            raise
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise ProviderInvalidResponseError from exc

    def _infer_url(self) -> str:
        model = _segment(self._spec.model_name, "model name")
        version = (
            f"/versions/{_segment(self._spec.model_version, 'model version')}"
            if self._spec.model_version
            else ""
        )
        return f"{normalize_base_url(self._spec.base_url)}/v2/models/{model}{version}/infer"

    def predict(
        self, inputs: np.ndarray, *, timeout_seconds: float | None = None
    ) -> np.ndarray:
        """Send validated input batches and return the selected output tensor."""
        if self._closed:
            msg = "Model provider is closed"
            raise ProviderUnavailableError(msg)
        values = np.asarray(inputs)
        if (
            values.ndim != _MATRIX_RANK
            or values.dtype.kind not in "bfiu"
            or not np.isfinite(values).all()
        ):
            msg = "Model inputs must be a finite numeric matrix"
            raise ProviderInvalidRequestError(msg)
        if values.size > _MAX_REQUEST_ELEMENTS:
            msg = "Model input request is too large"
            raise ProviderInvalidRequestError(msg)
        _validate_input_datatype(values, self._metadata.input_datatype)
        row_shape, batch_limit = _prediction_layout(
            self._metadata, values, self._max_batch_size
        )
        effective_timeout = (
            self._default_timeout if timeout_seconds is None else timeout_seconds
        )
        deadline = time.monotonic() + effective_timeout
        chunks: list[np.ndarray] = []
        for start in range(0, len(values), batch_limit):
            chunk = values[start : start + batch_limit]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderDeadlineError
            body = _inference_body(
                self._metadata,
                chunk,
                row_shape,
                output_name=self._spec.output_name,
            )
            response = self._post_inference(body, remaining)
            chunks.append(
                _decode_inference_response(
                    response, self._metadata, self._spec, len(chunk)
                )
            )
        if not chunks:
            return np.empty((0, 1), dtype=float)
        return np.concatenate(chunks, axis=0)

    def _post_inference(
        self, body: dict[str, object], timeout_seconds: float
    ) -> _HttpResponse:
        started = time.monotonic()
        try:
            response = _request_bounded(
                cast("_HttpClient", self._client),
                "POST",
                self._infer_url(),
                timeout_seconds,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            self._inference_batch_count += 1
        except ProviderError:
            raise
        except Exception as exc:
            if _timeout_error(exc):
                raise ProviderDeadlineError from exc
            raise ProviderUnavailableError from exc
        else:
            return response
        finally:
            self._provider_latency += time.monotonic() - started

    def close(self) -> None:
        """Close the underlying HTTP client exactly once."""
        if not self._closed:
            self._closed = True
            cast("_HttpClient", self._client).close()
