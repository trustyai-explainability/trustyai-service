"""Blocking outbound KServe V2 HTTP prediction provider."""

from __future__ import annotations

import importlib
import json
import math
import os
import socket
import threading
import time
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, NoReturn, Protocol, Self, cast, runtime_checkable
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np

from . import kserve_v2_codec as codec
from .model_provider import (
    DependencyUnavailableError,
    HttpTransportConfig,
    PredictionMetadata,
    PredictionProvider,
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderInvalidResponseError,
    ProviderUnavailableError,
)
from .transport_config import resolve_outbound_addresses

_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_HTTP_SUCCESS_MIN = 200
_HTTP_REDIRECT = 300
_HTTP_SERVER_ERROR = 500
_CONTROL_CHAR_LIMIT = 32
_DEL_CHAR = 127
_MATRIX_RANK = 2
_MAX_PORT = 65535
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024

_REQUEST_DEADLINE: ContextVar[float | None] = ContextVar(
    "trustyai_provider_request_deadline", default=None
)
_SUPPRESS_NETWORK_CLEANUP: ContextVar[bool] = ContextVar(
    "trustyai_provider_suppress_network_cleanup", default=False
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping


def _configuration_error(message: str) -> NoReturn:
    """Raise a typed configuration error at the provider boundary."""
    raise ProviderConfigurationError(message)


def _invalid_request(message: str) -> NoReturn:
    """Raise a typed request error at the provider boundary."""
    raise ProviderInvalidRequestError(message)


def _invalid_response(message: str) -> NoReturn:
    """Raise a typed response error at the provider boundary."""
    raise ProviderInvalidResponseError(message)


class _HttpResponse(Protocol):
    """Minimum HTTP response boundary required by the provider and codec."""

    status_code: int
    content: bytes

    def json(self) -> object:
        """Decode the response body."""
        ...


class _StreamResponse(Protocol):
    """Minimum streamed response boundary required by the provider."""

    status_code: int
    headers: object

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        """Iterate decoded response body chunks."""
        ...


@runtime_checkable
class _NonBlockingResponseCleanup(Protocol):
    """Explicit capability for bounded cleanup after a response error."""

    def close_nonblocking(self) -> None:
        """Release response resources without waiting on network coordination."""
        ...


class _NetworkStream(Protocol):
    """Synchronous httpcore network stream seam used by the deadline wrapper."""

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        """Read bytes from the underlying connection."""
        ...

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        """Write bytes to the underlying connection."""
        ...

    def close(self) -> None:
        """Close the underlying connection."""
        ...

    def start_tls(
        self,
        ssl_context: object,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> _NetworkStream:
        """Upgrade the connection to TLS."""
        ...

    def get_extra_info(self, info: str) -> object:
        """Return optional transport metadata."""
        ...


class _NetworkBackend(Protocol):
    """Synchronous httpcore network backend seam used by the provider."""

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> _NetworkStream:
        """Connect a TCP stream."""
        ...

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: object = None,
    ) -> _NetworkStream:
        """Connect a Unix-domain stream."""
        ...

    def sleep(self, seconds: float) -> None:
        """Sleep between connection retries."""
        ...


class _HttpClient(Protocol):
    """Minimum synchronous client boundary used by one provider instance."""

    def get(self, url: str, **kwargs: object) -> _HttpResponse:
        """Issue one metadata request."""
        ...

    def post(self, url: str, **kwargs: object) -> _HttpResponse:
        """Issue one inference request."""
        ...

    def close(self) -> None:
        """Close the synchronous client."""
        ...


@dataclass(frozen=True)
class _BufferedResponse:
    """Response with a body proven to be within the provider byte limit."""

    status_code: int
    content: bytes

    def json(self) -> object:
        """Decode the bounded response body as JSON."""
        return json.loads(self.content)


@dataclass(frozen=True)
class KServeModelSpec:
    """Identity and tensor selectors for one outbound KServe model."""

    base_url: str
    model_name: str
    model_version: str | None
    input_name: str | None
    output_name: str | None


def _invalid_base_url(message: str = "base_url must be an HTTP(S) URL") -> None:
    """Raise the canonical request error for an unsafe model server URL."""
    raise ProviderInvalidRequestError(message)


def _has_unsafe_url_character(value: str) -> bool:
    """Return whether a URL component contains unsafe raw characters."""
    return any(
        ord(character) < _CONTROL_CHAR_LIMIT
        or ord(character) == _DEL_CHAR
        or character.isspace()
        or character in {"\\", "%"}
        for character in value
    )


def _validate_authority_port(netloc: str) -> None:
    """Reject malformed, empty, and out-of-range explicit authority ports."""
    if netloc.startswith("["):
        closing_bracket = netloc.find("]")
        if closing_bracket < 0:
            _invalid_base_url("base_url has an invalid authority")
        port_suffix = netloc[closing_bracket + 1 :]
        if not port_suffix:
            return
        if not port_suffix.startswith(":"):
            _invalid_base_url("base_url has an invalid authority")
        raw_port = port_suffix[1:]
    else:
        if netloc.count(":") > 1:
            _invalid_base_url("base_url has an invalid authority")
        if ":" not in netloc:
            return
        raw_port = netloc.rsplit(":", 1)[1]

    if not raw_port or not raw_port.isascii() or not raw_port.isdecimal():
        _invalid_base_url("base_url has an invalid authority port")
    port = int(raw_port)
    if not 1 <= port <= _MAX_PORT:
        _invalid_base_url("base_url has an invalid authority port")


def _normalize_base_url(value: object) -> str:
    """Validate and normalize an HTTP(S) KServe server root."""
    if not isinstance(value, str) or _has_unsafe_url_character(value):
        _invalid_base_url("base_url contains unsafe characters")
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.netloc:
            _invalid_base_url()
        if (
            parsed.username is not None
            or parsed.password is not None
            or "@" in parsed.netloc
            or parsed.query
            or parsed.fragment
            or "?" in value
            or "#" in value
        ):
            _invalid_base_url(
                "base_url must not contain credentials, query, or fragment"
            )
        if _has_unsafe_url_character(parsed.netloc):
            _invalid_base_url("base_url contains an unsafe authority")
        if not parsed.hostname:
            _invalid_base_url("base_url has an invalid authority")
        _validate_authority_port(parsed.netloc)
    except ProviderInvalidRequestError:
        raise
    except (TypeError, ValueError) as exc:
        message = "base_url has an invalid authority"
        raise ProviderInvalidRequestError(message) from exc

    path = parsed.path
    if _has_unsafe_url_character(path):
        _invalid_base_url("base_url contains an unsafe path")
    if any(part in {".", ".."} for part in path.split("/")):
        _invalid_base_url("base_url contains path traversal")
    return urlunsplit((scheme, parsed.netloc, path.rstrip("/"), "", ""))


def _encode_path_segment(value: object, label: str) -> str:
    """Validate and percent-encode one model identity path segment."""
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(
            ord(character) < _CONTROL_CHAR_LIMIT
            or ord(character) == _DEL_CHAR
            or character in {"/", "\\", "%"}
            for character in value
        )
    ):
        _invalid_request(f"Invalid {label}")
    return quote(value, safe="-_.~")


def _model_url(base_url: str, spec: KServeModelSpec, *, infer: bool) -> str:
    """Build one KServe V2 metadata or inference URL."""
    model_name = _encode_path_segment(spec.model_name, "model name")
    version = (
        ""
        if spec.model_version is None
        else f"/versions/{_encode_path_segment(spec.model_version, 'model version')}"
    )
    suffix = "/infer" if infer else ""
    return f"{base_url}/v2/models/{model_name}{version}{suffix}"


def _load_http_client() -> type:
    """Load the optional synchronous HTTP client only at provider connect time."""
    module = importlib.import_module("httpx2")
    client_type = getattr(module, "Client", None)
    if not isinstance(client_type, type):
        raise DependencyUnavailableError
    return client_type


def _is_timeout_error(error: BaseException) -> bool:
    """Recognize stdlib and optional-client timeout exception families."""
    return isinstance(error, TimeoutError) or any(
        "timeout" in cls.__name__.lower() for cls in type(error).__mro__
    )


def _is_httpx2_decoding_error(error: BaseException) -> bool:
    """Recognize only httpx2 failures decoding an upstream response body."""
    try:
        httpx_module = importlib.import_module("httpx2")
    except ImportError:
        return False
    decoding_error = getattr(httpx_module, "DecodingError", None)
    return isinstance(decoding_error, type) and isinstance(error, decoding_error)


def _validate_connect_timeout(value: object) -> float:
    """Validate the provider's default request timeout."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        _configuration_error("Model timeout must be a positive finite number")
    return float(value)


def _remaining_timeout(deadline: float) -> float:
    """Return positive time remaining until one monotonic request deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ProviderDeadlineError
    return remaining


def _response_status(response: _HttpResponse) -> int:
    """Read and validate an upstream response status code."""
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, bool) or not isinstance(status_code, Integral):
        _invalid_response("model response has an invalid status")
    return int(status_code)


def _check_response_size(response: _HttpResponse) -> None:
    """Reject an oversized materialized response before JSON parsing."""
    try:
        content = getattr(response, "content", None)
        if content is not None and len(content) > _MAX_RESPONSE_BYTES:
            _invalid_response("model response is too large")
    except ProviderInvalidResponseError:
        raise
    except (AttributeError, TypeError) as exc:
        message = "model response has invalid content"
        raise ProviderInvalidResponseError(message) from exc


def _declared_content_length(response: object) -> int | None:
    """Return a valid Content-Length header when the upstream supplied one."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("Content-Length")
        if value is None:
            value = headers.get("content-length")
    except (AttributeError, TypeError) as exc:
        message = "model response has invalid headers"
        raise ProviderInvalidResponseError(message) from exc
    if value is None:
        return None
    if isinstance(value, str):
        raw_value = value.strip()
        if not raw_value.isascii() or not raw_value.isdecimal():
            _invalid_response("model response has an invalid content length")
        try:
            return int(raw_value)
        except ValueError as exc:
            message = "model response has an invalid content length"
            raise ProviderInvalidResponseError(message) from exc
    if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0:
        return int(value)
    _invalid_response("model response has an invalid content length")


def _has_non_identity_content_encoding(response: object) -> bool:
    """Return whether the response body must be decoded before limiting it."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return False
    try:
        value = headers.get("Content-Encoding")
        if value is None:
            value = headers.get("content-encoding")
    except (AttributeError, TypeError) as exc:
        message = "model response has invalid headers"
        raise ProviderInvalidResponseError(message) from exc
    if value is None:
        return False
    if not isinstance(value, str):
        _invalid_response("model response has invalid content encoding")
    return any(
        encoding.strip().lower() not in {"", "identity"}
        for encoding in value.split(",")
    )


def _request_io_timeout(timeout: float | None) -> float | None:
    """Bound one transport operation by the request's absolute deadline."""
    deadline = _REQUEST_DEADLINE.get()
    if deadline is None:
        return timeout
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    if timeout is None:
        return remaining
    return min(timeout, remaining)


def _request_deadline_expired() -> bool:
    """Return whether the active request deadline has elapsed."""
    deadline = _REQUEST_DEADLINE.get()
    return deadline is not None and time.monotonic() >= deadline


def _force_close_stream_socket(stream: object) -> None:
    """Close one active stream's descriptor without waiting on socket cleanup."""
    get_extra_info = getattr(stream, "get_extra_info", None)
    if not callable(get_extra_info):
        return
    try:
        response_socket = get_extra_info("socket")
        if not isinstance(response_socket, socket.socket):
            return
        descriptor = response_socket.detach()
        os.close(descriptor)
    except (OSError, TypeError, ValueError):
        return


class _DirectConnection(Protocol):
    """Small httpcore connection boundary owned by one response."""

    def handle_request(self, request: object) -> object:
        """Send one request on this connection."""
        ...

    def close(self) -> None:
        """Close this connection."""
        ...


class _DirectCoreResponse(Protocol):
    """Minimum httpcore response boundary needed for httpx response wrapping."""

    status: int
    headers: object
    stream: object
    extensions: object


class _HttpxUrl(Protocol):
    """Minimum httpx2 URL boundary passed to a custom transport."""

    raw_scheme: bytes
    raw_host: bytes
    port: int
    raw_path: bytes


class _HttpxHeaders(Protocol):
    """Minimum httpx2 headers boundary passed to a custom transport."""

    raw: object


class _HttpxRequest(Protocol):
    """Minimum httpx2 request boundary passed to a custom transport."""

    method: bytes
    url: _HttpxUrl
    headers: _HttpxHeaders
    stream: object
    extensions: object


class _HttpxModule(Protocol):
    """Minimum optional httpx2 module boundary used by the adapter."""

    Response: type
    SyncByteStream: type


class _HttpcoreModule(Protocol):
    """Minimum optional httpcore2 module boundary used by the adapter."""

    URL: type
    HTTPConnection: type
    Request: type


class _DirectResponseStream:
    """Expose a direct httpcore body while owning its connection lifetime."""

    def __init__(
        self,
        stream: object,
        connection: _DirectConnection,
        release: Callable[[_DirectConnection], None],
    ) -> None:
        self._stream = stream
        self._connection = connection
        self._release = release
        self._closed = False

    def __iter__(self) -> Iterator[bytes]:
        """Yield raw bytes for httpx2's decoded response iterator."""
        return iter(cast("Iterator[bytes]", self._stream))

    def close(self) -> None:
        """Close the body and direct connection without a pool lock."""
        if self._closed:
            return
        self._closed = True
        try:
            close_stream = getattr(self._stream, "close", None)
            if callable(close_stream):
                close_stream()
        finally:
            try:
                self._connection.close()
            finally:
                self._release(self._connection)


class _DirectHTTPTransport:
    """Provider-owned httpx2 transport with one httpcore connection per request."""

    def __init__(
        self,
        httpx_module: object,
        httpcore_module: object,
        ssl_context: object,
        network_backend: _NetworkBackend,
    ) -> None:
        httpx_module = cast("_HttpxModule", httpx_module)
        self._httpx_module = httpx_module
        self._httpcore_module = httpcore_module
        self._ssl_context = ssl_context
        self._network_backend = network_backend
        self._lock = threading.Lock()
        self._connections: set[_DirectConnection] = set()
        self._closed = False
        self._response_stream_type = type(
            "_ProviderDirectResponseStream",
            (_DirectResponseStream, httpx_module.SyncByteStream),
            {},
        )

    def __enter__(self) -> Self:
        """Support the synchronous httpx2 transport context protocol."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close provider-owned connections at transport scope exit."""
        self.close()

    def _release(self, connection: _DirectConnection) -> None:
        """Forget one connection after its response stream has closed."""
        with self._lock:
            self._connections.discard(connection)

    def handle_request(self, request: object) -> object:
        """Send one request over a fresh direct httpcore connection."""
        httpcore_module = cast("_HttpcoreModule", self._httpcore_module)
        httpx_module = cast("_HttpxModule", self._httpx_module)
        request = cast("_HttpxRequest", request)
        request_url = request.url
        core_url_type = cast("type", httpcore_module.URL)
        core_url = core_url_type(
            scheme=request_url.raw_scheme,
            host=request_url.raw_host,
            port=request_url.port,
            target=request_url.raw_path,
        )
        connection_type = cast("type", httpcore_module.HTTPConnection)
        connection = cast(
            "_DirectConnection",
            connection_type(
                origin=core_url.origin,
                ssl_context=self._ssl_context,
                http1=True,
                http2=False,
                network_backend=self._network_backend,
            ),
        )
        with self._lock:
            if self._closed:
                connection.close()
                message = "The HTTP transport is closed"
                raise RuntimeError(message)
            self._connections.add(connection)

        try:
            core_request_type = cast("type", httpcore_module.Request)
            core_request = core_request_type(
                method=request.method,
                url=core_url,
                headers=request.headers.raw,
                content=request.stream,
                extensions=request.extensions,
            )
            core_response = cast(
                "_DirectCoreResponse", connection.handle_request(core_request)
            )
            response_type = cast("type", httpx_module.Response)
            response_stream = self._response_stream_type(
                core_response.stream,
                connection,
                self._release,
            )
            return response_type(
                status_code=core_response.status,
                headers=core_response.headers,
                stream=response_stream,
                extensions=core_response.extensions,
            )
        except BaseException:
            try:
                connection.close()
            finally:
                self._release(connection)
            raise

    def close(self) -> None:
        """Close all direct connections without a shared pool cleanup lock."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connections = tuple(self._connections)
            self._connections.clear()
        for connection in connections:
            connection.close()


class _DeadlineNetworkStream:
    """Apply the caller thread's absolute deadline to one network stream."""

    def __init__(self, stream: object) -> None:
        self._stream = cast("_NetworkStream", stream)

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        """Read with the smaller of httpcore's timeout and the deadline."""
        return self._stream.read(max_bytes, timeout=_request_io_timeout(timeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        """Write with the smaller of httpcore's timeout and the deadline."""
        self._stream.write(buffer, timeout=_request_io_timeout(timeout))

    def close(self) -> None:
        """Avoid potentially blocking transport cleanup after a deadline/error."""
        if _SUPPRESS_NETWORK_CLEANUP.get() or _request_deadline_expired():
            _force_close_stream_socket(self._stream)
            return
        self._stream.close()

    def start_tls(
        self,
        ssl_context: object,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> _DeadlineNetworkStream:
        """Upgrade the stream while retaining request-scoped deadline control."""
        stream = self._stream.start_tls(
            ssl_context,
            server_hostname=server_hostname,
            timeout=_request_io_timeout(timeout),
        )
        return type(self)(stream)

    def get_extra_info(self, info: str) -> object:
        """Forward optional stream metadata without exposing client graph state."""
        get_extra_info = getattr(self._stream, "get_extra_info", None)
        if not callable(get_extra_info):
            return None
        return get_extra_info(info)


class _DeadlineNetworkBackend:
    """Wrap httpcore's backend without sharing cancellation across requests."""

    def __init__(
        self,
        backend: object,
        pinned_address: str | tuple[str, ...] | None = None,
    ) -> None:
        self._backend = cast("_NetworkBackend", backend)
        self._pinned_address = pinned_address

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> _DeadlineNetworkStream:
        """Open one TCP stream with the active request deadline."""
        addresses = (
            (self._pinned_address,)
            if isinstance(self._pinned_address, str)
            else self._pinned_address or (host,)
        )
        last_error: Exception | None = None
        for address in addresses:
            try:
                stream = self._backend.connect_tcp(
                    address,
                    port,
                    timeout=_request_io_timeout(timeout),
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as error:
                if not _is_retryable_connect_error(error):
                    raise
                last_error = error
            else:
                return _DeadlineNetworkStream(stream)
        if last_error is not None:
            raise last_error
        raise OSError from None

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: object = None,
    ) -> _DeadlineNetworkStream:
        """Open one Unix stream with the active request deadline."""
        stream = self._backend.connect_unix_socket(
            path,
            timeout=_request_io_timeout(timeout),
            socket_options=socket_options,
        )
        return _DeadlineNetworkStream(stream)

    def sleep(self, seconds: float) -> None:
        """Bound backend retry sleeps by the active deadline."""
        self._backend.sleep(min(seconds, _request_io_timeout(None) or seconds))


def _is_retryable_connect_error(error: Exception) -> bool:
    """Return whether a failed TCP attempt can safely try another address."""
    return isinstance(error, (OSError, TimeoutError)) or type(error).__name__ in {
        "ConnectError",
        "ConnectTimeout",
    }


def _deadline_transport(
    transport: HttpTransportConfig,
    *,
    pinned_address: str | tuple[str, ...] | None = None,
) -> object | None:
    """Create a direct httpx2 transport whose operations honor our deadline.

    The private httpx2/httpcore2 transport seam below is supported for versions
    2.0.0 through 2.12.0, matching the declared ``httpx2>=2.0,<2.13`` bound.
    httpx2 does not expose the httpcore SSL context or network backend in its
    transport constructor.  Build an ordinary transport once to obtain those
    deployment settings, then use them in a provider-owned direct
    ``HTTPConnection`` for each request.  A direct connection avoids the
    ``PoolByteStream`` ownership lock, while the returned httpx2 response still
    performs normal content decoding.
    """
    try:
        httpx_module = importlib.import_module("httpx2")
        httpcore_module = importlib.import_module("httpcore2")
        transport_type = getattr(httpx_module, "HTTPTransport", None)
        if not isinstance(transport_type, type):
            raise DependencyUnavailableError
        source_transport = transport_type(
            verify=transport.verify,
            cert=transport.cert,
            trust_env=False,
            http1=True,
            http2=False,
        )
        try:
            pool = getattr(source_transport, "_pool", None)
            ssl_context = None if pool is None else getattr(pool, "_ssl_context", None)
            backend = None if pool is None else getattr(pool, "_network_backend", None)
        finally:
            source_transport.close()
        if (
            ssl_context is None
            or backend is None
            or not all(
                callable(getattr(backend, name, None))
                for name in ("connect_tcp", "connect_unix_socket", "sleep")
            )
        ):
            message = (
                "The installed HTTP client does not expose the supported transport seam"
            )
            raise ProviderConfigurationError(message)
        return _DirectHTTPTransport(
            httpx_module,
            httpcore_module,
            ssl_context,
            _DeadlineNetworkBackend(backend, pinned_address),
        )
    except ProviderError:
        raise
    except ImportError as exc:
        raise DependencyUnavailableError from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ProviderConfigurationError from exc


def _read_streamed_response(
    response: _StreamResponse, *, deadline: float
) -> _HttpResponse:
    """Read at most one byte beyond the response limit from a streamed body."""
    _remaining_timeout(deadline)
    declared_length = (
        None
        if _has_non_identity_content_encoding(response)
        else _declared_content_length(response)
    )
    if declared_length is not None and declared_length > _MAX_RESPONSE_BYTES:
        _invalid_response("model response is too large")

    content = bytearray()
    chunk_size = min(_RESPONSE_READ_CHUNK_BYTES, _MAX_RESPONSE_BYTES + 1)
    try:
        body_chunks = response.iter_bytes(chunk_size=chunk_size)
        for chunk in body_chunks:
            _remaining_timeout(deadline)
            if not isinstance(chunk, bytes):
                _invalid_response("model response has invalid content")
            remaining = _MAX_RESPONSE_BYTES + 1 - len(content)
            content.extend(chunk[:remaining])
            if len(content) > _MAX_RESPONSE_BYTES:
                _invalid_response("model response is too large")
            _remaining_timeout(deadline)
        _remaining_timeout(deadline)
    except ProviderError:
        raise
    except Exception as exc:
        _remaining_timeout(deadline)
        if isinstance(exc, (AttributeError, TypeError, ValueError)) or (
            _is_httpx2_decoding_error(exc)
        ):
            message = "model response has invalid content"
            raise ProviderInvalidResponseError(message) from exc
        raise
    return _BufferedResponse(_response_status(response), bytes(content))


def _check_response_status(response: _HttpResponse, message: str) -> None:
    """Map a response status to the provider's stable error categories."""
    status_code = _response_status(response)
    if status_code in {401, 403, 404, 408, 429} or status_code >= _HTTP_SERVER_ERROR:
        raise ProviderUnavailableError
    if status_code < _HTTP_SUCCESS_MIN or status_code >= _HTTP_REDIRECT:
        _invalid_response(message)


def _check_metadata_status(response: _HttpResponse) -> None:
    """Map metadata endpoint statuses to canonical provider errors."""
    _check_response_status(response, "Model metadata request was rejected")


def _check_inference_status(response: _HttpResponse) -> None:
    """Map inference endpoint statuses before consuming an untrusted body."""
    _check_response_status(response, "Model inference request was rejected")


def _check_status_with_deadline(
    response: _HttpResponse,
    *,
    deadline: float,
    status_checker: Callable[[_HttpResponse], None],
) -> None:
    """Classify status only if the checker finished before the deadline."""
    _remaining_timeout(deadline)
    try:
        status_checker(response)
    except ProviderError:
        _remaining_timeout(deadline)
        raise
    _remaining_timeout(deadline)


def _is_httpx2_client(client_type: type) -> bool:
    """Return whether the client uses the provider-owned httpx2 transport."""
    client_module = getattr(client_type, "__module__", "")
    return client_module == "httpx2" or client_module.startswith("httpx2.")


def _create_client(
    client_type: type,
    transport: HttpTransportConfig,
    timeout_seconds: float,
    *,
    pinned_address: str | tuple[str, ...] | None = None,
) -> _HttpClient:
    """Construct one synchronous client with deployment-owned settings."""
    try:
        headers = dict(transport.headers)
        is_httpx2_client = _is_httpx2_client(client_type)
        client_kwargs: dict[str, object] = {
            "headers": headers,
            "verify": transport.verify,
            "cert": transport.cert,
            "follow_redirects": False,
            "trust_env": False,
            "timeout": timeout_seconds,
        }
        if is_httpx2_client:
            if not any(name.lower() == "accept-encoding" for name in headers):
                headers["Accept-Encoding"] = "identity"
            client_kwargs["transport"] = _deadline_transport(
                transport,
                pinned_address=pinned_address,
            )
        client = client_type(
            **client_kwargs,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ProviderConfigurationError from exc
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderUnavailableError from exc
    return cast("_HttpClient", client)


def _close_safely(client: _HttpClient) -> None:
    """Close a client during failed construction without masking its root error."""
    try:
        client.close()
    except Exception:  # noqa: BLE001
        return


def _is_direct_httpx2_response(response: object) -> bool:
    """Return whether response cleanup is backed by a direct connection."""
    module = getattr(type(response), "__module__", "")
    if module != "httpx2" and not module.startswith("httpx2."):
        return False
    stream = getattr(response, "stream", None)
    if isinstance(stream, _DirectResponseStream):
        return True
    return isinstance(getattr(stream, "_stream", None), _DirectResponseStream)


def _is_direct_httpx2_client(client: object) -> bool:
    """Return whether the client uses the provider's bounded direct transport."""
    transport = getattr(client, "_transport", None)
    return isinstance(transport, _DirectHTTPTransport)


def _uses_streaming_fallback(client: object) -> bool:
    """Return whether failed inference can retain an unbounded fallback stream."""
    if _is_direct_httpx2_client(client):
        return False
    if callable(getattr(client, "stream", None)):
        return True
    return callable(getattr(client, "build_request", None)) and callable(
        getattr(client, "send", None)
    )


class _ResponseContext:
    """Context adapter that closes a directly streamed response."""

    def __init__(self, response: object) -> None:
        self._response = response

    def __enter__(self) -> object:
        """Return the response without performing another synchronous request."""
        return self._response

    def __exit__(self, *_args: object) -> None:
        """Release the response when cleanup is safe to run synchronously."""
        close = getattr(self._response, "close", None)
        if callable(close):
            close()


class _DirectResponseContext(_ResponseContext):
    """Context adapter for the provider-owned direct httpx2 response stream."""

    def close_nonblocking(self) -> None:
        """Close the direct response through its provider-owned stream."""
        close = getattr(self._response, "close", None)
        if callable(close):
            close()


class _ResponseWithNonBlockingCleanup(_ResponseContext):
    """Adapt a bare fallback response with an explicit cleanup capability."""

    def close_nonblocking(self) -> None:
        """Delegate to the response's explicitly bounded cleanup method."""
        response = cast("_NonBlockingResponseCleanup", self._response)
        response.close_nonblocking()


def _response_context(response: object) -> _ResponseContext:
    """Adapt one response while retaining any explicit cleanup capability."""
    if _is_direct_httpx2_response(response):
        return _DirectResponseContext(response)
    if isinstance(response, _NonBlockingResponseCleanup):
        return _ResponseWithNonBlockingCleanup(response)
    return _ResponseContext(response)


def _stream_context(stream_context: object) -> AbstractContextManager[object]:
    """Normalize a streamed response or context manager to one context boundary."""
    if hasattr(stream_context, "__enter__") and hasattr(stream_context, "__exit__"):
        return cast("AbstractContextManager[object]", stream_context)
    if callable(getattr(stream_context, "close", None)):
        return _response_context(stream_context)
    return nullcontext(stream_context)


class _StreamCleanupError(Exception):
    """Internal marker that keeps cleanup-only failures separately mappable."""

    def __init__(self, cause: BaseException) -> None:
        self.cause = cause
        super().__init__(str(cause))


def _exit_stream_context(
    context: AbstractContextManager[object],
    primary: BaseException | None,
    *,
    skip: bool = False,
) -> None:
    """Run stream cleanup without replacing an error from the response body."""
    if skip:
        return
    try:
        context.__exit__(
            type(primary) if primary is not None else None,
            primary,
            primary.__traceback__ if primary is not None else None,
        )
    except BaseException:
        if primary is None:
            raise


def _read_stream_context(
    stream_context: object,
    *,
    deadline: float,
    status_checker: Callable[[_HttpResponse], None],
) -> _HttpResponse:
    """Read one response while preserving primary errors and bounded cleanup."""
    context = _stream_context(stream_context)

    entered = False
    response: object | None = None
    primary: BaseException | None = None
    suppression_token = _SUPPRESS_NETWORK_CLEANUP.set(True)
    try:
        response = context.__enter__()
        entered = True
        _check_status_with_deadline(
            cast("_HttpResponse", response),
            deadline=deadline,
            status_checker=status_checker,
        )
        return _read_streamed_response(
            cast("_StreamResponse", response), deadline=deadline
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if entered:
            skip_cleanup = primary is not None or _request_deadline_expired()
            if not skip_cleanup:
                _SUPPRESS_NETWORK_CLEANUP.reset(suppression_token)
                suppression_token = None
            try:
                if skip_cleanup:
                    if isinstance(context, _NonBlockingResponseCleanup):
                        context.close_nonblocking()
                else:
                    _exit_stream_context(context, primary)
            except BaseException as cleanup_error:
                if primary is None:
                    raise _StreamCleanupError(cleanup_error) from cleanup_error
        if suppression_token is not None:
            _SUPPRESS_NETWORK_CLEANUP.reset(suppression_token)


def _request_response(
    client: _HttpClient,
    method: str,
    url: str,
    *,
    deadline: float,
    status_checker: Callable[[_HttpResponse], None],
    **kwargs: object,
) -> _HttpResponse:
    """Request one response through a bounded stream when the client supports it."""
    timeout_seconds = _remaining_timeout(deadline)
    deadline_token = _REQUEST_DEADLINE.set(deadline)
    try:
        response = _request_streamed_response(
            client,
            method,
            url,
            timeout_seconds=timeout_seconds,
            deadline=deadline,
            status_checker=status_checker,
            **kwargs,
        )
        if response is not None:
            return response

        response = _request_materialized_response(
            client,
            method,
            url,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        _check_status_with_deadline(
            response,
            deadline=deadline,
            status_checker=status_checker,
        )
        _check_response_size(response)
    except _StreamCleanupError as exc:
        if isinstance(exc.cause, ProviderError):
            raise exc.cause from exc
        if _is_timeout_error(exc.cause):
            raise ProviderDeadlineError from exc.cause
        raise ProviderUnavailableError from exc.cause
    except ProviderError:
        raise
    except Exception as exc:
        if time.monotonic() >= deadline or _is_timeout_error(exc):
            raise ProviderDeadlineError from exc
        raise ProviderUnavailableError from exc
    else:
        return response
    finally:
        _REQUEST_DEADLINE.reset(deadline_token)


def _request_streamed_response(  # noqa: PLR0913
    client: _HttpClient,
    method: str,
    url: str,
    *,
    timeout_seconds: float,
    deadline: float,
    status_checker: Callable[[_HttpResponse], None],
    **kwargs: object,
) -> _HttpResponse | None:
    """Acquire and consume a streamed response when the client supports it."""
    build_request = getattr(client, "build_request", None)
    send = getattr(client, "send", None)
    if callable(build_request) and callable(send):
        request = build_request(
            method=method,
            url=url,
            timeout=timeout_seconds,
            **kwargs,
        )
        response = send(request, stream=True)
        return _read_stream_context(
            _response_context(response),
            deadline=deadline,
            status_checker=status_checker,
        )

    stream = getattr(client, "stream", None)
    if not callable(stream):
        return None
    stream_context = stream(
        method,
        url,
        timeout=timeout_seconds,
        **kwargs,
    )
    return _read_stream_context(
        stream_context,
        deadline=deadline,
        status_checker=status_checker,
    )


def _request_materialized_response(
    client: _HttpClient,
    method: str,
    url: str,
    *,
    timeout_seconds: float,
    **kwargs: object,
) -> _HttpResponse:
    """Use the legacy non-streaming client method when streaming is unavailable."""
    request = getattr(client, method.lower())
    return request(url, timeout=timeout_seconds, **kwargs)


def _metadata_response(client: _HttpClient, url: str, deadline: float) -> _HttpResponse:
    """Fetch one metadata response and classify transport failures."""
    return _request_response(
        client,
        "GET",
        url,
        deadline=deadline,
        status_checker=_check_metadata_status,
    )


def _output_width(shape: tuple[int, ...]) -> int:
    """Return the normalized per-row output width from negotiated metadata."""
    if len(shape) == 1:
        return 1 if shape[0] == -1 else shape[0]
    return shape[1]


class KServeV2HttpPredictionProvider(PredictionProvider):
    """One synchronous, metadata-negotiated KServe V2 provider."""

    def __init__(  # noqa: PLR0913
        self,
        client: _HttpClient,
        metadata: PredictionMetadata,
        spec: KServeModelSpec,
        base_url: str,
        max_batch_size: int,
        default_timeout: float,
    ) -> None:
        """Initialize a provider that owns the supplied synchronous client."""
        self._client = client
        self._metadata = metadata
        self._spec = spec
        self._infer_url = _model_url(base_url, spec, infer=True)
        self._max_batch_size = max_batch_size
        self._default_timeout = default_timeout
        self._retire_on_inference_failure = _uses_streaming_fallback(client)
        self._closed = False
        self._metadata_latency_seconds = 0.0
        self._inference_latency_seconds = 0.0
        self._inference_batch_count = 0

    @property
    def metadata(self) -> PredictionMetadata:
        """Return the typed metadata negotiated during connect."""
        return self._metadata

    @property
    def metadata_latency_seconds(self) -> float:
        """Return metadata negotiation latency in seconds."""
        return self._metadata_latency_seconds

    @property
    def inference_latency_seconds(self) -> float:
        """Return accumulated inference HTTP latency in seconds."""
        return self._inference_latency_seconds

    @property
    def inference_batch_count(self) -> int:
        """Return the number of inference requests sent by this provider."""
        return self._inference_batch_count

    @property
    def provider_latency(self) -> float:
        """Return metadata plus inference latency in seconds."""
        return self._metadata_latency_seconds + self._inference_latency_seconds

    @classmethod
    def connect(
        cls,
        spec: KServeModelSpec,
        transport: HttpTransportConfig,
        *,
        timeout_seconds: float,
    ) -> KServeV2HttpPredictionProvider:
        """Create one client, negotiate metadata, and return its provider."""
        timeout = _validate_connect_timeout(timeout_seconds)
        deadline = time.monotonic() + timeout
        base_url = _normalize_base_url(spec.base_url)
        if urlsplit(base_url).scheme == "http" and transport.has_sensitive_credentials:
            _configuration_error(
                "HTTPS is required for model URLs with transport credentials"
            )
        if not transport.allowed_hosts:
            _configuration_error("Outbound model host allowlist is required")
        if transport.follow_redirects or transport.trust_env:
            _configuration_error(
                "Redirects and ambient proxy settings must remain disabled"
            )
        if not transport.allows_url(base_url):
            _invalid_request("Model host is not in the outbound host allowlist")
        metadata_url = _model_url(base_url, spec, infer=False)

        try:
            client_type = _load_http_client()
        except DependencyUnavailableError:
            raise
        except ImportError as exc:
            raise DependencyUnavailableError from exc

        pinned_address = (
            resolve_outbound_addresses(
                base_url,
                timeout_seconds=_remaining_timeout(deadline),
                allow_private_addresses=transport.allows_private_url(base_url),
            )
            if _is_httpx2_client(client_type)
            else None
        )
        client = _create_client(
            client_type,
            transport,
            timeout,
            pinned_address=pinned_address,
        )
        try:
            started = time.monotonic()
            response = _metadata_response(client, metadata_url, deadline)
            _remaining_timeout(deadline)
            try:
                payload = response.json()
            except (AttributeError, TypeError, ValueError) as exc:
                raise ProviderInvalidResponseError from exc
            metadata = codec.parse_metadata(
                payload,
                spec.model_name,
                spec.model_version,
                input_name=spec.input_name,
                output_name=spec.output_name,
            )
            provider = cls(
                client,
                metadata,
                spec,
                base_url,
                transport.max_batch_size,
                timeout,
            )
            _remaining_timeout(deadline)
            provider._metadata_latency_seconds = time.monotonic() - started
            _remaining_timeout(deadline)
        except Exception:
            _close_safely(client)
            raise
        return provider

    def _batch_size(self) -> int:
        """Return the safe request batch size for negotiated fixed-batch metadata."""
        batch_size = self._max_batch_size
        if (
            len(self._metadata.input_shape) == _MATRIX_RANK
            and self._metadata.input_shape[0] == 1
        ) or (
            len(self._metadata.output_shape) == _MATRIX_RANK
            and self._metadata.output_shape[0] == 1
        ):
            return 1
        return batch_size

    def _post_inference(
        self, body: Mapping[str, object], deadline: float
    ) -> _HttpResponse:
        """Issue one inference request and account for its HTTP latency."""
        started = time.monotonic()
        try:
            response = _request_response(
                self._client,
                "POST",
                self._infer_url,
                deadline=deadline,
                status_checker=_check_inference_status,
                json=dict(body),
                headers={"Content-Type": "application/json"},
            )
        except ProviderError:
            raise
        except Exception as exc:
            if _is_timeout_error(exc):
                raise ProviderDeadlineError from exc
            raise ProviderUnavailableError from exc
        else:
            self._inference_batch_count += 1
            return response
        finally:
            self._inference_latency_seconds += time.monotonic() - started

    def predict(
        self,
        inputs: np.ndarray,
        *,
        timeout_seconds: float | None = None,
    ) -> np.ndarray:
        """Send bounded batches and concatenate decoded outputs in input order."""
        if self._closed:
            message = "Model provider is closed"
            raise ProviderUnavailableError(message)
        try:
            values = np.asarray(inputs)
        except (TypeError, ValueError) as exc:
            raise ProviderInvalidRequestError from exc
        if values.ndim != _MATRIX_RANK:
            _invalid_request("Model inputs must be a finite numeric matrix")

        effective_timeout = self._default_timeout
        if timeout_seconds is not None:
            if (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(float(timeout_seconds))
                or float(timeout_seconds) <= 0
            ):
                raise ProviderDeadlineError
            effective_timeout = float(timeout_seconds)
        deadline = time.monotonic() + effective_timeout

        if values.shape[0] == 0:
            codec.encode_request(
                self._metadata, values, output_name=self._spec.output_name
            )
            _remaining_timeout(deadline)
            return np.empty((0, _output_width(self._metadata.output_shape)))

        batch_size = self._batch_size()
        outputs: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            chunk = values[start : start + batch_size]
            body = codec.encode_request(
                self._metadata, chunk, output_name=self._spec.output_name
            )
            try:
                _remaining_timeout(deadline)
                response = self._post_inference(body, deadline)
                _remaining_timeout(deadline)
                output = codec.decode_response(
                    response,
                    self._metadata,
                    self._spec.model_name,
                    self._spec.model_version,
                    len(chunk),
                )
                _remaining_timeout(deadline)
            except ProviderError:
                # A fallback streamed response may have no bounded response
                # cleanup capability. Retire the owned client immediately so
                # its resources do not survive until a later provider.close().
                if self._retire_on_inference_failure:
                    self.close()
                raise
            outputs.append(output)
        _remaining_timeout(deadline)
        result = np.concatenate(outputs, axis=0)
        _remaining_timeout(deadline)
        return result

    def close(self) -> None:
        """Close the owned client idempotently."""
        if self._closed:
            return
        self._closed = True
        _close_safely(self._client)
