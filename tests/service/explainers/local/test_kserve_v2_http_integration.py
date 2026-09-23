"""Direct loopback integration tests for the outbound KServe V2 provider."""

from __future__ import annotations

import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import pytest

from trustyai_service.service.explainers.local import kserve_v2_http as provider_module
from trustyai_service.service.explainers.local.kserve_v2_http import (
    KServeModelSpec,
    KServeV2HttpPredictionProvider,
)
from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    ProviderDeadlineError,
    ProviderInvalidResponseError,
)


class _KServeHandler(BaseHTTPRequestHandler):
    """Serve one small, strict KServe V2 model contract over a real socket."""

    metadata_paths: ClassVar[list[str]] = []
    infer_paths: ClassVar[list[str]] = []
    requests: ClassVar[list[dict[str, object]]] = []

    def _send_json(self, payload: object) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        type(self).metadata_paths.append(self.path)
        self._send_json(
            {
                "name": "model name",
                "versions": ["version 1"],
                "inputs": [
                    {
                        "name": "features",
                        "datatype": "FP32",
                        "shape": [-1, 2],
                    }
                ],
                "outputs": [
                    {
                        "name": "unused",
                        "datatype": "FP32",
                        "shape": [-1, 1],
                    },
                    {
                        "name": "selected",
                        "datatype": "FP32",
                        "shape": [-1, 1],
                    },
                ],
            }
        )

    def do_POST(self) -> None:
        type(self).infer_paths.append(self.path)
        content_length = self.headers.get("Content-Length")
        assert content_length is not None
        request = json.loads(self.rfile.read(int(content_length)))
        type(self).requests.append(request)
        tensor = request["inputs"][0]
        rows = tensor["shape"][0]
        values = tensor["data"]
        self._send_json(
            {
                "model_name": "model name",
                "model_version": "version 1",
                "outputs": [
                    {
                        "name": "selected",
                        "datatype": "FP32",
                        "shape": [rows, 1],
                        "data": [
                            values[offset] + values[offset + 1]
                            for offset in range(0, len(values), 2)
                        ],
                    }
                ],
            }
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test server quiet."""
        del format, args


class _SlowDripHandler(BaseHTTPRequestHandler):
    """Serve valid KServe responses whose inference body arrives slowly."""

    metadata_body = json.dumps(
        {
            "name": "model",
            "versions": ["v1"],
            "inputs": [{"name": "features", "datatype": "FP32", "shape": [-1, 2]}],
            "outputs": [{"name": "prediction", "datatype": "FP32", "shape": [-1, 1]}],
        }
    ).encode()
    inference_body = json.dumps(
        {
            "model_name": "model",
            "outputs": [
                {
                    "name": "prediction",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [3.0],
                }
            ],
        }
    ).encode()
    drip_seconds: ClassVar[float] = 0.15
    expected_requests: ClassVar[int] = 0
    completed_requests: ClassVar[int] = 0
    all_requests_done: ClassVar[threading.Event] = threading.Event()

    def _send_body(self, body: bytes, *, drip: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        for value in body:
            self.wfile.write(bytes([value]))
            self.wfile.flush()
            if drip:
                time.sleep(type(self).drip_seconds)

    def do_GET(self) -> None:
        self._send_body(type(self).metadata_body, drip=False)

    def do_POST(self) -> None:
        content_length = self.headers.get("Content-Length")
        assert content_length is not None
        self.rfile.read(int(content_length))
        try:
            self._send_body(type(self).inference_body, drip=True)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            type(self).completed_requests += 1
            if type(self).completed_requests >= type(self).expected_requests:
                type(self).all_requests_done.set()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the slow test server quiet."""
        del format, args


class _SiblingHandler(BaseHTTPRequestHandler):
    """Serve one slow request alongside an independent fast request."""

    metadata_body = json.dumps(
        {
            "name": "model",
            "versions": ["v1"],
            "inputs": [{"name": "features", "datatype": "FP32", "shape": [-1, 2]}],
            "outputs": [{"name": "prediction", "datatype": "FP32", "shape": [-1, 1]}],
        }
    ).encode()
    inference_body = json.dumps(
        {
            "model_name": "model",
            "outputs": [
                {
                    "name": "prediction",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [3.0],
                }
            ],
        }
    ).encode()
    slow_started: ClassVar[threading.Event] = threading.Event()
    slow_finished: ClassVar[threading.Event] = threading.Event()
    fast_finished: ClassVar[threading.Event] = threading.Event()

    def _send(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self) -> None:
        self._send(type(self).metadata_body)

    def do_POST(self) -> None:
        content_length = self.headers.get("Content-Length")
        assert content_length is not None
        request = json.loads(self.rfile.read(int(content_length)))
        values = request["inputs"][0]["data"]
        if values[0] == 0.0:
            type(self).slow_started.set()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(type(self).inference_body)))
                self.end_headers()
                for value in type(self).inference_body:
                    self.wfile.write(bytes([value]))
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                type(self).slow_finished.set()
        else:
            self._send(type(self).inference_body)
            type(self).fast_finished.set()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test server quiet."""
        del format, args


class _PreBodyErrorHandler(BaseHTTPRequestHandler):
    """Return a response whose status or declared length fails before body read."""

    status_code: ClassVar[int] = 503
    body = b"{}"
    declared_length: ClassVar[int | None] = None

    def do_GET(self) -> None:
        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        content_length = (
            len(type(self).body)
            if type(self).declared_length is None
            else type(self).declared_length
        )
        self.send_header("Content-Length", str(content_length))
        self.end_headers()
        try:
            self.wfile.write(type(self).body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the test server quiet."""
        del format, args


class _GzipHandler(BaseHTTPRequestHandler):
    """Return metadata with an explicitly negotiated gzip content encoding."""

    body = json.dumps(
        {
            "name": "model",
            "versions": ["v1"],
            "inputs": [{"name": "features", "datatype": "FP32", "shape": [-1, 2]}],
            "outputs": [{"name": "prediction", "datatype": "FP32", "shape": [-1, 1]}],
        }
    ).encode()

    def do_GET(self) -> None:
        encoded = gzip.compress(type(self).body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the gzip test server quiet."""
        del format, args


class _MalformedGzipHandler(_GzipHandler):
    """Return an invalid gzip body with a valid content-encoding header."""

    body = b"not a gzip stream"

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)


class _SlowHeaderHandler(BaseHTTPRequestHandler):
    """Serve valid KServe responses whose headers arrive one byte at a time."""

    metadata_body = json.dumps(
        {
            "name": "model",
            "versions": ["v1"],
            "inputs": [{"name": "features", "datatype": "FP32", "shape": [-1, 2]}],
            "outputs": [{"name": "prediction", "datatype": "FP32", "shape": [-1, 1]}],
        }
    ).encode()
    inference_body = json.dumps(
        {
            "model_name": "model",
            "outputs": [
                {
                    "name": "prediction",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [3.0],
                }
            ],
        }
    ).encode()
    slow_methods: ClassVar[frozenset[str]] = frozenset()
    drip_seconds: ClassVar[float] = 0.01
    completed_requests: ClassVar[int] = 0
    expected_requests: ClassVar[int] = 0
    all_requests_done: ClassVar[threading.Event] = threading.Event()

    def _send_body(self, body: bytes) -> None:
        headers = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
        )
        try:
            if self.command in type(self).slow_methods:
                for value in headers:
                    self.connection.send(bytes([value]))
                    time.sleep(type(self).drip_seconds)
            else:
                self.connection.sendall(headers)
            self.connection.sendall(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            type(self).completed_requests += 1
            if type(self).completed_requests >= type(self).expected_requests:
                type(self).all_requests_done.set()

    def do_GET(self) -> None:
        self._send_body(type(self).metadata_body)

    def do_POST(self) -> None:
        content_length = self.headers.get("Content-Length")
        assert content_length is not None
        self.rfile.read(int(content_length))
        self._send_body(type(self).inference_body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Keep the slow-header test server quiet."""
        del format, args


def _start_slow_header_server(
    slow_methods: frozenset[str], expected_requests: int
) -> tuple[ThreadingHTTPServer, Thread]:
    """Start a loopback server with deterministic slow-header state."""
    _SlowHeaderHandler.slow_methods = slow_methods
    _SlowHeaderHandler.completed_requests = 0
    _SlowHeaderHandler.expected_requests = expected_requests
    _SlowHeaderHandler.all_requests_done = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHeaderHandler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _slow_header_transport(server: ThreadingHTTPServer) -> HttpTransportConfig:
    """Build an allowlisted transport for one loopback slow-header server."""
    return HttpTransportConfig(
        headers={},
        allowed_hosts=frozenset({f"127.0.0.1:{server.server_port}"}),
    )


def _slow_header_spec(server: ThreadingHTTPServer) -> KServeModelSpec:
    """Build the model spec used by the slow-header provider tests."""
    return KServeModelSpec(
        f"http://127.0.0.1:{server.server_port}",
        "model",
        None,
        "features",
        "prediction",
    )


@pytest.mark.parametrize(
    ("status_code", "declared_length", "max_response_bytes", "error_type"),
    [
        (400, None, None, provider_module.ProviderInvalidResponseError),
        (503, None, None, provider_module.ProviderUnavailableError),
        (200, 5, 4, provider_module.ProviderInvalidResponseError),
    ],
)
def test_actual_httpx2_releases_direct_connection_on_pre_body_provider_error(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    declared_length: int | None,
    max_response_bytes: int | None,
    error_type: type[Exception],
) -> None:
    """Release the direct connection before returning an error."""
    httpx2 = pytest.importorskip("httpx2")
    _PreBodyErrorHandler.status_code = status_code
    _PreBodyErrorHandler.declared_length = declared_length
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _PreBodyErrorHandler)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return

    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = HttpTransportConfig(
        headers={},
        allowed_hosts=frozenset({f"127.0.0.1:{server.server_port}"}),
    )
    client = provider_module._create_client(httpx2.Client, transport, 2.0)
    try:
        if max_response_bytes is not None:
            monkeypatch.setattr(
                provider_module, "_MAX_RESPONSE_BYTES", max_response_bytes
            )
        with pytest.raises(error_type):
            provider_module._request_response(
                client,
                "GET",
                f"http://127.0.0.1:{server.server_port}/v2/models/model",
                deadline=time.monotonic() + 1.0,
                status_checker=provider_module._check_metadata_status,
            )

        transport_instance = getattr(client, "_" + "transport")
        assert isinstance(transport_instance, provider_module._DirectHTTPTransport)
        assert getattr(transport_instance, "_" + "connections") == set()
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_legacy_httpx2_pool_lock_does_not_block_primary_error_return() -> None:  # noqa: PLR0915
    """Reproduce the old PoolByteStream stall and keep provider cleanup bounded."""
    httpx2 = pytest.importorskip("httpx2")
    _PreBodyErrorHandler.status_code = 400
    _PreBodyErrorHandler.declared_length = None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _PreBodyErrorHandler)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return

    server.daemon_threads = True
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    legacy_transport = httpx2.HTTPTransport(trust_env=False)
    legacy_client = httpx2.Client(transport=legacy_transport, trust_env=False)
    response = None
    release_pool_lock = threading.Event()
    pool_lock_held = threading.Event()
    cleanup_error: list[BaseException] = []
    cleanup_thread: Thread | None = None
    lock_thread: Thread | None = None
    try:
        url = f"http://127.0.0.1:{server.server_port}/v2/models/model"
        request = legacy_client.build_request("GET", url)
        response = legacy_client.send(request, stream=True)
        pool = getattr(legacy_transport, "_" + "pool")

        def hold_pool_lock() -> None:
            with getattr(pool, "_" + "optional_thread_lock"):
                pool_lock_held.set()
                release_pool_lock.wait(timeout=2.0)

        lock_thread = Thread(target=hold_pool_lock, daemon=True)
        lock_thread.start()
        assert pool_lock_held.wait(timeout=1.0)

        def invalid_iter_bytes(chunk_size: int | None = None) -> Iterator[bytes]:
            del chunk_size
            yield cast("bytes", "not bytes")

        response.iter_bytes = invalid_iter_bytes

        def read_response() -> None:
            try:
                provider_module._read_stream_context(
                    provider_module._ResponseContext(response),
                    deadline=time.monotonic() + 0.05,
                    status_checker=lambda _response: None,
                )
            except BaseException as exc:  # noqa: BLE001
                cleanup_error.append(exc)

        cleanup_thread = Thread(target=read_response, daemon=True)
        cleanup_thread.start()
        cleanup_thread.join(timeout=0.15)
        assert not cleanup_thread.is_alive()
        assert len(cleanup_error) == 1
        assert isinstance(
            cleanup_error[0], provider_module.ProviderInvalidResponseError
        )
    finally:
        release_pool_lock.set()
        if cleanup_thread is not None:
            cleanup_thread.join(timeout=1.0)
        if lock_thread is not None:
            lock_thread.join(timeout=1.0)
        if response is not None:
            response.close()
        legacy_client.close()
        server.shutdown()
        server_thread.join(timeout=2.0)
        server.server_close()


def test_actual_httpx2_decodes_explicit_non_identity_content_encoding() -> None:
    """Decode gzip bodies before enforcing the provider response contract."""
    httpx2 = pytest.importorskip("httpx2")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _GzipHandler)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return

    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = HttpTransportConfig(
        headers={"Accept-Encoding": "gzip"},
        allowed_hosts=frozenset({f"127.0.0.1:{server.server_port}"}),
    )
    client = provider_module._create_client(httpx2.Client, transport, 2.0)
    try:
        response = provider_module._request_response(
            client,
            "GET",
            f"http://127.0.0.1:{server.server_port}/v2/models/model",
            deadline=time.monotonic() + 1.0,
            status_checker=provider_module._check_metadata_status,
        )
        assert response.json() == {
            "name": "model",
            "versions": ["v1"],
            "inputs": [{"name": "features", "datatype": "FP32", "shape": [-1, 2]}],
            "outputs": [{"name": "prediction", "datatype": "FP32", "shape": [-1, 1]}],
        }
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_actual_httpx2_maps_malformed_content_encoding_to_invalid_response() -> None:
    """Map httpx2 decoding failures to invalid upstream response data."""
    httpx2 = pytest.importorskip("httpx2")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MalformedGzipHandler)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return

    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = HttpTransportConfig(
        headers={"Accept-Encoding": "gzip"},
        allowed_hosts=frozenset({f"127.0.0.1:{server.server_port}"}),
    )
    client = provider_module._create_client(httpx2.Client, transport, 2.0)
    try:
        with pytest.raises(ProviderInvalidResponseError, match="invalid content"):
            provider_module._request_response(
                client,
                "GET",
                f"http://127.0.0.1:{server.server_port}/v2/models/model",
                deadline=time.monotonic() + 1.0,
                status_checker=provider_module._check_metadata_status,
            )
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_provider_deadline_covers_real_slow_metadata_headers() -> None:
    """Abort repeated metadata acquisitions before headers finish arriving."""
    try:
        server, thread = _start_slow_header_server(frozenset({"GET"}), 3)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return

    elapsed: list[float] = []
    try:
        for _ in range(3):
            started = time.monotonic()
            with pytest.raises(ProviderDeadlineError):
                KServeV2HttpPredictionProvider.connect(
                    _slow_header_spec(server),
                    _slow_header_transport(server),
                    timeout_seconds=0.2,
                )
            elapsed.append(time.monotonic() - started)
        assert all(duration < 0.45 for duration in elapsed)
        assert _SlowHeaderHandler.all_requests_done.wait(timeout=2.0)
        assert not any(
            thread.name == "trustyai-provider-deadline"
            for thread in threading.enumerate()
        )
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_provider_deadline_covers_real_slow_inference_headers() -> None:
    """Abort inference acquisition before a loopback server finishes its headers."""
    try:
        server, thread = _start_slow_header_server(frozenset({"POST"}), 2)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return

    provider = None
    started = time.monotonic()
    try:
        provider = KServeV2HttpPredictionProvider.connect(
            _slow_header_spec(server),
            _slow_header_transport(server),
            timeout_seconds=2.0,
        )
        with pytest.raises(ProviderDeadlineError):
            provider.predict(np.ones((1, 2)), timeout_seconds=0.2)
        assert time.monotonic() - started < 0.45
        assert _SlowHeaderHandler.all_requests_done.wait(timeout=2.0)
        assert not any(
            thread.name == "trustyai-provider-deadline"
            for thread in threading.enumerate()
        )
    finally:
        if provider is not None:
            provider.close()
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_provider_returns_at_the_deadline_for_repeated_real_slow_drips() -> None:
    """Abort repeated loopback body reads at one deadline without leaking work."""
    _SlowDripHandler.completed_requests = 0
    _SlowDripHandler.expected_requests = 3
    _SlowDripHandler.all_requests_done = threading.Event()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowDripHandler)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return
    server.daemon_threads = True

    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = None
    elapsed: list[float] = []
    timeout_seconds = 0.2
    try:
        provider = KServeV2HttpPredictionProvider.connect(
            KServeModelSpec(
                f"http://127.0.0.1:{server.server_port}",
                "model",
                None,
                "features",
                "prediction",
            ),
            HttpTransportConfig(
                headers={},
                allowed_hosts=frozenset({f"127.0.0.1:{server.server_port}"}),
            ),
            timeout_seconds=2.0,
        )
        for _ in range(3):
            started = time.monotonic()
            with pytest.raises(ProviderDeadlineError):
                provider.predict(np.ones((1, 2)), timeout_seconds=timeout_seconds)
            elapsed.append(time.monotonic() - started)
    finally:
        if provider is not None:
            provider.close()
        assert _SlowDripHandler.all_requests_done.wait(timeout=2.0)
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()

    assert all(duration < timeout_seconds + 0.1 for duration in elapsed)
    assert not any(
        thread.name == "trustyai-provider-deadline" for thread in threading.enumerate()
    )


def test_provider_deadline_does_not_cancel_a_concurrent_sibling_request() -> None:
    """Keep a fast pooled request alive while a sibling body times out."""
    _SiblingHandler.slow_started = threading.Event()
    _SiblingHandler.slow_finished = threading.Event()
    _SiblingHandler.fast_finished = threading.Event()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _SiblingHandler)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = None
    try:
        provider = KServeV2HttpPredictionProvider.connect(
            KServeModelSpec(
                f"http://127.0.0.1:{server.server_port}",
                "model",
                None,
                "features",
                "prediction",
            ),
            HttpTransportConfig(
                headers={},
                allowed_hosts=frozenset({f"127.0.0.1:{server.server_port}"}),
            ),
            timeout_seconds=2.0,
        )

        def slow_prediction() -> None:
            with pytest.raises(ProviderDeadlineError):
                provider.predict(np.asarray([[0.0, 0.0]]), timeout_seconds=0.2)

        with ThreadPoolExecutor(max_workers=2) as executor:
            slow_future = executor.submit(slow_prediction)
            assert _SiblingHandler.slow_started.wait(timeout=1.0)
            fast_future = executor.submit(
                provider.predict,
                np.asarray([[1.0, 2.0]]),
                timeout_seconds=1.0,
            )
            np.testing.assert_allclose(fast_future.result(timeout=1.0), [[3.0]])
            slow_future.result(timeout=1.0)

        assert _SiblingHandler.slow_finished.wait(timeout=2.0)
        assert _SiblingHandler.fast_finished.is_set()
    finally:
        if provider is not None:
            provider.close()
        server.shutdown()
        thread.join(timeout=2.0)
        server.server_close()


def test_provider_calls_real_loopback_kserve_v2_metadata_and_infer() -> None:
    """Assert the negotiated names, shapes, version, and selected output on wire."""
    _KServeHandler.metadata_paths = []
    _KServeHandler.infer_paths = []
    _KServeHandler.requests = []
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _KServeHandler)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
        return

    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    provider = None
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/ingress/root/"
        provider = KServeV2HttpPredictionProvider.connect(
            KServeModelSpec(
                base_url,
                "model name",
                "version 1",
                "features",
                "selected",
            ),
            HttpTransportConfig(
                headers={"X-Deployment": "loopback"},
                allowed_hosts=frozenset({f"127.0.0.1:{server.server_port}"}),
                max_batch_size=2,
            ),
            timeout_seconds=5.0,
        )
        metadata = provider.metadata
        result = provider.predict(np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    finally:
        if provider is not None:
            provider.close()
            provider.close()
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert _KServeHandler.metadata_paths == [
        "/ingress/root/v2/models/model%20name/versions/version%201"
    ]
    assert metadata.input_name == "features"
    assert metadata.output_name == "selected"
    assert metadata.input_datatype == "FP32"
    assert metadata.output_datatype == "FP32"
    assert metadata.input_shape == (-1, 2)
    assert metadata.output_shape == (-1, 1)
    assert _KServeHandler.infer_paths == [
        "/ingress/root/v2/models/model%20name/versions/version%201/infer",
        "/ingress/root/v2/models/model%20name/versions/version%201/infer",
    ]
    assert len(_KServeHandler.requests) == 2
    assert _KServeHandler.requests[0] == {
        "inputs": [
            {
                "name": "features",
                "shape": [2, 2],
                "datatype": "FP32",
                "data": [1.0, 2.0, 3.0, 4.0],
            }
        ],
        "outputs": [{"name": "selected"}],
    }
    assert _KServeHandler.requests[1]["inputs"] == [
        {
            "name": "features",
            "shape": [1, 2],
            "datatype": "FP32",
            "data": [5.0, 6.0],
        }
    ]
    np.testing.assert_allclose(result, [[3.0], [7.0], [11.0]])
