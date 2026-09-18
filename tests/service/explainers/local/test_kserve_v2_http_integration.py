"""Loopback integration coverage for the real KServe V2 HTTP provider."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import ClassVar

import numpy as np
import pytest

from trustyai_service.service.explainers.local.kserve_v2_http import (
    KServeV2HttpPredictionProvider,
)
from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    KServeModelSpec,
)
from trustyai_service.service.explainers.local.types import TaskType


class _Handler(BaseHTTPRequestHandler):
    metadata_calls = 0
    infer_calls: ClassVar[list[list[float]]] = []

    def do_GET(self) -> None:
        type(self).metadata_calls += 1
        body = {
            "name": "m",
            "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 2]}],
            "outputs": [{"name": "output", "datatype": "FP32", "shape": [-1, 1]}],
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(size))
        rows = payload["inputs"][0]["shape"][0]
        values = payload["inputs"][0]["data"]
        type(self).infer_calls.append(values)
        body = {
            "model_name": "m",
            "outputs": [
                {
                    "name": "output",
                    "datatype": "FP32",
                    "shape": [rows, 1],
                    "data": [
                        sum(values[index : index + 2])
                        for index in range(0, len(values), 2)
                    ],
                }
            ],
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args


def test_provider_uses_real_loopback_kserve_http() -> None:
    _Handler.metadata_calls = 0
    _Handler.infer_calls = []
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    except PermissionError:
        pytest.skip("loopback socket binding is unavailable in this environment")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        provider = KServeV2HttpPredictionProvider.connect(
            KServeModelSpec(base_url, "m", None, None, None, TaskType.REGRESSION),
            5,
            HttpTransportConfig(
                headers={},
                allowed_hosts=frozenset({f"127.0.0.1:{server.server_port}"}),
                max_batch_size=2,
            ),
        )
        try:
            result = provider.predict(np.arange(10, dtype=float).reshape(5, 2))
        finally:
            provider.close()
        assert _Handler.metadata_calls == 1
        assert len(_Handler.infer_calls) == 3
        np.testing.assert_allclose(result.reshape(-1), [1, 5, 9, 13, 17])
    finally:
        server.shutdown()
        thread.join(timeout=5)
