"""Fixtures for true FastAPI-to-KServe local explainer integration tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar, Self

import numpy as np
import pytest


class FakeKServeHandler(BaseHTTPRequestHandler):
    """Minimal KServe V2 HTTP server used by endpoint integration tests."""

    metadata_calls = 0
    infer_calls: ClassVar[list[dict[str, Any]]] = []
    metadata_paths: ClassVar[list[str]] = []
    infer_paths: ClassVar[list[str]] = []
    classification = False

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress noisy standard-library HTTP server access logs."""
        del format, args

    def do_GET(self) -> None:
        """Serve the fake model metadata contract."""
        type(self).metadata_calls += 1
        type(self).metadata_paths.append(self.path)
        output_shape = [-1, 2] if type(self).classification else [-1, 1]
        payload = {
            "name": "m",
            "versions": ["v1"],
            "inputs": [{"name": "input", "datatype": "FP32", "shape": [-1, 2]}],
            "outputs": [{"name": "output", "datatype": "FP32", "shape": output_shape}],
        }
        self._write(payload)

    def do_POST(self) -> None:
        """Serve deterministic predictions for the submitted tensor."""
        type(self).infer_paths.append(self.path)
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).infer_calls.append(body)
        tensor = body["inputs"][0]
        rows = np.asarray(tensor["data"], dtype=float).reshape(tensor["shape"])
        if type(self).classification:
            probability = np.clip(0.5 + rows[:, 0] / 8.0, 0.1, 0.9)
            values = np.column_stack((1.0 - probability, probability))
        else:
            values = (rows.sum(axis=1) / 4.0).reshape(-1, 1)
        self._write(
            {
                "model_name": "m",
                "outputs": [
                    {
                        "name": "output",
                        "datatype": "FP32",
                        "shape": list(values.shape),
                        "data": values.reshape(-1).tolist(),
                    }
                ],
            }
        )

    def _write(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class FakeKServe:
    """Manage the lifetime of the loopback fake KServe server."""

    def __init__(self, *, classification: bool = False) -> None:
        """Configure the fake model's output task for one test server."""
        self.classification = classification

    def __enter__(self) -> Self:
        """Start the loopback server and return this context manager."""
        FakeKServeHandler.metadata_calls = 0
        FakeKServeHandler.infer_calls = []
        FakeKServeHandler.metadata_paths = []
        FakeKServeHandler.infer_paths = []
        FakeKServeHandler.classification = self.classification
        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeKServeHandler)
        except PermissionError:
            pytest.skip("loopback socket binding is unavailable in this environment")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the loopback server and join its worker thread."""
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


class LocalStorage:
    """Small in-memory storage implementation for endpoint tests."""

    def __init__(self) -> None:
        """Initialize one target, organic background, and synthetic row."""
        self.metadata = np.array(
            [
                ["target", "t", 0, []],
                ["organic-1", "t", 0, []],
                ["organic-2", "t", 0, []],
                ["synthetic", "t", 0, "_trustyai_synthetic"],
            ],
            dtype=object,
        )
        self.inputs = np.array(
            [[2.0, 2.0], [1.0, 0.0], [0.0, 1.0], [99.0, 99.0]],
            dtype=float,
        )
        self.outputs = np.array([[1.0], [0.25], [0.25], [99.0]], dtype=float)

    async def dataset_exists(self, _name: str) -> bool:
        """Report that all local-explanation datasets are available."""
        return True

    async def dataset_rows(self, _name: str) -> int:
        """Return the number of stored input rows."""
        return len(self.inputs)

    async def get_aliased_column_names(self, name: str) -> list[str]:
        """Return stable feature or output names for the requested dataset."""
        return ["f0", "f1"] if name.endswith("_inputs") else ["score"]

    async def read_data(
        self, name: str, start_row: int = 0, n_rows: int | None = None
    ) -> np.ndarray:
        """Read a bounded slice from the requested in-memory dataset."""
        values = self.metadata
        if name.endswith("_inputs"):
            values = self.inputs
        elif name.endswith("_outputs"):
            values = self.outputs
        end = None if n_rows is None else start_row + n_rows
        return values[start_row:end]
