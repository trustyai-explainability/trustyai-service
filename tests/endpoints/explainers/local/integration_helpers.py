"""Shared fixtures for local-explainer endpoint integration tests."""

from __future__ import annotations

import importlib
import json
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from typing import TYPE_CHECKING, Any, Self, cast
from urllib.parse import quote

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_MAIN_MODULE = "trustyai_service.main"
_MODULE_PREFIXES_TO_RESTORE = (
    _MAIN_MODULE,
    "trustyai_service.endpoints.explainers.local_explainer",
    "trustyai_service.endpoints.explainers.local_models",
    "trustyai_service.endpoints.explainers.local_lime",
    "trustyai_service.endpoints.explainers.local_shap",
    "trustyai_service.core.explainers.local",
    "trustyai_service.service.explainers.local",
    "lime",
    "shap",
    "httpx2",
)
_ENABLED_EXPLAINER_FLAGS = {
    "explainer": True,
    "explainer_local": True,
    "explainer_global": False,
}


class _InvalidFakeKServeRequestError(ValueError):
    """Marker for malformed requests rejected by the fake protocol."""


class _FakeKServeShutdownError(RuntimeError):
    """Raised when a fake server worker remains alive during cleanup."""


@dataclass
class _FakeKServeState:
    """Mutable state owned by one fake KServe server instance."""

    classification: bool
    model_name: str
    model_version: str | None
    input_name: str
    output_name: str
    feature_width: int
    metadata_status: int | None = None
    infer_status: int | None = None
    metadata_calls: int = 0
    infer_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata_paths: list[str] = field(default_factory=list)
    infer_paths: list[str] = field(default_factory=list)


class _FakeKServeHTTPServer(ThreadingHTTPServer):
    """Threading server carrying state for one fake KServe instance."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        state: _FakeKServeState,
    ) -> None:
        self.state = state
        super().__init__(address, handler)


class FakeKServeHandler(BaseHTTPRequestHandler):
    """Minimal deterministic KServe V2 server used by endpoint tests."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress noisy standard-library HTTP server access logs."""
        del format, args

    @property
    def state(self) -> _FakeKServeState:
        """Return the state attached to this request's server."""
        server = cast("_FakeKServeHTTPServer", self.server)
        return server.state

    def _model_paths(self, *, infer: bool) -> set[str]:
        """Return the accepted metadata or inference paths for this model."""
        state = self.state
        model_path = f"/v2/models/{quote(state.model_name, safe='-_.~')}"
        paths = {model_path}
        if state.model_version is not None:
            paths.add(
                f"{model_path}/versions/{quote(state.model_version, safe='-_.~')}"
            )
        return {f"{path}/infer" for path in paths} if infer else paths

    def do_GET(self) -> None:
        """Return deterministic numeric model metadata."""
        state = self.state
        state.metadata_paths.append(self.path)
        if self.path not in self._model_paths(infer=False):
            self._protocol_error()
            return
        state.metadata_calls += 1
        if state.metadata_status is not None:
            self._write(
                {"error": {"code": "upstream_failure", "message": "fake outage"}},
                status=state.metadata_status,
            )
            return
        output_width = 2 if state.classification else 1
        self._write(
            {
                "name": state.model_name,
                "versions": (
                    [] if state.model_version is None else [state.model_version]
                ),
                "inputs": [
                    {
                        "name": state.input_name,
                        "datatype": "FP32",
                        "shape": [-1, state.feature_width],
                    }
                ],
                "outputs": [
                    {
                        "name": state.output_name,
                        "datatype": "FP32",
                        "shape": [-1, output_width],
                    }
                ],
            }
        )

    def do_POST(self) -> None:
        """Record an inference batch and calculate output from its rows."""
        state = self.state
        state.infer_paths.append(self.path)
        if self.path not in self._model_paths(infer=True):
            self._protocol_error()
            return
        if state.infer_status is not None:
            self._write(
                {"error": {"code": "upstream_failure", "message": "fake outage"}},
                status=state.infer_status,
            )
            return

        try:
            body = self._read_json_body()
            rows = self._validate_inference_request(body)
        except (IndexError, OverflowError, TypeError, ValueError):
            self._protocol_error()
            return

        state.infer_calls.append(body)
        if state.classification:
            probability = np.clip(0.5 + rows[:, 0] / 8.0, 0.1, 0.9)
            values = np.column_stack((1.0 - probability, probability))
        else:
            values = (rows.sum(axis=1) / 4.0).reshape(-1, 1)
        self._write(
            {
                "model_name": state.model_name,
                "model_version": state.model_version,
                "outputs": [
                    {
                        "name": state.output_name,
                        "datatype": "FP32",
                        "shape": list(values.shape),
                        "data": values.reshape(-1).tolist(),
                    }
                ],
            }
        )

    def _read_json_body(self) -> dict[str, Any]:
        """Read a JSON object body or raise a local protocol validation error."""
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise _InvalidFakeKServeRequestError
        length = int(content_length)
        if length < 0:
            raise _InvalidFakeKServeRequestError
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise _InvalidFakeKServeRequestError
        return body

    def _validate_inference_request(self, body: dict[str, Any]) -> np.ndarray:
        """Validate the supported single-input, flat FP32 V2 request contract."""
        state = self.state
        inputs = body.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise _InvalidFakeKServeRequestError
        tensor = inputs[0]
        if not isinstance(tensor, Mapping):
            raise _InvalidFakeKServeRequestError
        if tensor.get("name") != state.input_name:
            raise _InvalidFakeKServeRequestError
        if tensor.get("datatype") != "FP32":
            raise _InvalidFakeKServeRequestError

        shape = tensor.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int)
                for dimension in shape
            )
        ):
            raise _InvalidFakeKServeRequestError
        batch, width = shape
        if batch <= 0 or width != state.feature_width:
            raise _InvalidFakeKServeRequestError

        data = tensor.get("data")
        if (
            not isinstance(data, list)
            or len(data) != batch * width
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in data
            )
        ):
            raise _InvalidFakeKServeRequestError
        values = np.asarray(data, dtype=float)
        if not np.isfinite(values).all():
            raise _InvalidFakeKServeRequestError

        outputs = body.get("outputs")
        if (
            not isinstance(outputs, list)
            or len(outputs) != 1
            or not isinstance(outputs[0], Mapping)
            or outputs[0].get("name") != state.output_name
        ):
            raise _InvalidFakeKServeRequestError
        return values.reshape(batch, width)

    def _protocol_error(self) -> None:
        """Return a stable error for any malformed protocol request."""
        self._write(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "invalid KServe V2 inference request",
                }
            },
            status=400,
        )

    def _write(self, payload: dict[str, Any], *, status: int = 200) -> None:
        """Write one JSON response with an explicit content length."""
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class FakeKServe:
    """Manage one loopback fake KServe V2 server."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        classification: bool = False,
        model_name: str = "m",
        model_version: str | None = "v1",
        input_name: str = "input",
        output_name: str = "output",
        feature_width: int = 2,
        metadata_status: int | None = None,
        infer_status: int | None = None,
    ) -> None:
        """Configure the deterministic model contract for one test server."""
        self._state = _FakeKServeState(
            classification=classification,
            model_name=model_name,
            model_version=model_version,
            input_name=input_name,
            output_name=output_name,
            feature_width=feature_width,
            metadata_status=metadata_status,
            infer_status=infer_status,
        )
        self.server: _FakeKServeHTTPServer
        self.thread: threading.Thread
        self.base_url: str

    @property
    def metadata_calls(self) -> int:
        """Return the number of valid metadata requests for this server."""
        return self._state.metadata_calls

    @property
    def infer_calls(self) -> list[dict[str, Any]]:
        """Return valid inference requests recorded by this server."""
        return self._state.infer_calls

    @property
    def metadata_paths(self) -> list[str]:
        """Return all metadata paths received by this server."""
        return self._state.metadata_paths

    @property
    def infer_paths(self) -> list[str]:
        """Return all inference paths received by this server."""
        return self._state.infer_paths

    def __enter__(self) -> Self:
        """Start the loopback server and return this context manager."""
        try:
            self.server = _FakeKServeHTTPServer(
                ("127.0.0.1", 0), FakeKServeHandler, self._state
            )
        except PermissionError:
            pytest.skip("loopback socket binding is unavailable in this environment")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the loopback server and join its worker thread."""
        try:
            self.server.shutdown()
        finally:
            self.thread.join(timeout=5)
            self.server.server_close()
        if self.thread.is_alive():
            raise _FakeKServeShutdownError


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
        """Read a bounded slice from an in-memory dataset."""
        values = self.metadata
        if name.endswith("_inputs"):
            values = self.inputs
        elif name.endswith("_outputs"):
            values = self.outputs
        end = None if n_rows is None else start_row + n_rows
        return values[start_row:end]


def _module_is_in_scope(name: str) -> bool:
    """Return whether a module belongs to the enabled-explainer import tree."""
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in _MODULE_PREFIXES_TO_RESTORE
    )


def _module_parent_names() -> set[str]:
    """Return parent packages whose child attributes can be changed by imports."""
    parents: set[str] = set()
    for prefix in _MODULE_PREFIXES_TO_RESTORE:
        parts = prefix.split(".")
        parents.update(".".join(parts[:index]) for index in range(1, len(parts)))
    return parents


def _snapshot_module_state() -> tuple[
    dict[str, ModuleType | None], dict[str, dict[str, object] | None]
]:
    """Capture scoped module entries and parent attributes before reloading."""
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if _module_is_in_scope(name)
    }
    parent_names = _module_parent_names() | {
        name for name, module in original_modules.items() if module is not None
    }
    original_parent_attributes = {
        name: (
            dict(module.__dict__)
            if isinstance(module := sys.modules.get(name), ModuleType)
            else None
        )
        for name in parent_names
    }
    return original_modules, original_parent_attributes


def _remove_new_scoped_modules(
    original_modules: dict[str, ModuleType | None],
) -> tuple[str, ...]:
    """Remove only scoped module entries that were absent at context entry."""
    current_scoped_names = tuple(
        name for name in sys.modules if _module_is_in_scope(name)
    )
    for name in list(sys.modules):
        if _module_is_in_scope(name) and name not in original_modules:
            sys.modules.pop(name, None)
    return current_scoped_names


def _restore_module_entries(original_modules: dict[str, ModuleType | None]) -> None:
    """Restore scoped module entries that existed before the context."""
    for name, module in original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _restore_parent_attributes(
    original_parent_attributes: dict[str, dict[str, object] | None],
    current_scoped_names: tuple[str, ...],
) -> None:
    """Restore import-created child attributes on relevant parent packages."""
    for name, attributes in original_parent_attributes.items():
        module = sys.modules.get(name)
        if not isinstance(module, ModuleType):
            continue
        if attributes is None:
            for child_module in current_scoped_names:
                prefix = f"{name}."
                if child_module.startswith(prefix):
                    child_name = child_module[len(prefix) :].split(".", 1)[0]
                    module.__dict__.pop(child_name, None)
            continue
        for attribute in tuple(module.__dict__):
            if attribute not in attributes:
                module.__dict__.pop(attribute, None)
        module.__dict__.update(attributes)


def _restore_module_state(
    original_modules: dict[str, ModuleType | None],
    original_parent_attributes: dict[str, dict[str, object] | None],
) -> None:
    """Restore scoped modules and parent attributes without touching other imports."""
    current_scoped_names = _remove_new_scoped_modules(original_modules)
    _restore_module_entries(original_modules)
    _restore_parent_attributes(original_parent_attributes, current_scoped_names)


@contextmanager
def main_with_feature_flags(
    overrides: Mapping[str, bool],
) -> Iterator[ModuleType]:
    """Reload ``trustyai_service.main`` with flags, then restore its state.

    Feature flags are changed before the application is imported or reloaded.
    The original flag values and optional route/provider module entries are
    restored when the context exits, including when application import fails.
    """
    feature_flags = importlib.import_module(
        "trustyai_service.service.config.feature_flags"
    )
    original_flags = feature_flags.ENDPOINTS.copy()  # type: ignore[attr-defined]
    original_modules, original_parent_attributes = _snapshot_module_state()

    feature_flags.ENDPOINTS.update(overrides)  # type: ignore[attr-defined]
    try:
        original_main = original_modules.get(_MAIN_MODULE)
        module = (
            importlib.reload(original_main)
            if original_main is not None
            else importlib.import_module(_MAIN_MODULE)
        )
        yield module
    finally:
        feature_flags.ENDPOINTS.clear()  # type: ignore[attr-defined]
        feature_flags.ENDPOINTS.update(original_flags)  # type: ignore[attr-defined]
        original_main = original_modules.get(_MAIN_MODULE)
        try:
            if original_main is not None:
                importlib.reload(original_main)
            else:
                sys.modules.pop(_MAIN_MODULE, None)
        finally:
            _restore_module_state(original_modules, original_parent_attributes)


@contextmanager
def enabled_test_client(
    overrides: Mapping[str, bool] | None = None,
) -> Iterator[Any]:
    """Yield a TestClient with local explainers enabled and restore state."""
    flags = dict(_ENABLED_EXPLAINER_FLAGS)
    if overrides is not None:
        flags.update(overrides)
    with main_with_feature_flags(flags) as module:
        from fastapi.testclient import TestClient  # noqa: PLC0415

        with TestClient(module.app) as client:
            yield client
