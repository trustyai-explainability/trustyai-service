"""Regression tests for local-explainer endpoint test infrastructure."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlsplit

from .factory import (
    LIME_MODULE,
    make_compute_endpoint_test,
    make_explainer_unavailable_test,
)
from .integration_helpers import FakeKServe


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, object]:
    """Send one JSON request through the standard-library HTTP client."""
    parsed = urlsplit(base_url)
    assert parsed.hostname is not None
    assert parsed.port is not None
    connection = HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    try:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {} if body is None else {"Content-Type": "application/json"}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def _valid_inference_payload() -> dict[str, object]:
    """Return one request matching the fake server's advertised contract."""
    return {
        "inputs": [
            {
                "name": "input",
                "datatype": "FP32",
                "shape": [1, 2],
                "data": [2.0, 2.0],
            }
        ],
        "outputs": [{"name": "output"}],
    }


def test_new_value_patch_generated_test_has_no_injected_argument() -> None:
    """Factories using patch(new=...) return zero-argument pytest tests."""
    generated = make_compute_endpoint_test(
        explainer_name="LIME",
        endpoint_path="/explainers/local/lime",
        client=None,  # type: ignore[arg-type]
        request_payload={"predictionId": "pred-123"},
        expected_response_keys=[],
        feature_names=["f0", "f1"],
        compute_thread_return=None,
        availability_flag=f"{__name__}.{LIME_MODULE}",
    )

    assert inspect.signature(generated).parameters == {}


def test_new_value_patch_unavailable_test_has_no_injected_argument() -> None:
    """The dependency-unavailable factory has the same zero-argument contract."""
    generated = make_explainer_unavailable_test(
        explainer_name="LIME",
        availability_flag=f"{__name__}.{LIME_MODULE}",
        endpoint_path="/explainers/local/lime",
        client=None,  # type: ignore[arg-type]
        request_payload={},
    )

    assert inspect.signature(generated).parameters == {}


def test_main_flag_context_restores_absent_module_tree_without_touching_unrelated() -> (
    None
):
    """Enabled imports do not leak local or optional modules into later tests."""
    repository = Path(__file__).resolve().parents[4]
    script = r"""
import sys
from types import ModuleType

from tests.endpoints.explainers.local.integration_helpers import (
    main_with_feature_flags,
)

relevant_prefixes = (
    "trustyai_service.main",
    "trustyai_service.endpoints.explainers.local_explainer",
    "trustyai_service.endpoints.explainers.local_models",
    "trustyai_service.endpoints.explainers.local_lime",
    "trustyai_service.endpoints.explainers.local_shap",
    "trustyai_service.service.explainers.local.kserve_v2_http",
    "lime",
    "shap",
    "httpx2",
)
for name in list(sys.modules):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in relevant_prefixes):
        sys.modules.pop(name, None)

unrelated = ModuleType("unrelated_preexisting")
sys.modules[unrelated.__name__] = unrelated

assert "trustyai_service.main" not in sys.modules
with main_with_feature_flags(
    {"explainer": True, "explainer_local": True, "explainer_global": False}
):
    assert "trustyai_service.main" in sys.modules
    import trustyai_service.endpoints.explainers.local_explainer  # noqa: F401
    import trustyai_service.endpoints.explainers.local_models  # noqa: F401

    for name in (
        "lime",
        "lime.lime_tabular",
        "shap",
        "shap._cext",
        "httpx2",
        "httpx2._client",
    ):
        module = ModuleType(name)
        sys.modules[name] = module
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, module)

for name in (
    "trustyai_service.main",
    "trustyai_service.endpoints.explainers.local_explainer",
    "trustyai_service.endpoints.explainers.local_models",
    "lime",
    "lime.lime_tabular",
    "shap",
    "shap._cext",
    "httpx2",
    "httpx2._client",
):
    assert name not in sys.modules, name

assert sys.modules["unrelated_preexisting"] is unrelated
explainer_package = sys.modules["trustyai_service.endpoints.explainers"]
assert not hasattr(explainer_package, "local_explainer")
assert not hasattr(explainer_package, "local_models")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_main_flag_context_restores_present_local_models_module_state() -> None:
    """Pre-existing local-model state survives the enabled reload unchanged."""
    repository = Path(__file__).resolve().parents[4]
    script = r"""
import importlib
import sys

from tests.endpoints.explainers.local.integration_helpers import (
    main_with_feature_flags,
)

local_models_name = "trustyai_service.endpoints.explainers.local_models"
provider_name = "trustyai_service.service.explainers.local.model_provider"
local_models = importlib.import_module(local_models_name)
provider = importlib.import_module(provider_name)
original_error = provider.ProviderInvalidRequestError
original_sentinel = object()
local_models._test_sentinel = original_sentinel

with main_with_feature_flags(
    {"explainer": True, "explainer_local": True, "explainer_global": False}
):
    importlib.reload(provider)
    importlib.reload(local_models)
    assert local_models.ProviderInvalidRequestError is not original_error
    assert local_models.ProviderInvalidRequestError is provider.ProviderInvalidRequestError
    local_models._test_sentinel = object()

assert sys.modules[local_models_name] is local_models
assert sys.modules[provider_name] is provider
assert provider.ProviderInvalidRequestError is original_error
assert local_models.ProviderInvalidRequestError is original_error
assert local_models._test_sentinel is original_sentinel
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_fake_kserve_rejects_unknown_paths_and_malformed_protocol_requests() -> None:
    """Malformed V2 requests receive one deterministic protocol error."""
    malformed_requests = [
        {
            "inputs": [],
            "outputs": [{"name": "output"}],
        },
        {
            "inputs": [
                {
                    "name": "wrong",
                    "datatype": "FP32",
                    "shape": [1, 2],
                    "data": [2.0, 2.0],
                }
            ],
            "outputs": [{"name": "output"}],
        },
        {
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP64",
                    "shape": [1, 2],
                    "data": [2.0, 2.0],
                }
            ],
            "outputs": [{"name": "output"}],
        },
        {
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP32",
                    "shape": [1, 2],
                    "data": [2.0, 2.0],
                },
                {
                    "name": "extra",
                    "datatype": "FP32",
                    "shape": [1, 2],
                    "data": [2.0, 2.0],
                },
            ],
            "outputs": [{"name": "output"}],
        },
        {
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP32",
                    "shape": [1, 2],
                    "data": [2.0, 2.0],
                }
            ],
            "outputs": [{"name": "wrong"}],
        },
        {
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP32",
                    "shape": [1, 3],
                    "data": [2.0, 2.0, 2.0],
                }
            ],
            "outputs": [{"name": "output"}],
        },
        {
            "inputs": [
                {
                    "name": "input",
                    "datatype": "FP32",
                    "shape": [2, 2],
                    "data": [2.0, 2.0],
                }
            ],
            "outputs": [{"name": "output"}],
        },
    ]

    with FakeKServe() as fake:
        expected = {
            "error": {
                "code": "invalid_request",
                "message": "invalid KServe V2 inference request",
            }
        }
        status, body = _request_json(fake.base_url, "GET", "/not-a-model")
        assert status == HTTPStatus.BAD_REQUEST
        assert body == expected

        for malformed in malformed_requests:
            status, body = _request_json(
                fake.base_url,
                "POST",
                "/v2/models/m/versions/v1/infer",
                malformed,
            )
            assert status == HTTPStatus.BAD_REQUEST
            assert body == expected


def test_fake_kserve_keeps_valid_inference_and_instance_state_isolated() -> None:
    """Overlapping fake servers retain their own model mode and recordings."""
    with FakeKServe() as regression, FakeKServe(classification=True) as classification:
        for fake, expected_width in ((regression, 1), (classification, 2)):
            status, body = _request_json(
                fake.base_url,
                "GET",
                "/v2/models/m/versions/v1",
            )
            assert status == HTTPStatus.OK
            assert body["outputs"][0]["shape"] == [-1, expected_width]  # type: ignore[index]

            payload = _valid_inference_payload()
            status, body = _request_json(
                fake.base_url,
                "POST",
                "/v2/models/m/versions/v1/infer",
                payload,
            )
            assert status == HTTPStatus.OK
            assert body["model_name"] == "m"  # type: ignore[index]
            assert fake.infer_calls == [payload]

        assert regression.metadata_calls == 1
        assert classification.metadata_calls == 1
        assert regression.infer_calls is not classification.infer_calls

    assert not regression.thread.is_alive()
    assert not classification.thread.is_alive()
