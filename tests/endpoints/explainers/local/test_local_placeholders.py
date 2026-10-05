"""Regression tests for local explainer routing, feature gates, and placeholders."""

from __future__ import annotations

import importlib
import subprocess
import sys
from http import HTTPStatus
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field

from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_explainer import build_router
from trustyai_service.service.config import feature_flags

_LIME_MODULE_NAME = "trustyai_service.endpoints.explainers.local_lime"
_SHAP_MODULE_NAME = "trustyai_service.endpoints.explainers.local_shap"
_ALGORITHM_MODULES = (_LIME_MODULE_NAME, _SHAP_MODULE_NAME)


class LimeExplanationConfig(BaseModel):
    """Minimal typed configuration for a fake LIME route."""

    model: dict[str, object]


class LimeExplanationRequest(BaseModel):
    """Canonical request contract for a fake LIME route."""

    model_config = ConfigDict(populate_by_name=True)

    prediction_id: str = Field(alias="predictionId")
    config: LimeExplanationConfig


class LIMEExplanationResponse(BaseModel):
    """Canonical response contract for a fake LIME route."""

    explanation: str


class SHAPExplanationRequest(BaseModel):
    """Canonical request contract for a fake SHAP route."""

    model_config = ConfigDict(populate_by_name=True)

    prediction_id: str = Field(alias="predictionId")
    config: dict[str, object]


class SHAPExplanationResponse(BaseModel):
    """Canonical response contract for a fake SHAP route."""

    explanation: str


def _module_available(module_name: str) -> bool:
    """Check optional module availability without importing its dependencies."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _registered_route_paths(router: APIRouter) -> list[str]:
    """Collect paths from direct and nested FastAPI router entries."""
    paths: list[str] = []
    for route in router.routes:
        nested_router = getattr(route, "original_router", None)
        if nested_router is not None:
            paths.extend(_registered_route_paths(nested_router))
            continue
        path = getattr(route, "path", None)
        if path is not None:
            paths.append(path)
    return paths


def _fake_algorithm_module(module_name: str) -> ModuleType:
    """Create a canonical implementation module for an absent sibling."""
    module = ModuleType(module_name)
    router = APIRouter()

    if module_name == _LIME_MODULE_NAME:

        @router.post(
            routes.EXPLAINER_LOCAL_LIME,
            response_model=LIMEExplanationResponse,
        )
        async def fake_lime_explanation(
            request: LimeExplanationRequest,
        ) -> LIMEExplanationResponse:
            """Expose the canonical LIME schemas without executing LIME."""
            del request
            return LIMEExplanationResponse(explanation="fake")

        module.router = router
        return module

    if module_name == _SHAP_MODULE_NAME:

        @router.post(
            routes.EXPLAINER_LOCAL_SHAP,
            response_model=SHAPExplanationResponse,
        )
        async def fake_shap_explanation(
            request: SHAPExplanationRequest,
        ) -> SHAPExplanationResponse:
            """Expose the canonical SHAP schemas without executing SHAP."""
            del request
            return SHAPExplanationResponse(explanation="fake")

        module.router = router
        return module

    raise AssertionError


def _implementation_contract(module_name: str) -> tuple[str, str]:
    """Return the canonical request and response schema names."""
    return {
        _LIME_MODULE_NAME: ("LimeExplanationRequest", "LIMEExplanationResponse"),
        _SHAP_MODULE_NAME: ("SHAPExplanationRequest", "SHAPExplanationResponse"),
    }[module_name]


@pytest.mark.filterwarnings("error:Duplicate Operation ID")
def test_enabled_shared_router_composes_both_algorithm_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose exactly one canonical LIME and SHAP route each."""
    monkeypatch.setitem(feature_flags.ENDPOINTS, "explainer", value=True)
    monkeypatch.setitem(feature_flags.ENDPOINTS, "explainer_local", value=True)

    for module_name in _ALGORITHM_MODULES:
        if not _module_available(module_name):
            monkeypatch.setitem(
                sys.modules,
                module_name,
                _fake_algorithm_module(module_name),
            )

    local_router = build_router()
    registered_paths = _registered_route_paths(local_router)
    for path in (
        routes.EXPLAINER_LOCAL_LIME,
        routes.EXPLAINER_LOCAL_SHAP,
    ):
        assert registered_paths.count(path) == 1

    app = FastAPI()
    app.include_router(local_router)
    paths = app.openapi()["paths"]

    for module_name, path in (
        (_LIME_MODULE_NAME, routes.EXPLAINER_LOCAL_LIME),
        (_SHAP_MODULE_NAME, routes.EXPLAINER_LOCAL_SHAP),
    ):
        request_schema, response_schema = _implementation_contract(module_name)
        operation = paths[path]["post"]
        assert operation["requestBody"]["content"]["application/json"]["schema"] == {
            "$ref": f"#/components/schemas/{request_schema}"
        }
        assert operation["responses"]["200"]["content"]["application/json"][
            "schema"
        ] == {"$ref": f"#/components/schemas/{response_schema}"}

    operation_ids = [
        operation["operationId"]
        for path_operations in paths.values()
        for operation in path_operations.values()
    ]
    assert len(operation_ids) == len(set(operation_ids))


@pytest.mark.parametrize(
    ("path", "module_name", "detail", "request_schema"),
    [
        (
            routes.EXPLAINER_LOCAL_LIME,
            _LIME_MODULE_NAME,
            "Local LIME explanation is not yet implemented",
            "LocalExplanationPlaceholderRequest",
        ),
        (
            routes.EXPLAINER_LOCAL_SHAP,
            _SHAP_MODULE_NAME,
            "Local SHAP explanation is not yet implemented",
            "LocalExplanationPlaceholderRequest",
        ),
    ],
)
def test_enabled_router_selects_implementation_or_placeholder(
    path: str,
    module_name: str,
    detail: str,
    request_schema: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the implementation when available and a placeholder otherwise."""
    monkeypatch.setitem(feature_flags.ENDPOINTS, "explainer", value=True)
    monkeypatch.setitem(feature_flags.ENDPOINTS, "explainer_local", value=True)
    app = FastAPI()
    app.include_router(build_router())

    operation = app.openapi()["paths"][path]["post"]
    if _module_available(module_name):
        expected_request, expected_response = _implementation_contract(module_name)
        assert operation["requestBody"]["content"]["application/json"]["schema"] == {
            "$ref": f"#/components/schemas/{expected_request}"
        }
        assert operation["responses"]["200"]["content"]["application/json"][
            "schema"
        ] == {"$ref": f"#/components/schemas/{expected_response}"}
        return

    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": f"#/components/schemas/{request_schema}"
    }
    response = TestClient(app).post(
        path,
        json={"predictionId": "prediction-1", "config": {}},
    )
    assert response.status_code == HTTPStatus.NOT_IMPLEMENTED
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(
    ("missing_module", "missing_path", "detail"),
    [
        (
            _LIME_MODULE_NAME,
            routes.EXPLAINER_LOCAL_LIME,
            "Local LIME explanation is not yet implemented",
        ),
        (
            _SHAP_MODULE_NAME,
            routes.EXPLAINER_LOCAL_SHAP,
            "Local SHAP explanation is not yet implemented",
        ),
    ],
)
def test_enabled_router_falls_back_for_each_missing_module(
    monkeypatch: pytest.MonkeyPatch,
    missing_module: str,
    missing_path: str,
    detail: str,
) -> None:
    """Keep only the missing algorithm on its dedicated placeholder."""
    monkeypatch.setitem(feature_flags.ENDPOINTS, "explainer", value=True)
    monkeypatch.setitem(feature_flags.ENDPOINTS, "explainer_local", value=True)
    other_module = next(
        module_name
        for module_name in _ALGORITHM_MODULES
        if module_name != missing_module
    )
    imported: list[str] = []
    real_import = importlib.import_module

    def import_optional(name: str, package: str | None = None) -> object:
        """Simulate one absent module while retaining the sibling route."""
        if name in _ALGORITHM_MODULES:
            imported.append(name)
        if name == missing_module:
            raise ModuleNotFoundError(name=name)
        if name == other_module:
            return _fake_algorithm_module(other_module)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_optional)
    app = FastAPI()
    app.include_router(build_router())

    assert imported == list(_ALGORITHM_MODULES)
    operation = app.openapi()["paths"][missing_path]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/LocalExplanationPlaceholderRequest"
    }
    response = TestClient(app).post(
        missing_path,
        json={"predictionId": "prediction-1", "config": {}},
    )
    assert response.status_code == HTTPStatus.NOT_IMPLEMENTED
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(
    ("broken_module", "error_name"),
    [
        (_LIME_MODULE_NAME, "missing-lime-dependency"),
        (_SHAP_MODULE_NAME, "missing-shap-dependency"),
    ],
)
def test_enabled_router_propagates_transitive_module_not_found(
    monkeypatch: pytest.MonkeyPatch,
    broken_module: str,
    error_name: str,
) -> None:
    """Do not hide a dependency import failure from an existing module."""
    monkeypatch.setitem(feature_flags.ENDPOINTS, "explainer", value=True)
    monkeypatch.setitem(feature_flags.ENDPOINTS, "explainer_local", value=True)
    other_module = next(
        module_name
        for module_name in _ALGORITHM_MODULES
        if module_name != broken_module
    )
    real_import = importlib.import_module

    def import_optional(name: str, package: str | None = None) -> object:
        """Simulate one broken dependency and one valid sibling module."""
        if name == broken_module:
            raise ModuleNotFoundError(name=error_name)
        if name == other_module:
            return _fake_algorithm_module(other_module)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_optional)
    with pytest.raises(ModuleNotFoundError) as raised:
        build_router()

    assert raised.value.name == error_name


@pytest.mark.parametrize("disabled_flag", ["explainer", "explainer_local"])
def test_disabled_local_router_does_not_import_optional_modules(
    monkeypatch: pytest.MonkeyPatch,
    disabled_flag: str,
) -> None:
    """Use both algorithm placeholders without importing optional modules."""
    flags = {"explainer": True, "explainer_local": True}
    flags[disabled_flag] = False
    for name, enabled in flags.items():
        monkeypatch.setitem(feature_flags.ENDPOINTS, name, enabled)
    optional_modules = set(_ALGORITHM_MODULES)
    real_import = importlib.import_module
    imported: list[str] = []

    def reject_optional(name: str, package: str | None = None) -> object:
        """Fail if disabled router construction imports an algorithm."""
        if name in optional_modules:
            imported.append(name)
            raise AssertionError
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", reject_optional)
    app = FastAPI()
    app.include_router(build_router())

    client = TestClient(app)
    for path, detail in (
        (
            routes.EXPLAINER_LOCAL_LIME,
            "Local LIME explanation is not yet implemented",
        ),
        (
            routes.EXPLAINER_LOCAL_SHAP,
            "Local SHAP explanation is not yet implemented",
        ),
    ):
        response = client.post(
            path,
            json={"predictionId": "prediction-1", "config": {}},
        )
        assert response.status_code == HTTPStatus.NOT_IMPLEMENTED
        assert response.json() == {"detail": detail}
        operation = app.openapi()["paths"][path]["post"]
        assert operation["requestBody"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/LocalExplanationPlaceholderRequest"
        }
    assert imported == []


@pytest.mark.parametrize(
    ("path", "prediction", "detail"),
    [
        (
            routes.EXPLAINER_LOCAL_CF,
            {"predictionId": "prediction-1"},
            "Local Counterfactual explanation is not yet implemented",
        ),
        (
            routes.EXPLAINER_LOCAL_TSSALIENCY,
            {"predictionIds": ["prediction-1"]},
            "Local TSSaliency explanation is not yet implemented",
        ),
    ],
)
@pytest.mark.parametrize(
    "flags",
    [
        {"explainer": True, "explainer_local": True},
        {"explainer": True, "explainer_local": False},
        {"explainer": False, "explainer_local": True},
    ],
)
def test_unfinished_placeholders_preserve_responses(
    monkeypatch: pytest.MonkeyPatch,
    flags: dict[str, bool],
    path: str,
    prediction: dict[str, str | list[str]],
    detail: str,
) -> None:
    """Preserve unfinished route responses under every feature-gate state."""
    for name, enabled in flags.items():
        monkeypatch.setitem(feature_flags.ENDPOINTS, name, enabled)
    app = FastAPI()
    app.include_router(build_router())

    response = TestClient(app).post(
        path,
        json={
            **prediction,
            "config": {
                "model": {"target": "target", "name": "model"},
                "explainer": {},
            },
        },
    )

    assert response.status_code == HTTPStatus.NOT_IMPLEMENTED
    assert response.json() == {"detail": detail}


def test_disabled_startup_does_not_import_optional_explainers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build disabled placeholders without importing optional explainers at startup."""
    monkeypatch.setenv("TRUSTYAI_ENABLE_EXPLAINER", "false")
    monkeypatch.setenv("TRUSTYAI_ENABLE_EXPLAINER_LOCAL", "false")
    monkeypatch.setenv("TRUSTYAI_ENABLE_EXPLAINER_GLOBAL", "false")
    script = r"""
import importlib
import sys
from fastapi import FastAPI
from trustyai_service.endpoints import routes

blocked_modules = (
    "trustyai_service.endpoints.explainers.local_lime",
    "trustyai_service.endpoints.explainers.local_shap",
    "lime",
    "shap",
)
blocked_attempts = []

class BlockedOptionalFinder:
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == module or fullname.startswith(module + ".")
            for module in blocked_modules
        ):
            blocked_attempts.append(fullname)
            raise AssertionError(f"blocked optional import attempted: {fullname}")
        return None

finder = BlockedOptionalFinder()
sys.meta_path.insert(0, finder)
try:
    importlib.import_module("trustyai_service.main")
    local_explainer = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_explainer"
    )
    app = FastAPI()
    app.include_router(local_explainer.build_router())
    paths = app.openapi()["paths"]
    for path in (routes.EXPLAINER_LOCAL_LIME, routes.EXPLAINER_LOCAL_SHAP):
        operation = paths[path]["post"]
        assert operation["requestBody"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/LocalExplanationPlaceholderRequest"
        }
        assert (
            operation["responses"]["200"]["content"]["application/json"]["schema"][
                "type"
            ]
            == "object"
        )
    assert blocked_attempts == []
finally:
    sys.meta_path.remove(finder)
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
