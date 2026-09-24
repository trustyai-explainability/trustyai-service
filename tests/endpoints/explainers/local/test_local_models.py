"""Tests for the canonical local-explainer request contract."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from trustyai_service.endpoints import routes
from trustyai_service.service.explainers.local.model_provider import (
    ProviderInvalidRequestError,
)
from trustyai_service.service.explainers.local.types import PredictionSource, TaskType


def test_canonical_model_config_uses_provider_types_and_all_fields() -> None:
    """The request model carries the canonical model and tensor selectors."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )

    config = models.LocalExplanationModelConfig(
        base_url="https://inference.example/prefix",
        model_name="credit-model",
        model_version="v1",
        prediction_source=PredictionSource.SURROGATE,
        task=TaskType.CLASSIFICATION,
        input_name="features",
        output_name="scores",
    )

    assert config.model_name == "credit-model"
    assert config.model_version == "v1"
    assert config.prediction_source is PredictionSource.SURROGATE
    assert config.task is TaskType.CLASSIFICATION
    assert config.input_name == "features"
    assert config.output_name == "scores"
    assert str(config.base_url) == "https://inference.example/prefix"


def test_model_source_is_the_default_and_requires_base_url() -> None:
    """MODEL is the safe default and validates its serving URL after parsing."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )

    config = models.LocalExplanationModelConfig(
        base_url="https://inference.example",
        model_name="credit-model",
        task=TaskType.REGRESSION,
    )
    assert config.prediction_source is PredictionSource.MODEL

    config_without_url = models.LocalExplanationModelConfig(
        model_name="credit-model",
        prediction_source=PredictionSource.MODEL,
        task=TaskType.REGRESSION,
    )
    with pytest.raises(
        ProviderInvalidRequestError,
        match="base_url is required when prediction_source=MODEL",
    ):
        models.validate_local_explanation_model_config(config_without_url)


def test_surrogate_source_may_omit_base_url() -> None:
    """An explicit SURROGATE request remains provider-free."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )

    config = models.LocalExplanationModelConfig(
        model_name="credit-model",
        prediction_source=PredictionSource.SURROGATE,
        task=TaskType.CLASSIFICATION,
    )

    assert config.base_url is None
    validator = getattr(models, "validate_local_explanation_model_config", None)
    assert callable(validator)
    assert validator(config) is config


def test_task_is_required_for_both_prediction_sources() -> None:
    """The semantic task cannot be inferred from a model response."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )

    for source, base_url in (
        (PredictionSource.MODEL, "https://inference.example"),
        (PredictionSource.SURROGATE, None),
    ):
        payload = {
            "model_name": "credit-model",
            "prediction_source": source,
        }
        if base_url is not None:
            payload["base_url"] = base_url
        with pytest.raises(ValidationError) as exc_info:
            models.LocalExplanationModelConfig(**payload)
        assert any(error["loc"] == ("task",) for error in exc_info.value.errors())


@pytest.mark.parametrize(
    "field",
    ["model_name", "model_version", "input_name", "output_name"],
)
@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_model_identity_and_tensor_selectors_cannot_be_blank(
    field: str, value: str
) -> None:
    """Reject blank model identity and optional tensor selectors."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )
    payload: dict[str, object] = {
        "base_url": "https://inference.example",
        "model_name": "credit-model",
        "task": TaskType.REGRESSION,
        field: value,
    }

    with pytest.raises(ValidationError):
        models.LocalExplanationModelConfig(**payload)


@pytest.mark.parametrize(
    "field",
    ["model_name", "model_version", "input_name", "output_name"],
)
@pytest.mark.parametrize(
    "value",
    [".", "..", "model/name", "model\\name", "model%name", "model\x00name"],
)
def test_model_identity_and_tensor_selectors_are_path_safe(
    field: str, value: str
) -> None:
    """Reject selectors that could escape a KServe path segment."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )
    payload: dict[str, object] = {
        "base_url": "https://inference.example",
        "model_name": "credit-model",
        "task": TaskType.REGRESSION,
        field: value,
    }

    with pytest.raises(ValidationError):
        models.LocalExplanationModelConfig(**payload)


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "https://",
        "inference.example:8080",
        "ftp://inference.example",
    ],
)
def test_malformed_base_url_remains_pydantic_validation_error(
    base_url: str,
) -> None:
    """Malformed or unsupported URL syntax remains Pydantic validation."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )

    with pytest.raises(ValidationError):
        models.LocalExplanationModelConfig(
            base_url=base_url,
            model_name="credit-model",
            task=TaskType.REGRESSION,
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:pass@inference.example",  # pragma: allowlist secret
        "https://inference.example/path?token=secret",  # pragma: allowlist secret
        "https://inference.example/path#fragment",
        "https://inference.example/%2fprivate",
    ],
)
def test_base_url_rejects_unsafe_components_after_url_parsing(
    base_url: str,
) -> None:
    """Credential and path-policy checks remain local Pydantic validation."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )

    with pytest.raises(ValidationError):
        models.LocalExplanationModelConfig(
            base_url=base_url,
            model_name="credit-model",
            task=TaskType.REGRESSION,
        )


def test_canonical_model_rejects_java_shaped_fields_and_task_sentinel() -> None:
    """Java request aliases and target task sentinels are not public API."""
    models = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_models"
    )

    with pytest.raises(ValidationError):
        models.LocalExplanationModelConfig(
            base_url="https://inference.example",
            model_name="credit-model",
            task="target",
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        models.LocalExplanationModelConfig(
            base_url="https://inference.example",
            model_name="credit-model",
            task=TaskType.REGRESSION,
            target="regressor",
            name="credit-model",
            version="v1",
        )


def test_placeholder_contract_is_not_replaced_by_canonical_contract() -> None:
    """Unfinished counterfactual and time-series routes keep their schemas."""
    local_explainer = importlib.import_module(
        "trustyai_service.endpoints.explainers.local_explainer"
    )

    placeholder = local_explainer.ModelConfig(target="regressor", name="legacy")
    assert placeholder.version is None
    assert local_explainer.CounterfactualExplanationRequest
    assert local_explainer.TSSaliencyExplanationRequest
    assert not hasattr(local_explainer, "LocalExplanationModelConfig")
    assert not hasattr(local_explainer, "LimeExplainerConfig")
    assert not hasattr(local_explainer, "SHAPExplainerConfig")


def test_local_explainer_routes_use_canonical_unversioned_paths() -> None:
    """LIME and SHAP use the public route names without a version prefix."""
    assert routes.EXPLAINER_LOCAL_LIME == "/explainers/local/lime"
    assert routes.EXPLAINER_LOCAL_SHAP == "/explainers/local/shap"
    assert not routes.EXPLAINER_LOCAL_LIME.startswith("/api/")
    assert not routes.EXPLAINER_LOCAL_SHAP.startswith("/api/")


def test_main_import_is_safe_when_explainers_are_disabled() -> None:
    """Disabled feature flags do not import optional algorithm/provider packages."""
    repository = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment.update(
        {
            "TRUSTYAI_ENABLE_EXPLAINER": "false",
            "TRUSTYAI_ENABLE_EXPLAINER_LOCAL": "false",
            "TRUSTYAI_ENABLE_EXPLAINER_GLOBAL": "false",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import trustyai_service.main

assert "trustyai_service.endpoints.explainers.local_lime" not in sys.modules
assert "trustyai_service.endpoints.explainers.local_shap" not in sys.modules
assert "trustyai_service.service.explainers.local.kserve_v2_http" not in sys.modules
assert "lime" not in sys.modules
assert "shap" not in sys.modules
assert "httpx2" not in sys.modules
""",
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_main_does_not_reflect_credentials_from_invalid_base_url() -> None:
    """Local model URL validation never echoes credentials in a 422 response."""
    repository = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    environment.update(
        {
            "TRUSTYAI_ENABLE_EXPLAINER": "true",
            "TRUSTYAI_ENABLE_EXPLAINER_LOCAL": "true",
            "TRUSTYAI_ENABLE_EXPLAINER_GLOBAL": "false",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from fastapi.testclient import TestClient
from fastapi import APIRouter
from pydantic import BaseModel
from trustyai_service.main import app
from trustyai_service.endpoints.explainers.local_models import LocalExplanationModelConfig

class RequestModel(BaseModel):
    model: LocalExplanationModelConfig

class UnrelatedRequestModel(BaseModel):
    count: int

router = APIRouter()

@router.post("/explainers/local/test-local-validation")
async def test_local_validation(request: RequestModel) -> dict[str, str]:
    return {"model": request.model.model_name}

@router.post("/test-unrelated-validation")
async def test_unrelated_validation(request: UnrelatedRequestModel) -> dict[str, int]:
    return {"count": request.count}

app.include_router(router)

response = TestClient(app).post(
    "/explainers/local/test-local-validation",
    json={
        "model": {
            "base_url": "https://user:secret@example.com",  # pragma: allowlist secret
            "model_name": "credit-model",
        },
    },
)
assert response.status_code == 422
assert "secret" not in response.text
details = response.json()["detail"]
assert any(error["loc"][-1] == "task" for error in details)
assert all("input" not in error and "ctx" not in error for error in details)

unrelated_response = TestClient(app).post(
    "/test-unrelated-validation",
    json={"count": "not-an-integer"},
)
assert unrelated_response.status_code == 422
assert unrelated_response.json()["detail"][0]["input"] == "not-an-integer"
""",
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
