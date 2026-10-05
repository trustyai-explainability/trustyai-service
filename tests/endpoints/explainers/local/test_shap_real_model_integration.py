"""Real FastAPI-to-KServe integration tests for local KernelSHAP."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints.explainers import local_shap

from .integration_helpers import FakeKServe, LocalStorage, enabled_test_client

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec("shap"),
    reason="optional SHAP dependency is unavailable",
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Exercise the SHAP router with real storage loading and model execution."""
    app = FastAPI()
    app.include_router(local_shap.router)
    with TestClient(app) as test_client:
        yield test_client


def _request(
    base_url: str,
    *,
    task: str = "REGRESSION",
    class_index: int | None = None,
    link: str = "IDENTITY",
    single_probability: bool = False,
) -> dict[str, object]:
    """Build one canonical model-backed request."""
    model: dict[str, object] = {
        "model_name": "m",
        "model_version": "v1",
        "task": task,
        "base_url": base_url,
        "input_name": "input",
        "output_name": "output",
    }
    explainer: dict[str, object] = {
        "n_samples": 12,
        "n_training_rows": 2,
        "confidence": 1.0,
        "timeout": 30,
        "link": link,
        "regularizer": "BIC",
        "single_probability": single_probability,
    }
    if class_index is not None:
        explainer["class_index"] = class_index
    return {
        "predictionId": "target",
        "config": {
            "model": model,
            "explainer": explainer,
        },
    }


def _install_storage(monkeypatch: pytest.MonkeyPatch, storage: LocalStorage) -> None:
    """Use the real bounded loader against the in-memory storage double."""
    data_module = importlib.import_module(
        "trustyai_service.service.data.local_explanation"
    )
    monkeypatch.setattr(data_module, "get_global_storage_interface", lambda: storage)


def test_model_shap_calls_loopback_provider_and_preserves_identity_additivity(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """Exercise metadata, coalition inference, model identity, and additivity."""
    storage = LocalStorage()
    _install_storage(monkeypatch, storage)

    with FakeKServe(model_name="m", model_version="v1") as fake:
        monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")
        response = client.post("/explainers/local/shap", json=_request(fake.base_url))

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model"] == "m"
    assert payload["task"] == "REGRESSION"
    assert payload["output_name"] == "output"
    assert payload["prediction_output"] == pytest.approx(1.0)
    assert payload["linked_prediction_output"] == pytest.approx(1.0)
    assert payload["shap_base_value"] + sum(
        item["importance"] for item in payload["attributions"]
    ) == pytest.approx(payload["linked_prediction_output"], abs=1e-5)
    assert fake.metadata_calls == 1
    assert fake.metadata_paths == ["/v2/models/m/versions/v1"]
    assert fake.infer_calls
    assert all(path == "/v2/models/m/versions/v1/infer" for path in fake.infer_paths)
    assert all(call["inputs"][0]["name"] == "input" for call in fake.infer_calls)
    assert all(call["outputs"] == [{"name": "output"}] for call in fake.infer_calls)
    assert any(call["inputs"][0]["shape"][0] > 1 for call in fake.infer_calls)
    generated_rows = np.concatenate(
        [
            np.asarray(call["inputs"][0]["data"]).reshape(-1, 2)
            for call in fake.infer_calls
        ]
    )
    assert any(
        (row[0] == 2.0 and row[1] != 2.0) or (row[1] == 2.0 and row[0] != 2.0)
        for row in generated_rows
    )


def test_enabled_application_serves_model_shap_explanation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep full-application coverage of the enabled SHAP route and provider."""
    storage = LocalStorage()
    _install_storage(monkeypatch, storage)
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")

    with (
        FakeKServe(model_name="m", model_version="v1") as fake,
        enabled_test_client() as app_client,
    ):
        response = app_client.post(
            "/explainers/local/shap", json=_request(fake.base_url)
        )

    assert response.status_code == 200, response.text
    assert response.json()["prediction_output"] == pytest.approx(1.0)
    assert fake.metadata_calls == 1
    assert any(call["inputs"][0]["shape"][0] > 1 for call in fake.infer_calls)


@pytest.mark.parametrize("link", ["IDENTITY", "LOGIT"])
@pytest.mark.parametrize("class_index", [0, 1])
def test_classification_shap_supports_identity_and_logit_link_spaces(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    link: str,
    class_index: int,
) -> None:
    """Return the selected class probability with link-space additivity."""
    storage = LocalStorage()
    _install_storage(monkeypatch, storage)
    with FakeKServe(
        classification=True,
        model_name="m",
        model_version="v1",
    ) as fake:
        monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")
        response = client.post(
            "/explainers/local/shap",
            json=_request(
                fake.base_url,
                task="CLASSIFICATION",
                class_index=class_index,
                link=link,
            ),
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["class_index"] == class_index
    expected_output = 0.75 if class_index == 1 else 0.25
    assert payload["prediction_output"] == pytest.approx(expected_output)
    expected_linked = (
        expected_output
        if link == "IDENTITY"
        else np.log(3.0) * (1 if class_index == 1 else -1)
    )
    assert payload["linked_prediction_output"] == pytest.approx(expected_linked)
    assert payload["shap_base_value"] + sum(
        item["importance"] for item in payload["attributions"]
    ) == pytest.approx(expected_linked, abs=1e-5)
    assert fake.metadata_calls == 1
    assert fake.infer_calls


def test_model_outage_returns_error_without_fallback_explanation(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """Return the mapped provider outage without an explanation."""
    storage = LocalStorage()
    _install_storage(monkeypatch, storage)
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")

    with FakeKServe(metadata_status=503) as fake:
        response = client.post("/explainers/local/shap", json=_request(fake.base_url))

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "unavailable"
    assert "attributions" not in response.json()
    assert "prediction_output" not in response.json()
    assert fake.metadata_calls == 1
    assert fake.infer_calls == []
