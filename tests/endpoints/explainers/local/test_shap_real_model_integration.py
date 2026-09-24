"""Real FastAPI-to-KServe integration tests for local KernelSHAP."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from trustyai_service.service.explainers.local import execution as execution_module

from .integration_helpers import FakeKServe, LocalStorage, enabled_test_client

pytestmark = pytest.mark.skipif(
    not importlib.util.find_spec("shap"),
    reason="optional SHAP dependency is unavailable",
)


def _request(  # noqa: PLR0913
    base_url: str | None,
    *,
    source: str = "MODEL",
    task: str = "REGRESSION",
    class_index: int | None = None,
    link: str = "IDENTITY",
    single_probability: bool = False,
) -> dict[str, object]:
    """Build one canonical real-model or explicit-surrogate request."""
    model: dict[str, object] = {
        "model_name": "m",
        "model_version": "v1",
        "prediction_source": source,
        "task": task,
    }
    if base_url is not None:
        model.update(
            {
                "base_url": base_url,
                "input_name": "input",
                "output_name": "output",
            }
        )
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
) -> None:
    """Exercise metadata, coalition inference, model identity, and additivity."""
    storage = LocalStorage()
    _install_storage(monkeypatch, storage)

    with FakeKServe(model_name="m", model_version="v1") as fake:
        monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")
        with enabled_test_client() as client:
            monkeypatch.setattr(
                execution_module,
                "_load_surrogate_builder",
                lambda: pytest.fail("MODEL attempted surrogate construction"),
            )
            response = client.post(
                "/explainers/local/shap", json=_request(fake.base_url)
            )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prediction_source"] == "MODEL"
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


def test_classification_shap_supports_identity_and_logit_link_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return selected class-one probability and link-space additivity."""
    for link in ("IDENTITY", "LOGIT"):
        storage = LocalStorage()
        _install_storage(monkeypatch, storage)
        with FakeKServe(
            classification=True,
            model_name="m",
            model_version="v1",
        ) as fake:
            monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")
            with enabled_test_client() as client:
                response = client.post(
                    "/explainers/local/shap",
                    json=_request(
                        fake.base_url,
                        task="CLASSIFICATION",
                        class_index=1,
                        link=link,
                    ),
                )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["class_index"] == 1
        assert payload["prediction_output"] == pytest.approx(0.75)
        expected_linked = 0.75 if link == "IDENTITY" else np.log(3.0)
        assert payload["linked_prediction_output"] == pytest.approx(expected_linked)
        assert payload["shap_base_value"] + sum(
            item["importance"] for item in payload["attributions"]
        ) == pytest.approx(expected_linked, abs=1e-5)
        assert fake.metadata_calls == 1
        assert fake.infer_calls


def test_explicit_surrogate_makes_zero_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explicit SURROGATE provider-free even with a live fake server."""
    storage = LocalStorage()
    _install_storage(monkeypatch, storage)
    monkeypatch.setattr(
        execution_module,
        "get_transport_config",
        lambda _source: pytest.fail("SURROGATE resolved transport settings"),
    )

    with (
        FakeKServe(model_name="m", model_version="v1") as fake,
        enabled_test_client() as client,
    ):
        response = client.post(
            "/explainers/local/shap",
            json=_request(None, source="SURROGATE"),
        )

    assert response.status_code == 200, response.text
    assert response.json()["prediction_source"] == "SURROGATE"
    assert fake.metadata_calls == 0
    assert fake.infer_calls == []


def test_model_outage_does_not_fall_back_to_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the real provider outage and never train a surrogate."""
    storage = LocalStorage()
    _install_storage(monkeypatch, storage)
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "127.0.0.1")

    with FakeKServe(metadata_status=503) as fake, enabled_test_client() as client:
        monkeypatch.setattr(
            execution_module,
            "_load_surrogate_builder",
            lambda: pytest.fail("MODEL outage attempted surrogate fallback"),
        )
        response = client.post("/explainers/local/shap", json=_request(fake.base_url))

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "unavailable"
    assert fake.metadata_calls == 1
    assert fake.infer_calls == []
