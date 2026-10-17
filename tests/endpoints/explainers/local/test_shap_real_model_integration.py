"""FastAPI-to-loopback-KServe integration coverage for KernelSHAP."""

import importlib
import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from trustyai_service import main
from trustyai_service.service.config import feature_flags

from .integration_helpers import FakeKServe, FakeKServeHandler, LocalStorage


def _request(
    base_url: str,
    *,
    source: str | None = None,
    task: str = "REGRESSION",
    class_index: int | None = None,
    link: str | None = None,
) -> dict:
    model = {"model_name": "m", "task": task}
    if source is None:
        model.update(
            {
                "base_url": base_url,
                "model_version": "v1",
                "input_name": "input",
                "output_name": "output",
            }
        )
    else:
        model["prediction_source"] = source
    explainer: dict[str, object] = {
        "n_samples": 8,
        "n_training_rows": 2,
        "confidence": 1.0,
        "timeout": 30,
    }
    if class_index is not None:
        explainer["class_index"] = class_index
    if link is not None:
        explainer["link"] = link
    return {
        "predictionId": "target",
        "config": {
            "model": model,
            "explainer": explainer,
        },
    }


def _enabled_client() -> TestClient:
    with patch.dict(
        feature_flags.ENDPOINTS,
        {"explainer": True, "explainer_local": True, "explainer_global": False},
    ):
        module = importlib.reload(main)
    return TestClient(module.app)


def test_shap_model_and_explicit_surrogate_use_distinct_paths() -> None:
    """Verify KernelSHAP uses the provider only for the real-model source."""
    storage = LocalStorage()
    with (
        FakeKServe() as fake,
        patch.dict(os.environ, {"TRUSTYAI_EXPLAINER_ALLOWED_HOSTS": "127.0.0.1"}),
        patch(
            "trustyai_service.service.data.local_explanation.get_global_storage_interface",
            return_value=storage,
        ),
    ):
        client = _enabled_client()
        try:
            response = client.post(
                "/explainers/local/shap", json=_request(fake.base_url)
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["prediction_source"] == "MODEL"
            assert payload["prediction_output"] is not None
            assert payload["linked_prediction_output"] == payload["prediction_output"]
            assert FakeKServeHandler.metadata_calls >= 1
            assert FakeKServeHandler.infer_calls
            assert FakeKServeHandler.metadata_paths[-1] == "/v2/models/m/versions/v1"
            assert FakeKServeHandler.infer_paths
            assert all(
                call["outputs"] == [{"name": "output"}]
                for call in FakeKServeHandler.infer_calls
            )
            model_call_count = len(FakeKServeHandler.infer_calls)

            surrogate = client.post(
                "/explainers/local/shap",
                json=_request(fake.base_url, source="SURROGATE"),
            )
            assert surrogate.status_code == 200, surrogate.text
            assert surrogate.json()["prediction_source"] == "SURROGATE"
            assert len(FakeKServeHandler.infer_calls) == model_call_count

            unavailable = client.post(
                "/explainers/local/shap", json=_request("http://127.0.0.1:1")
            )
            assert unavailable.status_code == 503, unavailable.text
        finally:
            client.close()
            feature_flags.ENDPOINTS.update(
                {
                    "explainer": False,
                    "explainer_local": False,
                    "explainer_global": False,
                }
            )
            importlib.reload(main)


def test_shap_classification_model_supports_logit_link() -> None:
    """Select a real class score and expose its linked SHAP prediction."""
    storage = LocalStorage()
    with (
        FakeKServe(classification=True) as fake,
        patch.dict(os.environ, {"TRUSTYAI_EXPLAINER_ALLOWED_HOSTS": "127.0.0.1"}),
        patch(
            "trustyai_service.service.data.local_explanation.get_global_storage_interface",
            return_value=storage,
        ),
    ):
        client = _enabled_client()
        try:
            response = client.post(
                "/explainers/local/shap",
                json=_request(
                    fake.base_url,
                    task="CLASSIFICATION",
                    class_index=1,
                    link="LOGIT",
                ),
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["prediction_source"] == "MODEL"
            assert payload["class_index"] == 1
            assert payload["prediction_output"] == 0.75
            assert payload["linked_prediction_output"] != payload["prediction_output"]
            assert FakeKServeHandler.infer_calls
        finally:
            client.close()
            feature_flags.ENDPOINTS.update(
                {
                    "explainer": False,
                    "explainer_local": False,
                    "explainer_global": False,
                }
            )
            importlib.reload(main)
