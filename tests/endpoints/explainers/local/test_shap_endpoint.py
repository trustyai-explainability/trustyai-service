"""KernelSHAP endpoint response and link-space contract tests."""

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.core.explainers.local.shap import ShapExplanationResult
from trustyai_service.endpoints.explainers import local_shap
from trustyai_service.service.data.local_explanation import LocalExplanationData
from trustyai_service.service.explainers.local.types import PredictionSource


class _Execution:
    source = PredictionSource.MODEL
    provider = None
    resolved_output_name = "score"

    def predict_fn(self, values: np.ndarray) -> np.ndarray:
        return values[:, :1] * 0.5 + 0.25

    def close(self) -> None:
        return None


def test_shap_response_exposes_raw_and_linked_outputs(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(local_shap.router)
    data = LocalExplanationData(
        "m", "target", np.array([0.8]), ["f0"], np.array([[0.2], [0.4]]), None
    )
    monkeypatch.setattr(local_shap, "_SHAP_AVAILABLE", True)

    async def load(*args, **kwargs):
        return data

    monkeypatch.setattr(local_shap, "load_local_explanation_data", load)
    monkeypatch.setattr(
        local_shap, "create_prediction_execution", lambda *args, **kwargs: _Execution()
    )
    monkeypatch.setattr(
        local_shap,
        "compute_shap_result",
        lambda *args, **kwargs: ShapExplanationResult(np.array([0.1]), 0.3, 0.7),
    )
    monkeypatch.setattr(
        local_shap, "compute_confidence_intervals", lambda *args, **kwargs: (None, None)
    )

    async def run(function, _duration):
        return function()

    monkeypatch.setattr(local_shap, "run_local_worker", run)

    response = TestClient(app).post(
        "/explainers/local/shap",
        json={
            "predictionId": "target",
            "config": {
                "model": {
                    "base_url": "http://model.example",
                    "model_name": "m",
                    "task": "REGRESSION",
                },
                "explainer": {"confidence": 1.0},
            },
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prediction_output"] == 0.65
    assert payload["shap_base_value"] == 0.3
    assert payload["linked_prediction_output"] == 0.7
