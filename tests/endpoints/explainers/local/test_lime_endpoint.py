"""LIME endpoint response and source-selection contract tests."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints.explainers import local_lime
from trustyai_service.service.data.local_explanation import LocalExplanationData
from trustyai_service.service.explainers.local.types import PredictionSource


class _Execution:
    source = PredictionSource.MODEL
    provider = None
    resolved_output_name = "score"

    def predict_fn(self, values: np.ndarray) -> np.ndarray:
        return np.column_stack((1 - values[:, 0], values[:, 0]))

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_lime_classification_response_keeps_raw_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = FastAPI()
    app.include_router(local_lime.router)
    data = LocalExplanationData(
        "m", "target", np.array([0.8]), ["f0"], np.array([[0.2], [0.4]]), None
    )
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def load(*args, **kwargs):
        return data

    monkeypatch.setattr(local_lime, "load_local_explanation_data", load)
    monkeypatch.setattr(
        local_lime, "create_prediction_execution", lambda *args, **kwargs: _Execution()
    )
    monkeypatch.setattr(
        local_lime, "create_lime_explainer", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        local_lime,
        "compute_lime_explanation",
        lambda *args, **kwargs: ([("f0", 0.2)], 0.9, 0.8, 0.1),
    )
    monkeypatch.setattr(
        local_lime,
        "compute_lime_confidence_intervals",
        lambda *args, **kwargs: (None, None),
    )

    async def run(function, _duration):
        return function()

    monkeypatch.setattr(local_lime, "run_local_worker", run)
    client = TestClient(app)
    response = client.post(
        "/explainers/local/lime",
        json={
            "predictionId": "target",
            "config": {
                "model": {
                    "base_url": "http://model.example",
                    "model_name": "m",
                    "task": "CLASSIFICATION",
                },
                "explainer": {"class_index": 1, "confidence": 1.0},
            },
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["class_index"] == 1
    assert payload["prediction_output"] == pytest.approx([0.2, 0.8])
    assert payload["local_prediction"] == pytest.approx(0.8)
