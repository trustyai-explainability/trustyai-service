"""LIME endpoint response and source-selection contract tests."""

from collections.abc import Callable

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints.explainers import local_lime
from trustyai_service.service.data.local_explanation import LocalExplanationData
from trustyai_service.service.explainers.local.model_provider import (
    LocalDataNotFoundError,
)
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
    """Preserve the raw classification vector alongside LIME's selected value."""
    app = FastAPI()
    app.include_router(local_lime.router)
    data = LocalExplanationData(
        "m", "target", np.array([0.8]), ["f0"], np.array([[0.2], [0.4]]), None
    )
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def load(*_args: object, **_kwargs: object) -> LocalExplanationData:
        return data

    monkeypatch.setattr(local_lime, "load_local_explanation_data", load)

    def execution_factory(*_args: object, **_kwargs: object) -> _Execution:
        return _Execution()

    monkeypatch.setattr(local_lime, "create_prediction_execution", execution_factory)

    def explainer_factory(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(local_lime, "create_lime_explainer", explainer_factory)

    def explanation(
        *_args: object, **_kwargs: object
    ) -> tuple[list[tuple[str, float]], float, float, float]:
        return [("f0", 0.2)], 0.9, 0.8, 0.1

    monkeypatch.setattr(local_lime, "compute_lime_explanation", explanation)

    def confidence_intervals(*_args: object, **_kwargs: object) -> tuple[None, None]:
        return None, None

    monkeypatch.setattr(
        local_lime, "compute_lime_confidence_intervals", confidence_intervals
    )

    async def run(function: Callable[[], object], _duration: float) -> object:
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


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LocalDataNotFoundError("stored data is missing"), 404),
        (ValueError("backend"), 500),
    ],
)
def test_lime_data_loading_uses_shared_error_mapping(
    monkeypatch: pytest.MonkeyPatch, error: Exception, status: int
) -> None:
    """Map data-loading failures through the endpoint-neutral policy."""
    app = FastAPI()
    app.include_router(local_lime.router)
    monkeypatch.setattr(local_lime, "_LIME_AVAILABLE", True)

    async def load(*_args: object, **_kwargs: object) -> LocalExplanationData:
        raise error

    monkeypatch.setattr(local_lime, "load_local_explanation_data", load)
    response = TestClient(app).post(
        "/explainers/local/lime",
        json={
            "predictionId": "target",
            "config": {
                "model": {
                    "base_url": "http://model.example",
                    "model_name": "m",
                    "task": "REGRESSION",
                }
            },
        },
    )
    assert response.status_code == status, response.text
