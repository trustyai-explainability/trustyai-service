"""Regression tests for the unfinished local LIME and SHAP routes."""

from __future__ import annotations

from http import HTTPStatus

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trustyai_service.endpoints import routes
from trustyai_service.endpoints.explainers.local_explainer import build_router


@pytest.mark.parametrize(
    ("path", "detail"),
    [
        (
            routes.EXPLAINER_LOCAL_LIME,
            "Local LIME explanation is not yet implemented",
        ),
        (
            routes.EXPLAINER_LOCAL_SHAP,
            "Local SHAP explanation is not yet implemented",
        ),
    ],
)
def test_enabled_local_explainer_keeps_algorithm_placeholders(
    path: str,
    detail: str,
) -> None:
    """Keep the PR A placeholder status and response detail for both routes."""
    app = FastAPI()
    app.include_router(build_router())

    response = TestClient(app).post(
        path,
        json={
            "predictionId": "prediction-1",
            "config": {
                "model": {
                    "target": "target",
                    "name": "model",
                }
            },
        },
    )

    assert response.status_code == HTTPStatus.NOT_IMPLEMENTED
    assert response.json() == {"detail": detail}
