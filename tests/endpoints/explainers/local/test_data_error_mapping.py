"""Integration coverage for local data failures at the HTTP mapping boundary."""

from __future__ import annotations

from http import HTTPStatus

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from trustyai_service.service.data.local_explanation import (
    load_local_explanation_data,
)
from trustyai_service.service.explainers.local.error_mapping import map_error


class _LoaderStorage:
    """Minimal storage fake for exercising the real loader boundary."""

    def __init__(self, *, missing: bool = False, invalid: bool = False) -> None:
        self.missing = missing
        self.invalid = invalid

    async def dataset_exists(self, _dataset_name: str) -> bool:
        return not self.missing

    async def dataset_rows(self, _dataset_name: str) -> int:
        return 1

    async def get_aliased_column_names(self, _dataset_name: str) -> list[str]:
        return ["feature"]

    async def read_data(
        self,
        dataset_name: str,
        start_row: int = 0,
        n_rows: int | None = None,
    ) -> np.ndarray:
        del start_row, n_rows
        if dataset_name.endswith("_metadata"):
            return np.array([["target", "2026-09-22T00:00:00", 0.0, []]], dtype=object)
        value = np.nan if self.invalid else 1.0
        return np.array([[value]], dtype=float)


def _app(storage: _LoaderStorage) -> FastAPI:
    """Build the smallest HTTP adapter using the production mapper."""
    app = FastAPI()

    @app.get("/local-data")
    async def local_data() -> dict[str, str]:
        try:
            await load_local_explanation_data(
                "model",
                "target",
                1,
                include_stored_output=False,
                storage_interface=storage,
            )
        except Exception as error:
            mapped = map_error(error)
            raise HTTPException(
                status_code=mapped.status_code,
                detail=mapped.as_http_detail(),
            ) from error
        return {"status": "ok"}

    return app


@pytest.mark.parametrize(
    ("storage", "status", "code"),
    [
        (_LoaderStorage(missing=True), HTTPStatus.NOT_FOUND, "data_missing"),
        (_LoaderStorage(invalid=True), HTTPStatus.BAD_REQUEST, "data_invalid"),
    ],
)
def test_loader_failures_keep_typed_http_categories(
    storage: _LoaderStorage,
    status: HTTPStatus,
    code: str,
) -> None:
    """Expose representative loader failures as stable public error categories."""
    response = TestClient(_app(storage)).get("/local-data")

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
