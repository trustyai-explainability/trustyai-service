"""End-to-end tests against a real SQLite storage backend.

Unlike the mocked metric endpoint tests (which patch ``get_data_source``), these
drive the full stack against a live in-memory SQLite database: data is written
through the real storage backend, and the metric/info endpoints read it back
through the real ``DataSource`` -> ``ModelData`` -> storage path. This exercises
the SQLite backend the same way a deployment would, with no DB server required.

Covers:
- Data upload API -> SQLite -> ``/info`` model registration + ``ModelData`` reads.
- Real fairness metric computation (SPD, DIR) reading from SQLite.
- Real batch-mean computation reading from SQLite.
- Name-mapping API round-trip.
"""

from __future__ import annotations

import asyncio
import os
import unittest
import uuid
from http import HTTPStatus
from importlib import reload

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

import trustyai_service.main
from trustyai_service.endpoints import routes
from trustyai_service.service.data.model_data import ModelData

# Deterministic fairness dataset:
#   privileged group "A": 40 rows, 32 favorable (rate 0.80)
#   unprivileged group "B": 40 rows, 20 favorable (rate 0.50)
# => SPD = 0.50 - 0.80 = -0.30 ; DIR = 0.50 / 0.80 = 0.625
_PRIV_N = 40
_UNPRIV_N = 40
_PRIV_FAVORABLE = 32
_UNPRIV_FAVORABLE = 20
_EXPECTED_SPD = -0.30
_EXPECTED_DIR = 0.625
# batch mean of the outcome column across all 80 rows: (32 + 20) / 80 = 0.65
_EXPECTED_OUTCOME_MEAN = 0.65
_TOL = 1e-6


class TestSQLiteEndToEnd(unittest.TestCase):
    """Full-stack tests backed by a real in-memory SQLite database."""

    def setUp(self) -> None:
        """Point the service at in-memory SQLite and build a fresh test client."""
        self.original_env = {
            "SERVICE_STORAGE_FORMAT": os.environ.get("SERVICE_STORAGE_FORMAT"),
            "STORAGE_DATABASE_PATH": os.environ.get("STORAGE_DATABASE_PATH"),
        }

        def restore_environment() -> None:
            for key, value in self.original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore_environment)

        os.environ["SERVICE_STORAGE_FORMAT"] = "SQLITE"
        os.environ["STORAGE_DATABASE_PATH"] = ":memory:"

        from trustyai_service.service.data import (  # noqa: PLC0415 -- reload for isolation
            storage,
        )

        self.storage_interface = storage.get_global_storage_interface(force_reload=True)
        self.addCleanup(storage.GlobalStorageInterface.reset)

        # Recreate the app so it binds to the fresh storage interface.
        reload(trustyai_service.main)
        from fastapi.testclient import TestClient  # noqa: PLC0415

        from trustyai_service.main import app  # noqa: PLC0415 -- reload for isolation

        self.client = TestClient(app)

    def _seed_fairness_data(self, model_id: str) -> None:
        """Write a deterministic fairness dataset straight to the SQLite backend."""
        # Equal-length category labels ("A"/"B") keep the stored column a uniform
        # string dtype so it round-trips as a proper categorical column.
        gender = np.array(["A"] * _PRIV_N + ["B"] * _UNPRIV_N).reshape(-1, 1)
        income = np.array(
            [1] * _PRIV_FAVORABLE
            + [0] * (_PRIV_N - _PRIV_FAVORABLE)
            + [1] * _UNPRIV_FAVORABLE
            + [0] * (_UNPRIV_N - _UNPRIV_FAVORABLE)
        ).reshape(-1, 1)
        total = _PRIV_N + _UNPRIV_N
        # Fixed-width ids keep the metadata columns uniform; tags mark data organic.
        metadata = np.array(
            [[f"id{i:04d}", ["TRAINING"]] for i in range(total)], dtype=object
        )

        async def _write() -> None:
            await self.storage_interface.write_data(
                f"{model_id}_inputs", gender, ["gender"]
            )
            await self.storage_interface.write_data(
                f"{model_id}_outputs", income, ["income"]
            )
            await self.storage_interface.write_data(
                f"{model_id}_metadata", metadata, ["id", "tags"]
            )

        asyncio.run(_write())

    def _fairness_payload(self, model_id: str) -> dict:
        return {
            "modelId": model_id,
            "protectedAttribute": "gender",
            "outcomeName": "income",
            "privilegedAttribute": "A",
            "unprivilegedAttribute": "B",
            "favorableOutcome": 1,
            "batchSize": 1000,
        }

    # === Data upload API -> SQLite -> reads ===================================
    def test_upload_api_persists_and_registers_model(self) -> None:
        """Uploading via the API writes to SQLite and registers the model."""
        model_name = f"e2e_upload_{uuid.uuid4().hex[:8]}"
        n_rows = 5
        payload = {
            "model_name": model_name,
            "data_tag": "TRAINING",
            "is_ground_truth": False,
            "request": {
                "inputs": [
                    {
                        "name": "input",
                        "shape": [n_rows, 2],
                        "datatype": "INT64",
                        "data": [[i, i + 1] for i in range(n_rows)],
                    }
                ]
            },
            "response": {
                "outputs": [
                    {
                        "name": "output",
                        "shape": [n_rows],
                        "datatype": "INT64",
                        "data": [i * 2 for i in range(n_rows)],
                    }
                ]
            },
        }

        response = self.client.post(routes.DATA_UPLOAD, json=payload)
        assert response.status_code == HTTPStatus.OK, response.text

        # Data landed in SQLite.
        datasets = asyncio.run(self.storage_interface.list_all_datasets())
        assert f"{model_name}_inputs" in datasets
        assert f"{model_name}_outputs" in datasets

        # ModelData reads it back through the real storage path.
        inputs, outputs, _ = asyncio.run(ModelData(model_name).data())
        assert inputs is not None
        assert outputs is not None
        assert len(inputs) == n_rows
        assert len(outputs) == n_rows

        # The model shows up in the service info endpoints.
        info = self.client.get(routes.INFO)
        assert info.status_code == HTTPStatus.OK
        assert model_name in info.json()

        names = self.client.get(routes.INFO_NAMES)
        assert names.status_code == HTTPStatus.OK

    # === Real fairness metrics reading from SQLite ===========================
    def test_spd_computed_from_sqlite(self) -> None:
        """SPD endpoint computes the expected value from real SQLite data."""
        model_id = f"e2e_spd_{uuid.uuid4().hex[:8]}"
        self._seed_fairness_data(model_id)

        response = self.client.post(
            routes.FAIRNESS_SPD.compute, json=self._fairness_payload(model_id)
        )
        assert response.status_code == HTTPStatus.OK, response.text
        data = response.json()
        assert data["value"] == pytest.approx(_EXPECTED_SPD, abs=_TOL)
        # -0.30 is well outside the default +/-0.1 fairness band.
        assert data["thresholds"]["outsideBounds"] is True

    def test_dir_computed_from_sqlite(self) -> None:
        """DIR endpoint computes the expected value from real SQLite data."""
        model_id = f"e2e_dir_{uuid.uuid4().hex[:8]}"
        self._seed_fairness_data(model_id)

        response = self.client.post(
            routes.FAIRNESS_DIR.compute, json=self._fairness_payload(model_id)
        )
        assert response.status_code == HTTPStatus.OK, response.text
        data = response.json()
        assert data["value"] == pytest.approx(_EXPECTED_DIR, abs=_TOL)

    def test_batch_mean_computed_from_sqlite(self) -> None:
        """Batch-mean endpoint computes the outcome mean from real SQLite data."""
        model_id = f"e2e_mean_{uuid.uuid4().hex[:8]}"
        self._seed_fairness_data(model_id)

        payload = {
            "modelId": model_id,
            "columnName": "income",
            "batchSize": 1000,
        }
        response = self.client.post(routes.BATCH_MEAN.compute, json=payload)
        assert response.status_code == HTTPStatus.OK, response.text
        data = response.json()
        assert data["value"] == pytest.approx(_EXPECTED_OUTCOME_MEAN, abs=_TOL)

    # === Name-mapping through the storage backend ============================
    def test_name_mapping_round_trip(self) -> None:
        """Applying and clearing a name mapping updates the stored aliases."""
        model_id = f"e2e_map_{uuid.uuid4().hex[:8]}"
        self._seed_fairness_data(model_id)
        inputs = f"{model_id}_inputs"

        async def _apply() -> list[str]:
            await self.storage_interface.apply_name_mapping(
                inputs, {"gender": "Gender"}
            )
            return await self.storage_interface.get_aliased_column_names(inputs)

        assert asyncio.run(_apply()) == ["Gender"]

        async def _clear() -> list[str]:
            await self.storage_interface.clear_name_mapping(inputs)
            return await self.storage_interface.get_aliased_column_names(inputs)

        # Original names are preserved and restored on clear.
        assert asyncio.run(_clear()) == ["gender"]

    # === Repeated upload appends through the real backend =====================
    def test_repeated_upload_appends_rows(self) -> None:
        """Uploading twice to the same model appends rows in SQLite."""
        model_name = f"e2e_append_{uuid.uuid4().hex[:8]}"
        n_rows = 4

        def _payload() -> dict:
            return {
                "model_name": model_name,
                "data_tag": "TRAINING",
                "is_ground_truth": False,
                "request": {
                    "inputs": [
                        {
                            "name": "input",
                            "shape": [n_rows, 2],
                            "datatype": "INT64",
                            "data": [[i, i + 1] for i in range(n_rows)],
                        }
                    ]
                },
                "response": {
                    "outputs": [
                        {
                            "name": "output",
                            "shape": [n_rows],
                            "datatype": "INT64",
                            "data": [i * 2 for i in range(n_rows)],
                        }
                    ]
                },
            }

        assert self.client.post(routes.DATA_UPLOAD, json=_payload()).status_code == (
            HTTPStatus.OK
        )
        assert self.client.post(routes.DATA_UPLOAD, json=_payload()).status_code == (
            HTTPStatus.OK
        )

        rows = asyncio.run(self.storage_interface.dataset_rows(f"{model_name}_inputs"))
        assert rows == 2 * n_rows


if __name__ == "__main__":
    unittest.main()
