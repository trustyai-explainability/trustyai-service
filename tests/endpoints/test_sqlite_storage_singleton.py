"""Exercise storage sharing across both FastAPI apps with fresh process state."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")


def _exercise_service() -> None:
    """Import the real apps after SQLite is configured, without resetting storage."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from trustyai_service.endpoints import routes  # noqa: PLC0415
    from trustyai_service.main import app, health_app  # noqa: PLC0415
    from trustyai_service.service.data.datasources.data_source import (  # noqa: PLC0415
        DataSource,
    )

    n_rows = 3
    request = {
        "inputs": [
            {
                "name": "input",
                "shape": [n_rows, 2],
                "datatype": "INT64",
                "data": [[1, 2], [3, 4], [5, 6]],
            },
        ],
    }
    response = {
        "outputs": [
            {
                "name": "output",
                "shape": [n_rows],
                "datatype": "INT64",
                "data": [2, 4, 6],
            },
        ],
    }

    with TestClient(app) as client, TestClient(health_app) as consumer:
        for source in ("upload", "consumer"):
            model = f"sqlite_{source}"
            if source == "upload":
                result = client.post(
                    routes.DATA_UPLOAD,
                    json={
                        "model_name": model,
                        "data_tag": "TRAINING",
                        "request": request,
                        "response": response,
                    },
                )
                assert result.status_code == 200, result.text
            else:
                for payload in (request, {**response, "model_name": model}):
                    result = consumer.post(
                        routes.CONSUMER_ROOT,
                        json=payload,
                        headers={"ce-id": "shared-storage-request"},
                        params={"tag": "TRAINING"},
                    )
                    assert result.status_code == 200, result.text

            info = client.get(routes.INFO)
            assert info.status_code == 200, info.text
            model_info = info.json()[model]
            assert "error" not in model_info, model_info
            assert model_info["data"]["observations"] == n_rows
            assert len(model_info["data"]["inputSchema"]["items"]) == 2
            assert len(model_info["data"]["outputSchema"]["items"]) == 1

            # A new DataSource must discover data written through either app.
            assert model in asyncio.run(DataSource().get_verified_models())

            ids = client.get(routes.INFO_INFERENCE_IDS.format(model=model))
            assert ids.status_code == 200, ids.text
            assert ids.json()["total"] == n_rows

            mapping = {"input-0": "Feature One"}
            result = client.post(
                routes.INFO_NAMES,
                json={"modelId": model, "inputMapping": mapping},
            )
            assert result.status_code == 200, result.text
            names = client.get(routes.INFO_NAMES)
            assert names.status_code == 200, names.text
            assert names.json()[model]["inputMapping"] == mapping
            result = client.request("DELETE", routes.INFO_NAMES, json=model)
            assert result.status_code == 200, result.text
            assert client.get(routes.INFO_NAMES).json()[model]["inputMapping"] == {}

            tags = client.get(routes.INFO_TAGS, params={"modelId": model})
            assert tags.status_code == 200, tags.text
            assert tags.json()["TRAINING"] == n_rows


@pytest.mark.parametrize("backend", ["memory", "file"])
def test_apps_share_sqlite_storage(backend: str, tmp_path: Path) -> None:
    """Both apps and newly constructed data sources see the same SQLite data."""
    database_path = ":memory:" if backend == "memory" else str(tmp_path / "data.sqlite")
    result = subprocess.run(  # noqa: S603 -- fixed test script in a fresh interpreter
        [sys.executable, str(Path(__file__).resolve())],
        env={
            **os.environ,
            "SERVICE_STORAGE_FORMAT": "SQLITE",
            "STORAGE_DATABASE_PATH": database_path,
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if backend == "memory":
        assert not list(tmp_path.iterdir())
    else:
        assert Path(database_path).is_file()


if __name__ == "__main__":
    _exercise_service()
