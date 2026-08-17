"""Tests for GET /info/tags and POST /info/tags endpoints."""

from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from trustyai_service.endpoints import routes
from trustyai_service.main import app

client = TestClient(app)


def _make_metadata_rows(
    n: int,
    tags_per_row: list[list[str]] | None = None,
) -> np.ndarray:
    """Build a metadata ndarray with shape (n, 4).

    Columns: id, iso_time, unix_timestamp, tags.
    """
    if tags_per_row is None:
        tags_per_row = [["_trustyai_unlabeled"] for _ in range(n)]
    return np.array(
        [
            [
                f"req_{i}",
                "2026-06-22T00:00:00",
                1750600000.0,
                tags_per_row[i] if i < len(tags_per_row) else [],
            ]
            for i in range(n)
        ],
        dtype="O",
    )


METADATA_NAMES = ["id", "iso_time", "unix_timestamp", "tags"]


def _mock_model_data(metadata: np.ndarray) -> MagicMock:
    md = MagicMock()
    md.data = AsyncMock(return_value=(None, None, metadata))
    md.column_names = AsyncMock(return_value=([], [], METADATA_NAMES))
    return md


class TestGetTags:
    """Tests for GET /info/tags."""

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_get_tags_single_model(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Returns correct tag counts for a specific model."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists to return True (model exists)
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        tags = [
            ["_trustyai_unlabeled"],
            ["_trustyai_unlabeled"],
            ["TRAINING"],
            ["TRAINING", "REFERENCE"],
        ]

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(4, tags)),
        ):
            response = client.get(routes.INFO_TAGS, params={"modelId": "test-model"})

        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert data["_trustyai_unlabeled"] == 2  # noqa: PLR2004
        assert data["TRAINING"] == 2  # noqa: PLR2004
        assert data["REFERENCE"] == 1

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_get_tags_model_not_found(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Returns 404 for a non-existent model."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value=set())
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists to return False (model doesn't exist)
        mock_storage.dataset_exists = AsyncMock(return_value=False)

        response = client.get(routes.INFO_TAGS, params={"modelId": "nonexistent"})

        assert response.status_code == HTTPStatus.NOT_FOUND

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_get_tags_empty_data(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Returns empty dict when model has no metadata rows."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"empty-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists to return True (model exists)
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        md = MagicMock()
        md.data = AsyncMock(return_value=(None, None, None))
        md.column_names = AsyncMock(return_value=([], [], METADATA_NAMES))

        with patch("trustyai_service.endpoints.metadata.ModelData", return_value=md):
            response = client.get(routes.INFO_TAGS, params={"modelId": "empty-model"})

        assert response.status_code == HTTPStatus.OK
        assert response.json() == {}

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_get_tags_all_models(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Returns tag counts for all known models when modelId is omitted."""
        mock_ds = MagicMock()
        mock_ds.get_verified_models = AsyncMock(return_value=["model-a", "model-b"])
        mock_ds.get_known_models = AsyncMock(return_value={"model-a", "model-b"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists to return True (models exist)
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        tags_a = [["TRAINING"], ["TRAINING"]]
        tags_b = [["REFERENCE"]]
        meta_a = _make_metadata_rows(2, tags_a)
        meta_b = _make_metadata_rows(1, tags_b)

        def make_model_data(model_id: str) -> MagicMock:
            return _mock_model_data(meta_a if model_id == "model-a" else meta_b)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            side_effect=make_model_data,
        ):
            response = client.get(routes.INFO_TAGS)

        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert "model-a" in data
        assert "model-b" in data
        assert data["model-a"]["TRAINING"] == 2  # noqa: PLR2004
        assert data["model-b"]["REFERENCE"] == 1


class TestApplyTags:
    """Tests for POST /info/tags."""

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_success(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Applies tags to specified row ranges."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface for model existence check and persistence
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.read_data = AsyncMock(return_value=_make_metadata_rows(10))
        mock_storage.read_column_names = AsyncMock(return_value=METADATA_NAMES)
        mock_storage.delete_dataset = AsyncMock()
        mock_storage.write_data = AsyncMock()

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[0, 5]]},
                },
            )

        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert data["applied"]["TRAINING"] == 5  # noqa: PLR2004
        mock_storage.delete_dataset.assert_awaited_once()
        mock_storage.write_data.assert_awaited_once()

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_reserved_prefix(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Rejects tags with the reserved _trustyai prefix."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists (validation happens before storage)
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        response = client.post(
            routes.INFO_TAGS,
            json={
                "modelId": "test-model",
                "dataTagging": {"_trustyai_custom": [[0, 5]]},
            },
        )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "_trustyai" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_out_of_bounds(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Rejects ranges that exceed dataset size."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(5)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[0, 100]]},
                },
            )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "exceeds dataset size" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_idempotent(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Applying the same tag twice does not duplicate it in the row."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        metadata = _make_metadata_rows(3, [["TRAINING"]] * 3)
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.read_data = AsyncMock(return_value=_make_metadata_rows(10))
        mock_storage.read_column_names = AsyncMock(return_value=METADATA_NAMES)
        mock_storage.delete_dataset = AsyncMock()

        written_data: dict[str, np.ndarray] = {}

        async def capture_write(
            _dataset_name: str,
            data: np.ndarray,
            _col_names: list[str],
        ) -> None:
            written_data["metadata"] = data

        mock_storage.write_data = AsyncMock(side_effect=capture_write)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(metadata),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[0, 3]]},
                },
            )

        assert response.status_code == HTTPStatus.OK
        saved = written_data["metadata"]
        for row in saved:
            assert row[3].count("TRAINING") == 1

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_model_not_found(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Returns 404 for a non-existent model."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value=set())
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists to return False
        mock_storage.dataset_exists = AsyncMock(return_value=False)

        response = client.post(
            routes.INFO_TAGS,
            json={
                "modelId": "nonexistent",
                "dataTagging": {"TRAINING": [[0, 5]]},
            },
        )

        assert response.status_code == HTTPStatus.NOT_FOUND

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_invalid_range(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Rejects ranges where start >= end."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[5, 3]]},
                },
            )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "start must be less than end" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_negative_index(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Rejects ranges containing negative indices."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[-1, 5]]},
                },
            )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "non-negative" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_overlapping_ranges(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Overlapping ranges apply tags correctly without duplicates."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.read_data = AsyncMock(return_value=_make_metadata_rows(10))
        mock_storage.read_column_names = AsyncMock(return_value=METADATA_NAMES)
        mock_storage.delete_dataset = AsyncMock()

        written_data: dict[str, np.ndarray] = {}

        async def capture_write(
            _dataset_name: str,
            data: np.ndarray,
            _col_names: list[str],
        ) -> None:
            written_data["metadata"] = data

        mock_storage.write_data = AsyncMock(side_effect=capture_write)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TAG": [[0, 5], [3, 8]]},
                },
            )

        assert response.status_code == HTTPStatus.OK
        saved = written_data["metadata"]
        # Rows 3 and 4 appear in both ranges but TAG should appear only once
        for idx in range(8):
            assert saved[idx][3].count("TAG") == 1

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_multiple_tags_different_ranges(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Multiple tags applied to different ranges in a single request."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.read_data = AsyncMock(return_value=_make_metadata_rows(10))
        mock_storage.read_column_names = AsyncMock(return_value=METADATA_NAMES)
        mock_storage.delete_dataset = AsyncMock()

        written_data: dict[str, np.ndarray] = {}

        async def capture_write(
            _dataset_name: str,
            data: np.ndarray,
            _col_names: list[str],
        ) -> None:
            written_data["metadata"] = data

        mock_storage.write_data = AsyncMock(side_effect=capture_write)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {
                        "TRAINING": [[0, 3]],
                        "REFERENCE": [[7, 10]],
                    },
                },
            )

        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert data["applied"]["TRAINING"] == 3  # noqa: PLR2004
        assert data["applied"]["REFERENCE"] == 3  # noqa: PLR2004
        saved = written_data["metadata"]
        assert "TRAINING" in saved[0][3]
        assert "TRAINING" not in saved[9][3]
        assert "REFERENCE" in saved[9][3]
        assert "REFERENCE" not in saved[0][3]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_non_contiguous_ranges(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Non-contiguous ranges tag only specified rows."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.read_data = AsyncMock(return_value=_make_metadata_rows(10))
        mock_storage.read_column_names = AsyncMock(return_value=METADATA_NAMES)
        mock_storage.delete_dataset = AsyncMock()

        written_data: dict[str, np.ndarray] = {}

        async def capture_write(
            _dataset_name: str,
            data: np.ndarray,
            _col_names: list[str],
        ) -> None:
            written_data["metadata"] = data

        mock_storage.write_data = AsyncMock(side_effect=capture_write)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TAG": [[0, 2], [5, 8]]},
                },
            )

        assert response.status_code == HTTPStatus.OK
        saved = written_data["metadata"]
        # Rows 0, 1 and 5, 6, 7 should have TAG
        for idx in (0, 1, 5, 6, 7):
            assert "TAG" in saved[idx][3]
        # Gap rows should not have TAG
        for idx in (2, 3, 4, 8, 9):
            assert "TAG" not in saved[idx][3]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_singleton_range(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Singleton range [n] tags only row n."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.read_data = AsyncMock(return_value=_make_metadata_rows(10))
        mock_storage.read_column_names = AsyncMock(return_value=METADATA_NAMES)
        mock_storage.delete_dataset = AsyncMock()

        written_data: dict[str, np.ndarray] = {}

        async def capture_write(
            _dataset_name: str,
            data: np.ndarray,
            _col_names: list[str],
        ) -> None:
            written_data["metadata"] = data

        mock_storage.write_data = AsyncMock(side_effect=capture_write)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TAG": [[3]]},
                },
            )

        assert response.status_code == HTTPStatus.OK
        saved = written_data["metadata"]
        assert "TAG" in saved[3][3]
        assert "TAG" not in saved[2][3]
        assert "TAG" not in saved[4][3]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_empty_data_tagging(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Empty dataTagging dict returns 400."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists (validation happens before storage)
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        response = client.post(
            routes.INFO_TAGS,
            json={
                "modelId": "test-model",
                "dataTagging": {},
            },
        )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "at least one tag" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_apply_tags_range_too_many_elements(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Range with 3+ elements returns 400."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        # Configure storage_interface.dataset_exists
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        with patch(
            "trustyai_service.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                routes.INFO_TAGS,
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TAG": [[0, 5, 10]]},
                },
            )

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "[start, end] or [index]" in response.json()["detail"]
