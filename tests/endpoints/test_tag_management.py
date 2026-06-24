"""Tests for GET /info/tags and POST /info/tags endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.main import app

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

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_get_tags_single_model(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Returns correct tag counts for a specific model."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        tags = [
            ["_trustyai_unlabeled"],
            ["_trustyai_unlabeled"],
            ["TRAINING"],
            ["TRAINING", "REFERENCE"],
        ]

        with patch(
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(4, tags)),
        ):
            response = client.get("/info/tags", params={"modelId": "test-model"})

        assert response.status_code == 200  # noqa: PLR2004
        data = response.json()
        assert data["_trustyai_unlabeled"] == 2  # noqa: PLR2004
        assert data["TRAINING"] == 2  # noqa: PLR2004
        assert data["REFERENCE"] == 1

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_get_tags_model_not_found(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Returns 404 for a non-existent model."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value=set())
        mock_get_ds.return_value = mock_ds

        response = client.get("/info/tags", params={"modelId": "nonexistent"})

        assert response.status_code == 404  # noqa: PLR2004

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_get_tags_empty_data(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Returns empty dict when model has no metadata rows."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"empty-model"})
        mock_get_ds.return_value = mock_ds

        md = MagicMock()
        md.data = AsyncMock(return_value=(None, None, None))
        md.column_names = AsyncMock(return_value=([], [], METADATA_NAMES))

        with patch("src.endpoints.metadata.ModelData", return_value=md):
            response = client.get("/info/tags", params={"modelId": "empty-model"})

        assert response.status_code == 200  # noqa: PLR2004
        assert response.json() == {}

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_get_tags_all_models(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Returns tag counts for all known models when modelId is omitted."""
        mock_ds = MagicMock()
        mock_ds.get_verified_models = AsyncMock(return_value=["model-a", "model-b"])
        mock_ds.get_known_models = AsyncMock(return_value={"model-a", "model-b"})
        mock_get_ds.return_value = mock_ds

        tags_a = [["TRAINING"], ["TRAINING"]]
        tags_b = [["REFERENCE"]]
        meta_a = _make_metadata_rows(2, tags_a)
        meta_b = _make_metadata_rows(1, tags_b)

        def make_model_data(model_id: str) -> MagicMock:
            return _mock_model_data(meta_a if model_id == "model-a" else meta_b)

        with patch(
            "src.endpoints.metadata.ModelData",
            side_effect=make_model_data,
        ):
            response = client.get("/info/tags")

        assert response.status_code == 200  # noqa: PLR2004
        data = response.json()
        assert "model-a" in data
        assert "model-b" in data
        assert data["model-a"]["TRAINING"] == 2  # noqa: PLR2004
        assert data["model-b"]["REFERENCE"] == 1


class TestApplyTags:
    """Tests for POST /info/tags."""

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_success(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Applies tags to specified row ranges."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        mock_storage.delete_dataset = AsyncMock()
        mock_storage.write_data = AsyncMock()

        with patch(
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[0, 5]]},
                },
            )

        assert response.status_code == 200  # noqa: PLR2004
        data = response.json()
        assert data["applied"]["TRAINING"] == 5  # noqa: PLR2004
        mock_storage.delete_dataset.assert_awaited_once()
        mock_storage.write_data.assert_awaited_once()

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_reserved_prefix(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Rejects tags with the reserved _trustyai prefix."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        response = client.post(
            "/info/tags",
            json={
                "modelId": "test-model",
                "dataTagging": {"_trustyai_custom": [[0, 5]]},
            },
        )

        assert response.status_code == 400  # noqa: PLR2004
        assert "_trustyai" in response.json()["detail"]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_out_of_bounds(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Rejects ranges that exceed dataset size."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(5)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[0, 100]]},
                },
            )

        assert response.status_code == 400  # noqa: PLR2004
        assert "exceeds dataset size" in response.json()["detail"]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_idempotent(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Applying the same tag twice does not duplicate it in the row."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        metadata = _make_metadata_rows(3, [["TRAINING"]] * 3)
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
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(metadata),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[0, 3]]},
                },
            )

        assert response.status_code == 200  # noqa: PLR2004
        saved = written_data["metadata"]
        for row in saved:
            assert row[3].count("TRAINING") == 1

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_model_not_found(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Returns 404 for a non-existent model."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value=set())
        mock_get_ds.return_value = mock_ds

        response = client.post(
            "/info/tags",
            json={
                "modelId": "nonexistent",
                "dataTagging": {"TRAINING": [[0, 5]]},
            },
        )

        assert response.status_code == 404  # noqa: PLR2004

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_invalid_range(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Rejects ranges where start >= end."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[5, 3]]},
                },
            )

        assert response.status_code == 400  # noqa: PLR2004
        assert "start must be less than end" in response.json()["detail"]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_negative_index(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Rejects ranges containing negative indices."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TRAINING": [[-1, 5]]},
                },
            )

        assert response.status_code == 400  # noqa: PLR2004
        assert "non-negative" in response.json()["detail"]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_overlapping_ranges(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Overlapping ranges apply tags correctly without duplicates."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

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
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TAG": [[0, 5], [3, 8]]},
                },
            )

        assert response.status_code == 200  # noqa: PLR2004
        saved = written_data["metadata"]
        # Rows 3 and 4 appear in both ranges but TAG should appear only once
        for idx in range(8):
            assert saved[idx][3].count("TAG") == 1

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_multiple_tags_different_ranges(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Multiple tags applied to different ranges in a single request."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

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
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {
                        "TRAINING": [[0, 3]],
                        "REFERENCE": [[7, 10]],
                    },
                },
            )

        assert response.status_code == 200  # noqa: PLR2004
        data = response.json()
        assert data["applied"]["TRAINING"] == 3  # noqa: PLR2004
        assert data["applied"]["REFERENCE"] == 3  # noqa: PLR2004
        saved = written_data["metadata"]
        assert "TRAINING" in saved[0][3]
        assert "TRAINING" not in saved[9][3]
        assert "REFERENCE" in saved[9][3]
        assert "REFERENCE" not in saved[0][3]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_non_contiguous_ranges(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Non-contiguous ranges tag only specified rows."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

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
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TAG": [[0, 2], [5, 8]]},
                },
            )

        assert response.status_code == 200  # noqa: PLR2004
        saved = written_data["metadata"]
        # Rows 0, 1 and 5, 6, 7 should have TAG
        for idx in (0, 1, 5, 6, 7):
            assert "TAG" in saved[idx][3]
        # Gap rows should not have TAG
        for idx in (2, 3, 4, 8, 9):
            assert "TAG" not in saved[idx][3]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_singleton_range(
        self, mock_get_ds: MagicMock, mock_storage: MagicMock
    ) -> None:
        """Singleton range [n] tags only row n."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

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
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TAG": [[3]]},
                },
            )

        assert response.status_code == 200  # noqa: PLR2004
        saved = written_data["metadata"]
        assert "TAG" in saved[3][3]
        assert "TAG" not in saved[2][3]
        assert "TAG" not in saved[4][3]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_empty_data_tagging(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Empty dataTagging dict returns 400."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        response = client.post(
            "/info/tags",
            json={
                "modelId": "test-model",
                "dataTagging": {},
            },
        )

        assert response.status_code == 400  # noqa: PLR2004
        assert "at least one tag" in response.json()["detail"]

    @patch("src.endpoints.metadata.storage_interface")
    @patch("src.endpoints.metadata.get_data_source")
    def test_apply_tags_range_too_many_elements(
        self, mock_get_ds: MagicMock, _mock_storage: MagicMock
    ) -> None:
        """Range with 3+ elements returns 400."""
        mock_ds = MagicMock()
        mock_ds.get_known_models = AsyncMock(return_value={"test-model"})
        mock_get_ds.return_value = mock_ds

        with patch(
            "src.endpoints.metadata.ModelData",
            return_value=_mock_model_data(_make_metadata_rows(10)),
        ):
            response = client.post(
                "/info/tags",
                json={
                    "modelId": "test-model",
                    "dataTagging": {"TAG": [[0, 5, 10]]},
                },
            )

        assert response.status_code == 400  # noqa: PLR2004
        assert "[start, end] or [index]" in response.json()["detail"]
