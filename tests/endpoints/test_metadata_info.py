"""Tests for /info and /info/names metadata endpoints.

Covers GET /info, GET /info/names, POST /info/names, DELETE /info/names,
GET /info/tags, POST /info/tags, and the _build_readable_schema helper.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from trustyai_service.endpoints.metadata import _build_readable_schema
from trustyai_service.main import app
from trustyai_service.service.payloads.service.schema import Schema
from trustyai_service.service.payloads.service.schema_item import SchemaItem
from trustyai_service.service.payloads.values.data_type import DataType

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_metadata(
    *,
    input_items: dict[str, SchemaItem] | None = None,
    output_items: dict[str, SchemaItem] | None = None,
    input_tensor: str = "input",
    output_tensor: str = "output",
) -> MagicMock:
    """Build a mock StorageMetadata with sensible defaults."""
    meta = MagicMock()
    meta.input_tensor_name = input_tensor
    meta.output_tensor_name = output_tensor
    meta.input_schema = Schema(
        items=input_items
        if input_items is not None
        else {"f1": SchemaItem(DataType.DOUBLE, "f1", 0)},
    )
    meta.output_schema = Schema(
        items=output_items
        if output_items is not None
        else {"out": SchemaItem(DataType.DOUBLE, "out", 0)},
    )
    return meta


def _make_mock_data_source(
    known_models: set[str] | None = None,
    metadata_map: dict[str, MagicMock] | None = None,
    observation_counts: dict[str, int] | None = None,
    has_inferences: dict[str, bool] | None = None,
) -> MagicMock:
    """Build a mock DataSource with common async methods wired up."""
    models = known_models or set()
    meta_map = metadata_map or {}
    obs = observation_counts or {}
    infs = has_inferences or {}

    ds = MagicMock()
    ds.get_known_models = AsyncMock(return_value=models)
    ds.get_metadata = AsyncMock(
        side_effect=meta_map.get,
    )
    ds.get_num_observations = AsyncMock(
        side_effect=lambda mid: obs.get(mid, 0),
    )
    ds.has_recorded_inferences = AsyncMock(
        side_effect=lambda mid: infs.get(mid, False),
    )
    return ds


# ---------------------------------------------------------------------------
# _build_readable_schema unit tests
# ---------------------------------------------------------------------------


class TestBuildReadableSchema:
    """Tests for _build_readable_schema helper."""

    def test_converts_schema_with_items(self) -> None:
        """Schema items are serialized with type and index."""
        schema = Schema(
            items={
                "col1": SchemaItem(DataType.DOUBLE, "col1", 0),
                "col2": SchemaItem(DataType.INT32, "col2", 1),
            },
        )
        result = _build_readable_schema(schema, ["col1", "col2"], ["col1", "col2"])

        assert result["items"]["col1"] == {"type": "DOUBLE", "index": 0}
        assert result["items"]["col2"] == {"type": "INT32", "index": 1}
        assert result["nameMapping"] == {}

    def test_includes_name_mapping(self) -> None:
        """Name mappings derived from original vs aliased column names."""
        schema = Schema(
            items={"col1": SchemaItem(DataType.DOUBLE, "col1", 0)},
        )
        result = _build_readable_schema(schema, ["col1"], ["Friendly Name"])

        assert result["nameMapping"] == {"col1": "Friendly Name"}

    def test_empty_schema(self) -> None:
        """Empty schema returns empty items and nameMapping."""
        schema = Schema()
        result = _build_readable_schema(schema, [], [])

        assert result == {"items": {}, "nameMapping": {}}

    def test_multiple_renamed_columns(self) -> None:
        """Multiple columns with different aliases are all captured."""
        schema = Schema(
            items={
                "a": SchemaItem(DataType.FLOAT, "a", 0),
                "b": SchemaItem(DataType.INT64, "b", 1),
            },
        )
        result = _build_readable_schema(schema, ["a", "b"], ["Alpha", "Beta"])

        assert result["nameMapping"] == {"a": "Alpha", "b": "Beta"}

    def test_partial_rename_only_includes_changed(self) -> None:
        """Only columns whose alias differs from original appear in mapping."""
        schema = Schema(
            items={
                "x": SchemaItem(DataType.DOUBLE, "x", 0),
                "y": SchemaItem(DataType.DOUBLE, "y", 1),
            },
        )
        result = _build_readable_schema(schema, ["x", "y"], ["Renamed X", "y"])

        assert result["nameMapping"] == {"x": "Renamed X"}

    def test_string_enum_value_used_as_type(self) -> None:
        """DataType StrEnum values are serialized via .value."""
        schema = Schema(
            items={"col": SchemaItem(DataType.BOOL, "col", 0)},
        )
        result = _build_readable_schema(schema, ["col"], ["col"])

        assert result["items"]["col"]["type"] == "BOOL"


# ---------------------------------------------------------------------------
# GET /info
# ---------------------------------------------------------------------------


class TestGetInfoEndpoint:
    """Tests for GET /info endpoint."""

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_returns_empty_when_no_models(
        self,
        mock_get_ds: MagicMock,
        _mock_storage: MagicMock,
        _mock_sched: MagicMock,
    ) -> None:
        """No known models returns an empty dict."""
        mock_get_ds.return_value = _make_mock_data_source(known_models=set())

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        assert response.json() == {}

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_single_model_full_metadata(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
        mock_sched: MagicMock,
    ) -> None:
        """Single model returns complete metadata structure."""
        meta = _make_mock_metadata()
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"model-a"},
            metadata_map={"model-a": meta},
            observation_counts={"model-a": 42},
            has_inferences={"model-a": True},
        )
        mock_storage.get_original_column_names = AsyncMock(
            side_effect=lambda ds: ["f1"] if "inputs" in ds else ["out"],
        )
        mock_storage.get_aliased_column_names = AsyncMock(
            side_effect=lambda ds: ["f1"] if "inputs" in ds else ["out"],
        )
        mock_sched.return_value = None  # no scheduler

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert "model-a" in body

        data = body["model-a"]["data"]
        assert data["observations"] == 42  # noqa: PLR2004
        assert data["hasRecordedInferences"] is True
        assert data["inputTensorName"] == "input"
        assert data["outputTensorName"] == "output"
        assert "inputSchema" in data
        assert "outputSchema" in data
        assert data["inputSchema"]["items"]["f1"] == {
            "type": "DOUBLE",
            "index": 0,
        }

        metrics = body["model-a"]["metrics"]
        assert "scheduledMetadata" in metrics

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_includes_schemas_with_name_mapping(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
        mock_sched: MagicMock,
    ) -> None:
        """Aliased column names appear in inputSchema.nameMapping."""
        meta = _make_mock_metadata()
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"test-model"},
            metadata_map={"test-model": meta},
            observation_counts={"test-model": 100},
            has_inferences={"test-model": True},
        )
        mock_storage.get_original_column_names = AsyncMock(
            side_effect=lambda ds: ["f1"] if "inputs" in ds else ["out"],
        )
        mock_storage.get_aliased_column_names = AsyncMock(
            side_effect=lambda ds: ["Feature One"] if "inputs" in ds else ["out"],
        )
        mock_sched.return_value = None

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        data = response.json()["test-model"]["data"]
        assert data["inputSchema"]["nameMapping"] == {"f1": "Feature One"}
        assert data["outputSchema"]["nameMapping"] == {}

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_multiple_models(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
        mock_sched: MagicMock,
    ) -> None:
        """Multiple models each get their own metadata block."""
        meta_a = _make_mock_metadata(input_tensor="in_a", output_tensor="out_a")
        meta_b = _make_mock_metadata(input_tensor="in_b", output_tensor="out_b")
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"alpha", "beta"},
            metadata_map={"alpha": meta_a, "beta": meta_b},
            observation_counts={"alpha": 10, "beta": 20},
            has_inferences={"alpha": True, "beta": False},
        )
        mock_storage.get_original_column_names = AsyncMock(return_value=["f1"])
        mock_storage.get_aliased_column_names = AsyncMock(return_value=["f1"])
        mock_sched.return_value = None

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert "alpha" in body
        assert "beta" in body
        assert body["alpha"]["data"]["inputTensorName"] == "in_a"
        assert body["beta"]["data"]["inputTensorName"] == "in_b"
        assert body["alpha"]["data"]["observations"] == 10  # noqa: PLR2004
        assert body["beta"]["data"]["observations"] == 20  # noqa: PLR2004

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_scheduled_metrics_counted(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
        mock_sched: MagicMock,
    ) -> None:
        """Scheduled metric requests are counted per model."""
        meta = _make_mock_metadata()
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"my-model"},
            metadata_map={"my-model": meta},
            observation_counts={"my-model": 5},
            has_inferences={"my-model": True},
        )
        mock_storage.get_original_column_names = AsyncMock(return_value=["f1"])
        mock_storage.get_aliased_column_names = AsyncMock(return_value=["f1"])

        # Build a scheduler mock with scheduled requests
        scheduler = MagicMock()
        req1 = MagicMock()
        req1.model_id = "my-model"
        req2 = MagicMock()
        req2.model_id = "my-model"
        req_other = MagicMock()
        req_other.model_id = "other-model"
        scheduler.get_all_requests.return_value = {
            "spd": {uuid.uuid4(): req1, uuid.uuid4(): req2},
            "dir": {uuid.uuid4(): req_other},
        }
        mock_sched.return_value = scheduler

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        scheduled = response.json()["my-model"]["metrics"]["scheduledMetadata"]
        assert scheduled["spd"] == 2  # noqa: PLR2004
        assert "dir" not in scheduled

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_uses_model_id_attr_fallback(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
        mock_sched: MagicMock,
    ) -> None:
        """Scheduler request checked via modelId attr when model_id absent."""
        meta = _make_mock_metadata()
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"m1"},
            metadata_map={"m1": meta},
            observation_counts={"m1": 1},
            has_inferences={"m1": True},
        )
        mock_storage.get_original_column_names = AsyncMock(return_value=["f1"])
        mock_storage.get_aliased_column_names = AsyncMock(return_value=["f1"])

        scheduler = MagicMock()
        req = MagicMock(spec=[])  # no model_id attribute
        req.modelId = "m1"
        scheduler.get_all_requests.return_value = {
            "drift": {uuid.uuid4(): req},
        }
        mock_sched.return_value = scheduler

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        scheduled = response.json()["m1"]["metrics"]["scheduledMetadata"]
        assert scheduled["drift"] == 1

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_per_model_error_does_not_break_response(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
        mock_sched: MagicMock,
    ) -> None:
        """If one model fails, others still appear and the failing model gets a fallback entry."""
        meta_ok = _make_mock_metadata()
        ds = _make_mock_data_source(
            known_models={"good", "bad"},
            metadata_map={"good": meta_ok},
            observation_counts={"good": 10},
            has_inferences={"good": True},
        )
        # bad model raises on get_metadata
        original_get_metadata = ds.get_metadata.side_effect

        async def metadata_side_effect(mid: str) -> MagicMock | None:
            if mid == "bad":
                msg = "storage offline"
                raise RuntimeError(msg)
            return original_get_metadata(mid)

        ds.get_metadata = AsyncMock(side_effect=metadata_side_effect)

        # Also wire get_num_observations and has_recorded_inferences to error for "bad"
        original_get_obs = ds.get_num_observations.side_effect

        async def obs_side_effect(mid: str) -> int:
            if mid == "bad":
                msg = "storage offline"
                raise RuntimeError(msg)
            return original_get_obs(mid)

        ds.get_num_observations = AsyncMock(side_effect=obs_side_effect)

        mock_get_ds.return_value = ds
        mock_storage.get_original_column_names = AsyncMock(return_value=["f1"])
        mock_storage.get_aliased_column_names = AsyncMock(return_value=["f1"])
        mock_sched.return_value = None

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        # "good" should have real data
        assert body["good"]["data"]["observations"] == 10  # noqa: PLR2004
        # "bad" should have the fallback structure with an error key
        assert body["bad"]["data"]["observations"] == 0
        assert body["bad"]["data"]["hasRecordedInferences"] is False
        assert "error" in body["bad"]

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_scheduler_error_does_not_break_response(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
        mock_sched: MagicMock,
    ) -> None:
        """Scheduler errors are silently handled; metadata still returned."""
        meta = _make_mock_metadata()
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"m"},
            metadata_map={"m": meta},
            observation_counts={"m": 7},
            has_inferences={"m": True},
        )
        mock_storage.get_original_column_names = AsyncMock(return_value=["f1"])
        mock_storage.get_aliased_column_names = AsyncMock(return_value=["f1"])

        scheduler = MagicMock()
        scheduler.get_all_requests.side_effect = RuntimeError("scheduler crash")
        mock_sched.return_value = scheduler

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert body["m"]["data"]["observations"] == 7  # noqa: PLR2004
        # scheduledMetadata should be empty because the scheduler errored
        assert body["m"]["metrics"]["scheduledMetadata"] == {}

    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_data_source_failure_returns_500(
        self,
        mock_get_ds: MagicMock,
    ) -> None:
        """Complete data source failure yields HTTP 500."""
        mock_get_ds.return_value.get_known_models = AsyncMock(
            side_effect=RuntimeError("DB down"),
        )

        response = client.get("/info")

        assert response.status_code == 500  # noqa: PLR2004
        assert "Error retrieving service info" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.get_prometheus_scheduler")
    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_info_none_metadata_uses_defaults(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
        mock_sched: MagicMock,
    ) -> None:
        """When get_metadata returns None, default tensor names and empty schemas are used."""
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"null-meta"},
            metadata_map={"null-meta": None},
            observation_counts={"null-meta": 0},
            has_inferences={"null-meta": False},
        )
        mock_storage.get_original_column_names = AsyncMock(return_value=[])
        mock_storage.get_aliased_column_names = AsyncMock(return_value=[])
        mock_sched.return_value = None

        response = client.get("/info")

        assert response.status_code == 200  # noqa: PLR2004
        data = response.json()["null-meta"]["data"]
        assert data["inputTensorName"] == "input"
        assert data["outputTensorName"] == "output"
        assert data["inputSchema"] == {"items": {}, "nameMapping": {}}
        assert data["outputSchema"] == {"items": {}, "nameMapping": {}}


# ---------------------------------------------------------------------------
# GET /info/names
# ---------------------------------------------------------------------------


class TestGetInfoNames:
    """Tests for GET /info/names endpoint."""

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_names_empty_when_no_models(
        self,
        mock_get_ds: MagicMock,
        _mock_storage: MagicMock,
    ) -> None:
        """No models -> empty dict."""
        mock_get_ds.return_value = _make_mock_data_source(known_models=set())

        response = client.get("/info/names")

        assert response.status_code == 200  # noqa: PLR2004
        assert response.json() == {}

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_names_no_mapping_returns_empty_dicts(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """When original == aliased names, mappings are empty."""
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"mod1"},
        )
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.get_original_column_names = AsyncMock(return_value=["a", "b"])
        mock_storage.get_aliased_column_names = AsyncMock(return_value=["a", "b"])

        response = client.get("/info/names")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert body["mod1"]["modelId"] == "mod1"
        assert body["mod1"]["inputMapping"] == {}
        assert body["mod1"]["outputMapping"] == {}

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_names_with_input_and_output_mappings(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """Aliased columns show up as mappings in both input and output."""
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"mapped-model"},
        )
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        def orig_names(ds: str) -> list[str]:
            return ["feat1", "feat2"] if "inputs" in ds else ["pred"]

        def alias_names(ds: str) -> list[str]:
            return ["Feature 1", "feat2"] if "inputs" in ds else ["Prediction"]

        mock_storage.get_original_column_names = AsyncMock(side_effect=orig_names)
        mock_storage.get_aliased_column_names = AsyncMock(side_effect=alias_names)

        response = client.get("/info/names")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()["mapped-model"]
        assert body["inputMapping"] == {"feat1": "Feature 1"}
        assert body["outputMapping"] == {"pred": "Prediction"}

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_names_dataset_not_found(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """Missing datasets still return model entry with empty mappings."""
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"orphan"},
        )
        mock_storage.dataset_exists = AsyncMock(return_value=False)

        response = client.get("/info/names")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()["orphan"]
        assert body["inputMapping"] == {}
        assert body["outputMapping"] == {}

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_names_none_column_names_handled(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """None returned from column name methods yields empty mappings."""
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"none-cols"},
        )
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.get_original_column_names = AsyncMock(return_value=None)
        mock_storage.get_aliased_column_names = AsyncMock(return_value=None)

        response = client.get("/info/names")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()["none-cols"]
        assert body["inputMapping"] == {}
        assert body["outputMapping"] == {}

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_names_per_model_error_skipped(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """A per-model error is silently skipped; other models still returned."""
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"ok-model", "bad-model"},
        )

        async def exists_side_effect(ds: str) -> bool:
            if "bad-model" in ds:
                msg = "disk read error"
                raise RuntimeError(msg)
            return True

        mock_storage.dataset_exists = AsyncMock(side_effect=exists_side_effect)
        mock_storage.get_original_column_names = AsyncMock(return_value=["c"])
        mock_storage.get_aliased_column_names = AsyncMock(return_value=["c"])

        response = client.get("/info/names")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()
        assert "ok-model" in body
        # bad-model is skipped entirely due to the per-model catch
        assert "bad-model" not in body

    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_names_data_source_failure_returns_500(
        self,
        mock_get_ds: MagicMock,
    ) -> None:
        """Complete data source failure yields HTTP 500."""
        mock_get_ds.return_value.get_known_models = AsyncMock(
            side_effect=RuntimeError("DB down"),
        )

        response = client.get("/info/names")

        assert response.status_code == 500  # noqa: PLR2004
        assert "Error retrieving name mappings" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    @patch("trustyai_service.endpoints.metadata.get_data_source")
    def test_names_input_error_does_not_block_output(
        self,
        mock_get_ds: MagicMock,
        mock_storage: MagicMock,
    ) -> None:
        """Error reading input names does not prevent output names from appearing."""
        mock_get_ds.return_value = _make_mock_data_source(
            known_models={"partial"},
        )
        mock_storage.dataset_exists = AsyncMock(return_value=True)

        async def orig_side_effect(ds: str) -> list[str]:
            if "inputs" in ds:
                msg = "corrupt input data"
                raise RuntimeError(msg)
            return ["out_col"]

        async def alias_side_effect(ds: str) -> list[str]:
            if "inputs" in ds:
                msg = "corrupt input data"
                raise RuntimeError(msg)
            return ["Output Column"]

        mock_storage.get_original_column_names = AsyncMock(
            side_effect=orig_side_effect,
        )
        mock_storage.get_aliased_column_names = AsyncMock(
            side_effect=alias_side_effect,
        )

        response = client.get("/info/names")

        assert response.status_code == 200  # noqa: PLR2004
        body = response.json()["partial"]
        # Input mapping should be empty due to error
        assert body["inputMapping"] == {}
        # Output mapping should still work
        assert body["outputMapping"] == {"out_col": "Output Column"}


# ---------------------------------------------------------------------------
# POST /info/names
# ---------------------------------------------------------------------------


class TestPostInfoNames:
    """Tests for POST /info/names endpoint."""

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_apply_name_mapping_success(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Applying name mappings returns success message."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.apply_name_mapping = AsyncMock()

        payload = {
            "modelId": "my-model",
            "inputMapping": {"col_a": "Column A"},
            "outputMapping": {"pred": "Prediction"},
        }

        response = client.post("/info/names", json=payload)

        assert response.status_code == 200  # noqa: PLR2004
        assert "successfully applied" in response.json()["message"]
        assert mock_storage.apply_name_mapping.await_count == 2  # noqa: PLR2004

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_apply_name_mapping_input_only(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Mapping with only inputMapping calls apply only for input dataset."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.apply_name_mapping = AsyncMock()

        payload = {
            "modelId": "my-model",
            "inputMapping": {"col_a": "Column A"},
        }

        response = client.post("/info/names", json=payload)

        assert response.status_code == 200  # noqa: PLR2004
        # Only input mapping called (output mapping is empty)
        mock_storage.apply_name_mapping.assert_awaited_once_with(
            "my-model_inputs",
            {"col_a": "Column A"},
        )

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_apply_name_mapping_output_only(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Mapping with only outputMapping calls apply only for output dataset."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.apply_name_mapping = AsyncMock()

        payload = {
            "modelId": "my-model",
            "outputMapping": {"pred": "Prediction"},
        }

        response = client.post("/info/names", json=payload)

        assert response.status_code == 200  # noqa: PLR2004
        mock_storage.apply_name_mapping.assert_awaited_once_with(
            "my-model_outputs",
            {"pred": "Prediction"},
        )

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_apply_name_mapping_unknown_model(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Nonexistent model datasets returns 400."""
        mock_storage.dataset_exists = AsyncMock(return_value=False)

        payload = {
            "modelId": "unknown",
            "inputMapping": {"col": "Column"},
        }

        response = client.post("/info/names", json=payload)

        assert response.status_code == 400  # noqa: PLR2004
        assert "No metadata found" in response.json()["detail"]

    def test_apply_name_mapping_missing_model_id(self) -> None:
        """Missing modelId returns 422 validation error."""
        payload = {"inputMapping": {"col": "Column"}}

        response = client.post("/info/names", json=payload)

        assert response.status_code == 422  # noqa: PLR2004

    def test_apply_name_mapping_empty_body(self) -> None:
        """Empty body returns 422 validation error."""
        response = client.post("/info/names", json={})

        assert response.status_code == 422  # noqa: PLR2004

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_apply_name_mapping_empty_mappings_still_succeeds(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Empty inputMapping and outputMapping still returns success."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.apply_name_mapping = AsyncMock()

        payload = {
            "modelId": "my-model",
            "inputMapping": {},
            "outputMapping": {},
        }

        response = client.post("/info/names", json=payload)

        assert response.status_code == 200  # noqa: PLR2004
        # No apply calls since mappings are empty
        mock_storage.apply_name_mapping.assert_not_awaited()

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_apply_name_mapping_storage_error_returns_500(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Storage failure during apply returns 500."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.apply_name_mapping = AsyncMock(
            side_effect=RuntimeError("write failed"),
        )

        payload = {
            "modelId": "my-model",
            "inputMapping": {"c": "C"},
        }

        response = client.post("/info/names", json=payload)

        assert response.status_code == 500  # noqa: PLR2004
        assert "Error applying column names" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_apply_partial_existence_input_only(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Only input dataset exists; output mapping provided but skipped."""

        async def exists_side_effect(ds: str) -> bool:
            return "inputs" in ds

        mock_storage.dataset_exists = AsyncMock(side_effect=exists_side_effect)
        mock_storage.apply_name_mapping = AsyncMock()

        payload = {
            "modelId": "partial-model",
            "inputMapping": {"c": "C"},
            "outputMapping": {"p": "P"},
        }

        response = client.post("/info/names", json=payload)

        assert response.status_code == 200  # noqa: PLR2004
        # Only input mapping applied since output dataset does not exist
        mock_storage.apply_name_mapping.assert_awaited_once_with(
            "partial-model_inputs",
            {"c": "C"},
        )


# ---------------------------------------------------------------------------
# DELETE /info/names
# ---------------------------------------------------------------------------


class TestDeleteInfoNames:
    """Tests for DELETE /info/names endpoint."""

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_remove_name_mapping_success(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Clearing name mappings returns success message."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.clear_name_mapping = AsyncMock()

        response = client.request(
            "DELETE",
            "/info/names",
            json={"modelId": "my-model"},
        )

        assert response.status_code == 200  # noqa: PLR2004
        assert "successfully cleared" in response.json()["message"]
        assert mock_storage.clear_name_mapping.await_count == 2  # noqa: PLR2004

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_remove_name_mapping_unknown_model(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Nonexistent model datasets returns 400."""
        mock_storage.dataset_exists = AsyncMock(return_value=False)

        response = client.request(
            "DELETE",
            "/info/names",
            json={"modelId": "ghost"},
        )

        assert response.status_code == 400  # noqa: PLR2004
        assert "No metadata found" in response.json()["detail"]

    def test_remove_name_mapping_missing_model_id(self) -> None:
        """Missing modelId returns 422 validation error."""
        response = client.request("DELETE", "/info/names", json={})

        assert response.status_code == 422  # noqa: PLR2004

    def test_remove_name_mapping_no_body(self) -> None:
        """No body at all returns 422 validation error."""
        response = client.request("DELETE", "/info/names")

        assert response.status_code == 422  # noqa: PLR2004

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_remove_name_mapping_storage_error_returns_500(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Storage failure during clear returns 500."""
        mock_storage.dataset_exists = AsyncMock(return_value=True)
        mock_storage.clear_name_mapping = AsyncMock(
            side_effect=RuntimeError("delete failed"),
        )

        response = client.request(
            "DELETE",
            "/info/names",
            json={"modelId": "my-model"},
        )

        assert response.status_code == 500  # noqa: PLR2004
        assert "Error removing column names" in response.json()["detail"]

    @patch("trustyai_service.endpoints.metadata.storage_interface")
    def test_remove_clears_only_existing_datasets(
        self,
        mock_storage: MagicMock,
    ) -> None:
        """Only calls clear on datasets that actually exist."""

        async def exists_side_effect(ds: str) -> bool:
            return "outputs" in ds

        mock_storage.dataset_exists = AsyncMock(side_effect=exists_side_effect)
        mock_storage.clear_name_mapping = AsyncMock()

        response = client.request(
            "DELETE",
            "/info/names",
            json={"modelId": "half-model"},
        )

        assert response.status_code == 200  # noqa: PLR2004
        # Only output clear called since input dataset does not exist
        mock_storage.clear_name_mapping.assert_awaited_once_with(
            "half-model_outputs",
        )


# ---------------------------------------------------------------------------
# GET /info/tags and POST /info/tags (not-implemented stubs)
# ---------------------------------------------------------------------------


class TestTagEndpoints:
    """Tests for tag endpoints (currently return 501 Not Implemented)."""

    def test_get_tags_returns_501(self) -> None:
        """GET /info/tags returns 501 Not Implemented."""
        response = client.get("/info/tags")

        assert response.status_code == 501  # noqa: PLR2004
        assert "not yet implemented" in response.json()["detail"]

    def test_post_tags_returns_501(self) -> None:
        """POST /info/tags returns 501 Not Implemented."""
        payload = {
            "modelId": "any-model",
            "dataTagging": {"tag1": [[0, 5]]},
        }

        response = client.post("/info/tags", json=payload)

        assert response.status_code == 501  # noqa: PLR2004
        assert "not yet implemented" in response.json()["detail"]

    def test_post_tags_missing_model_id(self) -> None:
        """POST /info/tags without modelId returns 422."""
        response = client.post("/info/tags", json={})

        assert response.status_code == 422  # noqa: PLR2004
