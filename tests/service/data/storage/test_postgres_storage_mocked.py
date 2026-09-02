"""Tests for PostgreSQL storage backend using mocked database connections.

Covers: connection manager, SSL/TLS configuration, schema metadata operations,
get_known_models, get_metadata, error handling, and the require_existing_dataset
decorator. These tests do NOT require a running PostgreSQL instance.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Coroutine

psycopg = pytest.importorskip("psycopg")

from psycopg.types.json import Jsonb  # noqa: E402

from trustyai_service.service.data.modelmesh_parser import PartialPayload  # noqa: E402
from trustyai_service.service.data.storage.exceptions import (  # noqa: E402
    DeserializationError,
)
from trustyai_service.service.data.storage.postgres.utils import (  # noqa: E402
    PostgresConnectionManager,
    get_clean_column_names,
    require_existing_dataset,
)
from trustyai_service.service.serialization import serialize_model  # noqa: E402
from trustyai_service.service.serialization.encoders import json_encoder  # noqa: E402


def _make_mocked_storage() -> Any:  # noqa: ANN401 -- returns mocked PostgreSQLStorage
    """Create a PostgreSQLStorage with mocked __init__ to avoid a DB connection."""
    with patch(
        "trustyai_service.service.data.storage.postgres.postgres.PostgreSQLStorage.__init__",
        return_value=None,
    ):
        from trustyai_service.service.data.storage.postgres.postgres import (  # noqa: PLC0415
            PostgreSQLStorage,
        )

        storage = PostgreSQLStorage.__new__(PostgreSQLStorage)
        storage.__init__ = MagicMock()  # type: ignore[method-assign]
        storage.schema_prefix = "trustyai_v2"
        storage.dataset_reference_table = "trustyai_v2_table_reference"
        storage.partial_payload_table = "trustyai_v2_partial_payloads"
        storage.connection_manager = MagicMock()
        return storage


def _wire_connection(
    storage: Any,  # noqa: ANN401 -- storage is a mocked PostgreSQLStorage
    mock_conn: MagicMock,
    mock_cursor: MagicMock,
) -> None:
    """Wire a mocked (conn, cursor) pair into storage.connection_manager's context protocol."""
    storage.connection_manager.__enter__ = MagicMock(
        return_value=(mock_conn, mock_cursor)
    )
    storage.connection_manager.__exit__ = MagicMock(return_value=False)


def _run(coro: Coroutine[Any, Any, Any]) -> Any:  # noqa: ANN401 -- generic test runner
    """Run an async coroutine synchronously for tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# PostgresConnectionManager
# ===========================================================================


class TestPostgresConnectionManager:
    """Tests for PostgresConnectionManager context manager."""

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connect_without_ssl(self, mock_connect: MagicMock) -> None:
        """Connection without SSL does not include sslmode or sslrootcert."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="user",
            password="pass",  # noqa: S106  # pragma: allowlist secret
            host="localhost",
            port=5432,
            database="testdb",
            ssl_ca=None,
        )
        with mgr as (conn, _cursor):
            assert conn is mock_conn

        call_kwargs = mock_connect.call_args[1]
        assert "sslmode" not in call_kwargs
        assert "sslrootcert" not in call_kwargs

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connect_with_ssl(self, mock_connect: MagicMock) -> None:
        """Connection with SSL includes sslmode=verify-full and sslrootcert."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="user",
            password="pass",  # noqa: S106  # pragma: allowlist secret
            host="localhost",
            port=5432,
            database="testdb",
            ssl_ca="/path/to/ca.crt",
        )
        with mgr as (conn, _cursor):
            assert conn is mock_conn

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["sslmode"] == "verify-full"
        assert call_kwargs["sslrootcert"] == "/path/to/ca.crt"

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connect_with_timeout(self, mock_connect: MagicMock) -> None:
        """connect_timeout is passed through to psycopg.connect when provided."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=5432,
            database="d",
            connect_timeout=2,
        )
        with mgr:
            pass

        call_kwargs = mock_connect.call_args[1]
        assert call_kwargs["connect_timeout"] == 2  # noqa: PLR2004 -- exact timeout value under test

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connect_without_timeout_omits_kwarg(self, mock_connect: MagicMock) -> None:
        """connect_timeout is omitted from connect kwargs when not provided."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=5432,
            database="d",
        )
        with mgr:
            pass

        call_kwargs = mock_connect.call_args[1]
        assert "connect_timeout" not in call_kwargs

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connection_closed_on_exit(self, mock_connect: MagicMock) -> None:
        """Connection is closed when exiting the context manager."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=5432,
            database="d",
        )
        with mgr:
            pass

        mock_conn.close.assert_called_once()

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_connection_closed_on_exception(self, mock_connect: MagicMock) -> None:
        """Connection is closed even when an exception occurs inside the context."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = PostgresConnectionManager(
            user="u",
            password="p",  # noqa: S106 -- test credential
            host="h",
            port=5432,
            database="d",
        )
        msg = "boom"
        with pytest.raises(RuntimeError, match=msg), mgr:
            raise RuntimeError(msg)

        mock_conn.close.assert_called_once()


# ===========================================================================
# get_clean_column_names
# ===========================================================================


class TestGetCleanColumnNames:
    """Tests for get_clean_column_names utility."""

    def test_generates_safe_names(self) -> None:
        """Generates column_0, column_1, ... regardless of input names."""
        result = get_clean_column_names(["'; DROP TABLE--", "normal", "x"])
        assert result == ["column_0", "column_1", "column_2"]

    def test_empty_list(self) -> None:
        """Returns empty list for empty input."""
        assert get_clean_column_names([]) == []


# ===========================================================================
# require_existing_dataset decorator
# ===========================================================================


class TestRequireExistingDataset:
    """Tests for the require_existing_dataset decorator."""

    def test_raises_for_nonexistent_dataset(self) -> None:
        """Decorated function raises ValueError if dataset does not exist."""

        class _FakeStorage:
            async def dataset_exists(self, _name: str) -> bool:
                return False

        @require_existing_dataset
        async def dummy_func(storage: Any, dataset_name: str) -> str:  # noqa: ANN401, ARG001 -- test mock
            return "should not reach"

        with pytest.raises(ValueError, match="does not exist"):
            _run(dummy_func(_FakeStorage(), "ghost_dataset"))

    def test_passes_for_existing_dataset(self) -> None:
        """Decorated function executes normally if dataset exists."""

        class _FakeStorage:
            async def dataset_exists(self, _name: str) -> bool:
                return True

        @require_existing_dataset
        async def dummy_func(storage: Any, dataset_name: str) -> str:  # noqa: ANN401, ARG001 -- test mock
            return "success"

        result = _run(dummy_func(_FakeStorage(), "real_dataset"))
        assert result == "success"


# ===========================================================================
# PostgreSQLStorage with mocked connections
# ===========================================================================


class TestPostgreSQLStorageMocked:
    """Tests for PostgreSQLStorage methods using mocked database connections."""

    def _make_storage(self) -> Any:  # noqa: ANN401 -- returns mocked PostgreSQLStorage
        """Create a PostgreSQLStorage with mocked __init__ to avoid DB connection."""
        with patch(
            "trustyai_service.service.data.storage.postgres.postgres.PostgreSQLStorage.__init__",
            return_value=None,
        ):
            from trustyai_service.service.data.storage.postgres.postgres import (  # noqa: PLC0415
                PostgreSQLStorage,
            )

            storage = PostgreSQLStorage.__new__(PostgreSQLStorage)
            storage.__init__ = MagicMock()  # type: ignore[method-assign]
            storage.schema_prefix = "trustyai_v2"
            storage.dataset_reference_table = "trustyai_v2_table_reference"
            storage.partial_payload_table = "trustyai_v2_partial_payloads"
            storage.connection_manager = MagicMock()
            return storage

    def test_dataset_exists_returns_false_on_programming_error(self) -> None:
        """dataset_exists returns False when psycopg raises ProgrammingError."""
        storage = self._make_storage()

        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = psycopg.errors.ProgrammingError(
            "table missing"
        )
        mock_conn = MagicMock()
        storage.connection_manager.__enter__ = MagicMock(
            return_value=(mock_conn, mock_cursor)
        )
        storage.connection_manager.__exit__ = MagicMock(return_value=False)

        result = _run(storage.dataset_exists("some_dataset"))
        assert result is False

    def test_dataset_exists_returns_true(self) -> None:
        """dataset_exists returns True when a row is found."""
        storage = self._make_storage()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("some_dataset",)
        mock_conn = MagicMock()
        storage.connection_manager.__enter__ = MagicMock(
            return_value=(mock_conn, mock_cursor)
        )
        storage.connection_manager.__exit__ = MagicMock(return_value=False)

        result = _run(storage.dataset_exists("some_dataset"))
        assert result is True

    def test_dataset_exists_returns_false_when_not_found(self) -> None:
        """dataset_exists returns False when no matching row is found."""
        storage = self._make_storage()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        storage.connection_manager.__enter__ = MagicMock(
            return_value=(mock_conn, mock_cursor)
        )
        storage.connection_manager.__exit__ = MagicMock(return_value=False)

        result = _run(storage.dataset_exists("missing"))
        assert result is False

    def test_get_known_models_extracts_ids(self) -> None:
        """get_known_models extracts model IDs by stripping suffixes."""
        storage = self._make_storage()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("model_a_inputs",),
            ("model_a_outputs",),
            ("model_b_inputs",),
            ("model_b_metadata",),
            ("trustyai_internal_foo",),
        ]
        mock_conn = MagicMock()
        storage.connection_manager.__enter__ = MagicMock(
            return_value=(mock_conn, mock_cursor)
        )
        storage.connection_manager.__exit__ = MagicMock(return_value=False)

        models = sorted(_run(storage.get_known_models()))
        assert models == ["model_a", "model_b"]

    def test_get_known_models_empty(self) -> None:
        """get_known_models returns empty list when no datasets exist."""
        storage = self._make_storage()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        storage.connection_manager.__enter__ = MagicMock(
            return_value=(mock_conn, mock_cursor)
        )
        storage.connection_manager.__exit__ = MagicMock(return_value=False)

        assert _run(storage.get_known_models()) == []

    def test_build_table_name(self) -> None:
        """_build_table_name produces the expected format."""
        storage = self._make_storage()
        assert storage._build_table_name(42) == "trustyai_v2_dataset_42"

    def test_list_all_datasets_sync(self) -> None:
        """_list_all_datasets_sync returns dataset names from DB."""
        storage = self._make_storage()

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [("ds_a",), ("ds_b",)]
        mock_conn = MagicMock()
        storage.connection_manager.__enter__ = MagicMock(
            return_value=(mock_conn, mock_cursor)
        )
        storage.connection_manager.__exit__ = MagicMock(return_value=False)

        result = storage._list_all_datasets_sync()
        assert result == ["ds_a", "ds_b"]

    def test_get_metadata_empty_model(self) -> None:
        """get_metadata returns dict with None values for a model with no data."""
        storage = self._make_storage()

        # Mock dataset_exists to return False for all
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        storage.connection_manager.__enter__ = MagicMock(
            return_value=(mock_conn, mock_cursor)
        )
        storage.connection_manager.__exit__ = MagicMock(return_value=False)

        result = _run(storage.get_metadata("empty_model"))
        assert result["modelId"] == "empty_model"
        assert result["inputData"] is None
        assert result["outputData"] is None
        assert result["metadataData"] is None

    def test_persist_modelmesh_delegates(self) -> None:
        """persist_modelmesh_payload delegates to persist_partial_payload."""
        storage = self._make_storage()

        async def fake_persist(
            _payload: Any,  # noqa: ANN401
            _payload_id: str,
            *,
            is_input: bool,
        ) -> None:
            pass

        storage.persist_partial_payload = MagicMock(side_effect=fake_persist)

        payload = MagicMock()
        _run(storage.persist_modelmesh_payload(payload, "req-1", is_input=True))
        storage.persist_partial_payload.assert_called_once_with(
            payload, "req-1", is_input=True
        )

    def test_delete_modelmesh_delegates(self) -> None:
        """delete_modelmesh_payload delegates to delete_partial_payload."""
        storage = self._make_storage()

        async def fake_delete(
            _payload_id: str,
            *,
            is_input: bool,
        ) -> None:
            pass

        storage.delete_partial_payload = MagicMock(side_effect=fake_delete)

        _run(storage.delete_modelmesh_payload("req-2", is_input=False))
        storage.delete_partial_payload.assert_called_once_with("req-2", is_input=False)


# ===========================================================================
# Constructor DDL
# ===========================================================================


class TestPostgreSQLStorageConstructorDDL:
    """Tests for PostgreSQLStorage.__init__ schema-creation DDL."""

    @patch("trustyai_service.service.data.storage.postgres.utils.psycopg.connect")
    def test_creates_metadata_tables_with_correct_ddl(
        self, mock_connect: MagicMock
    ) -> None:
        """Constructor issues the correct dialect-specific DDL and commits once."""
        from trustyai_service.service.data.storage.postgres.postgres import (  # noqa: PLC0415
            PostgreSQLStorage,
        )

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        storage = PostgreSQLStorage(
            "user",
            "pass",  # pragma: allowlist secret
            "host",
            5432,
            "db",
        )

        assert mock_cursor.execute.call_count == 2  # noqa: PLR2004 -- exactly 2 DDL statements
        ddl_calls = [call.args[0] for call in mock_cursor.execute.call_args_list]

        table_ref_ddl = ddl_calls[0].as_string(None)
        assert table_ref_ddl == (
            'CREATE TABLE IF NOT EXISTS "trustyai_v2_table_reference" '
            "(table_idx BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
            "dataset_name varchar(255), metadata JSONB, n_rows BIGINT)"
        )

        partial_ddl = ddl_calls[1].as_string(None)
        assert partial_ddl == (
            'CREATE TABLE IF NOT EXISTS "trustyai_v2_partial_payloads" '
            "(payload_id varchar(255), is_input BOOLEAN, payload_data BYTEA)"
        )

        mock_conn.commit.assert_called_once()
        assert storage.schema_prefix == "trustyai_v2"
        assert storage.dataset_reference_table == "trustyai_v2_table_reference"
        assert storage.partial_payload_table == "trustyai_v2_partial_payloads"


# ===========================================================================
# write_data
# ===========================================================================


class TestWriteDataValidation:
    """Tests for write_data() input validation that happens before any DB access."""

    def test_empty_input_raises(self) -> None:
        """Empty new_rows raises ValueError before touching the database."""
        storage = _make_mocked_storage()
        with pytest.raises(ValueError, match="No data provided"):
            _run(storage.write_data("ds", np.array([]), []))

    def test_column_count_mismatch_raises(self) -> None:
        """A mismatch between array columns and column_names raises before any DB access."""
        storage = _make_mocked_storage()
        new_rows = np.array([[1, 2, 3]])
        with pytest.raises(ValueError, match="Shape mismatch"):
            _run(storage.write_data("ds", new_rows, ["only_one_name"]))


class TestWriteDataNewDataset:
    """Tests for write_data() when creating a dataset for the first time."""

    def test_full_flow_golden(self) -> None:
        """First write creates the metadata row, the BYTEA/IDENTITY table, and inserts rows."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=False)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5,)  # DB-assigned table_idx
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        new_rows = np.array([[1, 2], [3, 4]])
        column_names = ["colA", "colB"]

        _run(storage.write_data("my_dataset", new_rows, column_names))

        execute_calls = mock_cursor.execute.call_args_list
        assert len(execute_calls) == 4  # noqa: PLR2004 -- INSERT, SELECT, CREATE, UPDATE

        insert_query, insert_params = execute_calls[0].args
        assert insert_query.as_string(None) == (
            'INSERT INTO "trustyai_v2_table_reference" (dataset_name, metadata, n_rows) '
            "VALUES (%s, %s, 0)"
        )
        assert insert_params[0] == "my_dataset"
        assert isinstance(insert_params[1], Jsonb)
        assert insert_params[1].obj == {
            "column_names": column_names,
            "aliased_names": column_names,
            "shape": (-1, 2),
        }

        select_query, select_params = execute_calls[1].args
        assert select_query.as_string(None) == (
            'SELECT table_idx FROM "trustyai_v2_table_reference" WHERE dataset_name=%s'
        )
        assert select_params == ("my_dataset",)

        create_query = execute_calls[2].args[0]
        assert create_query.as_string(None) == (
            'CREATE TABLE IF NOT EXISTS "trustyai_v2_dataset_5" '
            "(row_idx BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
            '"column_0" BYTEA, "column_1" BYTEA)'
        )

        executemany_query, byte_matrix = mock_cursor.executemany.call_args.args
        assert executemany_query.as_string(None) == (
            'INSERT INTO "trustyai_v2_dataset_5" ("column_0", "column_1") VALUES (%s, %s)'
        )
        for row_idx, row in enumerate(byte_matrix):
            for col_idx, cell in enumerate(row):
                decompressed = gzip.decompress(cell).decode("utf-8")
                expected = json.dumps(new_rows[row_idx][col_idx], default=json_encoder)
                assert decompressed == expected

        update_query, update_params = execute_calls[3].args
        assert update_query.as_string(None) == (
            'UPDATE "trustyai_v2_table_reference" SET n_rows=%s WHERE dataset_name=%s'
        )
        assert update_params == (2, "my_dataset")

        assert mock_conn.commit.call_count == 2  # noqa: PLR2004 -- once per `with` block

    def test_single_row_reshape(self) -> None:
        """A 1-D new_rows array is reshaped into a single-column matrix."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=False)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        new_rows = np.array([10, 20, 30])
        _run(storage.write_data("single_col_ds", new_rows, ["only_col"]))

        create_query = mock_cursor.execute.call_args_list[2].args[0]
        assert create_query.as_string(None) == (
            'CREATE TABLE IF NOT EXISTS "trustyai_v2_dataset_1" '
            "(row_idx BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
            '"column_0" BYTEA)'
        )
        _, byte_matrix = mock_cursor.executemany.call_args.args
        assert len(byte_matrix) == 3  # noqa: PLR2004 -- three reshaped rows


class TestWriteDataAppend:
    """Tests for write_data() when appending to an already-existing dataset."""

    def test_append_to_existing_dataset(self) -> None:
        """Appending uses the existing table name and increments n_rows."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage.dataset_shape = AsyncMock(return_value=(-1, 2))
        storage.dataset_rows = AsyncMock(return_value=3)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_7")

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        new_rows = np.array([[10, 20]])
        _run(storage.write_data("ds", new_rows, ["a", "b"]))

        executemany_query, byte_matrix = mock_cursor.executemany.call_args.args
        assert executemany_query.as_string(None) == (
            'INSERT INTO "trustyai_v2_dataset_7" ("column_0", "column_1") VALUES (%s, %s)'
        )
        assert len(byte_matrix) == 1

        update_query, update_params = mock_cursor.execute.call_args.args
        assert update_query.as_string(None) == (
            'UPDATE "trustyai_v2_table_reference" SET n_rows=%s WHERE dataset_name=%s'
        )
        assert update_params == (4, "ds")
        mock_conn.commit.assert_called_once()

    def test_append_column_count_mismatch_raises(self) -> None:
        """Appending with a column count differing from the stored shape raises."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage.dataset_shape = AsyncMock(return_value=(-1, 5))
        storage.dataset_rows = AsyncMock(return_value=3)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_7")

        new_rows = np.array([[1, 2]])
        with pytest.raises(ValueError, match="existing database"):
            _run(storage.write_data("ds", new_rows, ["a", "b"]))

    def test_append_extra_dims_mismatch_raises(self) -> None:
        """Appending with mismatched higher-order dimensions raises."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage.dataset_shape = AsyncMock(return_value=(-1, 2, 3))
        storage.dataset_rows = AsyncMock(return_value=1)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_9")

        new_rows = np.zeros((1, 2, 4))
        with pytest.raises(ValueError, match="Shape mismatch"):
            _run(storage.write_data("ds", new_rows, ["a", "b"]))


# ===========================================================================
# read_data
# ===========================================================================


class TestReadData:
    """Tests for read_data() LIMIT/OFFSET paging and JSON+gzip decoding."""

    def test_limit_offset_query_golden(self) -> None:
        """read_data issues the correct dialect LIMIT/OFFSET SQL with bound params."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_3")

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        _run(storage.read_data("ds", start_row=10, n_rows=5))

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'SELECT * FROM "trustyai_v2_dataset_3" ORDER BY row_idx ASC LIMIT %s OFFSET %s'
        )
        assert params == (5, 10)

    def test_default_n_rows_uses_dataset_rows(self) -> None:
        """When n_rows is None, read_data uses dataset_rows() as the LIMIT."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_3")
        storage.dataset_rows = AsyncMock(return_value=42)

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        _run(storage.read_data("ds"))

        _, params = mock_cursor.execute.call_args.args
        assert params == (42, 0)

    def test_gzip_json_roundtrip(self) -> None:
        """Rows are decompressed and JSON-decoded back into a NumPy array."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_3")

        cell_a = gzip.compress(json.dumps(1.5).encode("utf-8"))
        cell_b = gzip.compress(json.dumps(2.5).encode("utf-8"))
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(0, cell_a, cell_b)]
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(storage.read_data("ds", n_rows=1))
        assert result.tolist() == [[1.5, 2.5]]

    def test_empty_result(self) -> None:
        """An empty result set produces an empty array without error."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_3")

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(storage.read_data("ds", n_rows=0))
        assert result.size == 0


# ===========================================================================
# Name mapping
# ===========================================================================


class TestNameMapping:
    """Golden SQL tests for apply_name_mapping() and clear_name_mapping()."""

    def test_apply_name_mapping_golden_sql(self) -> None:
        """apply_name_mapping issues the exact jsonb_set UPDATE with a Jsonb-bound param."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage.get_original_column_names = AsyncMock(return_value=["a", "b"])
        storage.get_aliased_column_names = AsyncMock(return_value=["a", "b"])

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        _run(storage.apply_name_mapping("ds", {"a": "alpha"}))

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'UPDATE "trustyai_v2_table_reference" '
            "SET metadata = jsonb_set(metadata, '{aliased_names}', %s::jsonb) "
            "WHERE dataset_name=%s"
        )
        assert isinstance(params[0], Jsonb)
        assert params[0].obj == ["alpha", "b"]
        assert params[1] == "ds"
        mock_conn.commit.assert_called_once()

    def test_clear_name_mapping_golden_sql(self) -> None:
        """clear_name_mapping resets aliased_names to column_names via the same jsonb_set path."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage.get_original_column_names = AsyncMock(return_value=["a", "b"])

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        _run(storage.clear_name_mapping("ds"))

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'UPDATE "trustyai_v2_table_reference" '
            "SET metadata = jsonb_set(metadata, '{aliased_names}', %s::jsonb) "
            "WHERE dataset_name=%s"
        )
        assert isinstance(params[0], Jsonb)
        assert params[0].obj == ["a", "b"]
        assert params[1] == "ds"
        mock_conn.commit.assert_called_once()


# ===========================================================================
# Column names
# ===========================================================================


class TestInternalHelpers:
    """Tests for the private _get_clean_table_name() and _get_dataset_metadata() helpers."""

    def test_get_clean_table_name_golden_sql(self) -> None:
        """_get_clean_table_name looks up table_idx and builds the dataset table name."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (12,)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(storage._get_clean_table_name("ds"))
        assert result == "trustyai_v2_dataset_12"

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'SELECT table_idx FROM "trustyai_v2_table_reference" WHERE dataset_name=%s'
        )
        assert params == ("ds",)

    def test_get_dataset_metadata_golden_sql(self) -> None:
        """_get_dataset_metadata queries the metadata column and returns the parsed dict."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ({"column_names": ["a"]},)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(storage._get_dataset_metadata("ds"))
        assert result == {"column_names": ["a"]}

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'SELECT metadata FROM "trustyai_v2_table_reference" WHERE dataset_name=%s'
        )
        assert params == ("ds",)


class TestColumnNames:
    """Tests for get_original_column_names() and get_aliased_column_names()."""

    def test_get_original_column_names(self) -> None:
        """Returns the column_names entry from dataset metadata."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage._get_dataset_metadata = AsyncMock(
            return_value={"column_names": ["a", "b"], "aliased_names": ["x", "y"]}
        )
        result = _run(storage.get_original_column_names("ds"))
        assert result == ["a", "b"]

    def test_get_aliased_column_names(self) -> None:
        """Returns the aliased_names entry from dataset metadata."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage._get_dataset_metadata = AsyncMock(
            return_value={"column_names": ["a", "b"], "aliased_names": ["x", "y"]}
        )
        result = _run(storage.get_aliased_column_names("ds"))
        assert result == ["x", "y"]


# ===========================================================================
# Dataset dimensions
# ===========================================================================


class TestDatasetDimensions:
    """Tests for dataset_rows(), dataset_cols(), and dataset_shape()."""

    def test_dataset_rows_query(self) -> None:
        """dataset_rows issues the expected SELECT and returns n_rows."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (17,)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(storage.dataset_rows("ds"))
        assert result == 17  # noqa: PLR2004 -- exact row count under test

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'SELECT n_rows FROM "trustyai_v2_table_reference" WHERE dataset_name=%s'
        )
        assert params == ("ds",)

    def test_dataset_cols_information_schema_golden(self) -> None:
        """dataset_cols queries information_schema.columns and subtracts the row_idx column."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_4")

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (6,)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(storage.dataset_cols("ds"))
        assert result == 5  # noqa: PLR2004 -- 6 total columns minus row_idx

        query, params = mock_cursor.execute.call_args.args
        assert query == (
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name=%s AND table_schema=current_schema()"
        )
        assert params == ("trustyai_v2_dataset_4",)

    def test_dataset_shape(self) -> None:
        """dataset_shape combines dataset_rows() with the stored metadata shape."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage.dataset_rows = AsyncMock(return_value=9)
        storage._get_dataset_metadata = AsyncMock(return_value={"shape": [-1, 3]})

        result = _run(storage.dataset_shape("ds"))
        assert result == (9, 3)


# ===========================================================================
# get_metadata (full path)
# ===========================================================================


class TestGetMetadataFullPath:
    """Tests for get_metadata() when input/output/metadata datasets all exist."""

    def test_full_metadata_all_datasets_present(self) -> None:
        """get_metadata assembles shapes and (aliased) column names for all three datasets."""
        storage = _make_mocked_storage()

        async def fake_exists(name: str) -> bool:
            return name in {"m_inputs", "m_outputs", "m_metadata"}

        async def fake_shape(name: str) -> tuple[int, ...]:
            shapes = {"m_inputs": (10, 3), "m_outputs": (10, 1), "m_metadata": (10, 2)}
            return shapes[name]

        async def fake_orig(name: str) -> list[str]:
            names = {
                "m_inputs": ["a", "b", "c"],
                "m_outputs": ["y"],
                "m_metadata": ["meta1", "meta2"],
            }
            return names[name]

        async def fake_alias(name: str) -> list[str]:
            names = {"m_inputs": ["A", "B", "C"], "m_outputs": ["Y"]}
            return names[name]

        storage.dataset_exists = fake_exists
        storage.dataset_shape = fake_shape
        storage.get_original_column_names = fake_orig
        storage.get_aliased_column_names = fake_alias

        result = _run(storage.get_metadata("m"))

        assert result["modelId"] == "m"
        assert result["inputData"] == {
            "shape": [10, 3],
            "columnNames": ["a", "b", "c"],
            "aliasedNames": ["A", "B", "C"],
        }
        assert result["outputData"] == {
            "shape": [10, 1],
            "columnNames": ["y"],
            "aliasedNames": ["Y"],
        }
        assert result["metadataData"] == {
            "shape": [10, 2],
            "columnNames": ["meta1", "meta2"],
        }

    def test_errors_in_each_section_are_logged_and_swallowed(self) -> None:
        """A failure fetching shape/column data for any dataset is caught and logged, not raised."""
        storage = _make_mocked_storage()

        async def fake_exists(name: str) -> bool:
            return name in {"m_inputs", "m_outputs", "m_metadata"}

        async def fake_shape(_name: str) -> tuple[int, ...]:
            msg = "boom"
            raise RuntimeError(msg)

        storage.dataset_exists = fake_exists
        storage.dataset_shape = fake_shape
        storage.get_original_column_names = AsyncMock(return_value=["a"])
        storage.get_aliased_column_names = AsyncMock(return_value=["a"])

        result = _run(storage.get_metadata("m"))

        assert result["modelId"] == "m"
        assert result["inputData"] is None
        assert result["outputData"] is None
        assert result["metadataData"] is None


# ===========================================================================
# Partial payloads
# ===========================================================================


class TestPartialPayloads:
    """Tests for persist/get/delete of partial payloads, including error paths."""

    def test_persist_partial_payload_golden_sql(self) -> None:
        """persist_partial_payload issues the correct INSERT and commits."""
        storage = _make_mocked_storage()
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        payload = PartialPayload(data="abc", metadata={})
        _run(storage.persist_partial_payload(payload, "req-1", is_input=True))

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'INSERT INTO "trustyai_v2_partial_payloads" '
            "(payload_id, is_input, payload_data) VALUES (%s, %s, %s)"
        )
        assert params[0] == "req-1"
        assert params[1] is True
        assert isinstance(params[2], bytes)
        mock_conn.commit.assert_called_once()

    def test_get_partial_payload_found(self) -> None:
        """get_partial_payload deserializes a found row back into a PartialPayload."""
        storage = _make_mocked_storage()
        payload = PartialPayload(data="hello", metadata={"k": "v"})
        serialized = serialize_model(payload)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (serialized,)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(
            storage.get_partial_payload("req-1", is_input=True, is_modelmesh=True)
        )
        assert isinstance(result, PartialPayload)
        assert result.data == "hello"
        assert result.metadata == {"k": "v"}

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'SELECT payload_data FROM "trustyai_v2_partial_payloads" '
            "WHERE payload_id=%s AND is_input=%s"
        )
        assert params == ("req-1", True)

    def test_get_partial_payload_kserve_input(self) -> None:
        """get_partial_payload deserializes into KServeInferenceRequest for non-ModelMesh input."""
        from trustyai_service.endpoints.consumer import (  # noqa: PLC0415
            KServeData,
            KServeInferenceRequest,
        )

        storage = _make_mocked_storage()
        payload = KServeInferenceRequest(
            inputs=[KServeData(name="x", shape=[1], datatype="INT32", data=[1])]
        )
        serialized = serialize_model(payload)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (serialized,)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(
            storage.get_partial_payload("req-2", is_input=True, is_modelmesh=False)
        )
        assert isinstance(result, KServeInferenceRequest)
        assert result.inputs[0].name == "x"

    def test_get_partial_payload_kserve_output(self) -> None:
        """get_partial_payload deserializes into KServeInferenceResponse for non-ModelMesh output."""
        from trustyai_service.endpoints.consumer import (  # noqa: PLC0415
            KServeData,
            KServeInferenceResponse,
        )

        storage = _make_mocked_storage()
        payload = KServeInferenceResponse(
            outputs=[KServeData(name="y", shape=[1], datatype="FP32", data=[1.0])]
        )
        serialized = serialize_model(payload)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (serialized,)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(
            storage.get_partial_payload("req-3", is_input=False, is_modelmesh=False)
        )
        assert isinstance(result, KServeInferenceResponse)
        assert result.outputs[0].name == "y"

    def test_get_partial_payload_not_found(self) -> None:
        """get_partial_payload returns None when no row is found."""
        storage = _make_mocked_storage()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(
            storage.get_partial_payload("missing", is_input=False, is_modelmesh=True)
        )
        assert result is None

    def test_get_partial_payload_deserialization_error(self) -> None:
        """Corrupted payload bytes raise DeserializationError, not a bare exception."""
        storage = _make_mocked_storage()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (b"not valid gzip or json data",)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        with pytest.raises(DeserializationError):
            _run(storage.get_partial_payload("req-x", is_input=True, is_modelmesh=True))

    def test_delete_partial_payload_golden_sql(self) -> None:
        """delete_partial_payload issues the correct DELETE and commits."""
        storage = _make_mocked_storage()
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        _run(storage.delete_partial_payload("req-1", is_input=False))

        query, params = mock_cursor.execute.call_args.args
        assert query.as_string(None) == (
            'DELETE FROM "trustyai_v2_partial_payloads" '
            "WHERE payload_id=%s AND is_input=%s"
        )
        assert params == ("req-1", False)
        mock_conn.commit.assert_called_once()

    def test_get_modelmesh_payload_get_path(self) -> None:
        """get_modelmesh_payload retrieves and deserializes a stored ModelMesh payload."""
        storage = _make_mocked_storage()
        payload = PartialPayload(data="mm-data", metadata={})
        serialized = serialize_model(payload)

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (serialized,)
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        result = _run(storage.get_modelmesh_payload("req-mm", is_input=True))
        assert isinstance(result, PartialPayload)
        assert result.data == "mm-data"


# ===========================================================================
# Database cleanup
# ===========================================================================


class TestDatabaseCleanup:
    """Tests for delete_dataset(), delete_all_datasets(), and reset_database()."""

    def test_delete_dataset_golden_sql(self) -> None:
        """delete_dataset removes the reference row and drops the data table."""
        storage = _make_mocked_storage()
        storage.dataset_exists = AsyncMock(return_value=True)
        storage._get_clean_table_name = AsyncMock(return_value="trustyai_v2_dataset_2")

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        _run(storage.delete_dataset("ds"))

        calls = mock_cursor.execute.call_args_list
        assert len(calls) == 2  # noqa: PLR2004 -- DELETE reference row, DROP data table

        delete_query, delete_params = calls[0].args
        assert delete_query.as_string(None) == (
            'DELETE FROM "trustyai_v2_table_reference" WHERE dataset_name=%s'
        )
        assert delete_params == ("ds",)

        drop_query = calls[1].args[0]
        assert drop_query.as_string(None) == (
            'DROP TABLE IF EXISTS "trustyai_v2_dataset_2"'
        )
        mock_conn.commit.assert_called_once()

    def test_delete_all_datasets(self) -> None:
        """delete_all_datasets deletes every dataset returned by list_all_datasets."""
        storage = _make_mocked_storage()
        storage.list_all_datasets = AsyncMock(return_value=["a", "b"])
        storage.delete_dataset = AsyncMock()

        _run(storage.delete_all_datasets())

        assert storage.delete_dataset.call_count == 2  # noqa: PLR2004 -- two datasets
        storage.delete_dataset.assert_any_call("a")
        storage.delete_dataset.assert_any_call("b")

    def test_reset_database_golden_sql(self) -> None:
        """reset_database deletes all datasets then drops both metadata tables."""
        storage = _make_mocked_storage()
        storage.delete_all_datasets = AsyncMock()

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        _wire_connection(storage, mock_conn, mock_cursor)

        _run(storage.reset_database())

        calls = mock_cursor.execute.call_args_list
        assert calls[0].args[0].as_string(None) == (
            'DROP TABLE IF EXISTS "trustyai_v2_table_reference"'
        )
        assert calls[1].args[0].as_string(None) == (
            'DROP TABLE IF EXISTS "trustyai_v2_partial_payloads"'
        )
        mock_conn.commit.assert_called_once()
        storage.delete_all_datasets.assert_called_once()
