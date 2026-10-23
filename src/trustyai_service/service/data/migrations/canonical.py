"""Adapter for the service's physical canonical v2 SQL storage contract.

The migration bookkeeping tables are deliberately kept in ``db.schema``. This
module writes only the service's public v2 layout so a service using the shared
SQL backend can read migrated data without a translation step.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import MetaData, Table, delete, insert, select, update
from sqlalchemy.schema import CreateTable

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.db.schema import create_schema
from trustyai_service.service.data.storage.sql.schema import (
    build_dataset_table,
    build_partial_payload_table,
    build_reference_table,
)
from trustyai_service.service.serialization import deserialize_model, serialize_model
from trustyai_service.service.serialization.detection import safe_gzip_decompress
from trustyai_service.service.serialization.encoders import (
    json_decoder_hook,
    json_encoder,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

_MIN_MATRIX_DIMENSIONS = 2
_MAX_DATASET_NAME_LENGTH = 255


@dataclass(frozen=True)
class CanonicalTables:
    """Static tables in the service's v2 storage layout."""

    metadata: MetaData
    reference: Table
    partial_payloads: Table

    @property
    def datasets(self) -> Table:
        """Compatibility alias for callers that inspect the reference table."""
        return self.reference


def _build_tables() -> CanonicalTables:
    """Build the static canonical v2 tables without creating them on the server."""
    metadata = MetaData()
    reference = build_reference_table(metadata, "trustyai_v2_table_reference")
    partial_payloads = build_partial_payload_table(
        metadata, "trustyai_v2_partial_payloads"
    )
    return CanonicalTables(metadata, reference, partial_payloads)


def _build_dataset_table(table_idx: int, column_count: int) -> Table:
    """Build one dynamic canonical dataset table from a database-generated index."""
    if table_idx < 0 or column_count <= 0:
        msg = "canonical dataset table indexes and column counts must be positive"
        raise ValueError(msg)
    metadata = MetaData()
    return build_dataset_table(
        metadata,
        f"trustyai_v2_dataset_{table_idx}",
        [f"column_{index}" for index in range(column_count)],
    )


class CanonicalStorage:
    """Async SQLAlchemy adapter for the service's canonical v2 schema."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize a target adapter using the shared SQL schema builders."""
        self.engine = engine
        self.tables = _build_tables()
        self.reference_table = self.tables.reference
        self.partial_payload_table = self.tables.partial_payloads
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create only the static service tables."""
        await self._ensure_initialized()

    async def _ensure_initialized(self) -> None:
        """Create static tables once per target object, tolerating DDL races."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await create_schema(self.tables.metadata, self.engine)
            self._initialized = True

    async def reset_for_test(self) -> None:
        """Remove all canonical data; callers must use this only for test setup."""
        await self._ensure_initialized()
        datasets = await self._list_references()
        for table_idx, _ in datasets:
            table = _build_dataset_table(table_idx, await self._column_count(table_idx))
            async with self.engine.begin() as connection:
                await connection.run_sync(table.drop, checkfirst=True)
        async with self.engine.begin() as connection:
            await connection.execute(delete(self.tables.reference))
            await connection.execute(delete(self.tables.partial_payloads))
        self._initialized = False

    async def ensure_dataset(  # noqa: PLR0913, PLR0912, PLR0915, C901 -- one public metadata boundary
        self,
        dataset_name: str,
        column_names: Sequence[str],
        *,
        aliases: Sequence[str] | None = None,
        column_types: Sequence[str] | None = None,
        shape: Sequence[int] | None = None,
        serialized: bool | None = None,
        service_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create or validate a dataset registry row and dynamic table."""
        _validate_dataset_name(dataset_name)
        names = [str(name) for name in column_names]
        if not names:
            msg = f"Dataset '{dataset_name}' must contain at least one column"
            raise ValueError(msg)
        alias_names = list(aliases) if aliases is not None else names
        if len(alias_names) != len(names):
            msg = f"Dataset '{dataset_name}' has mismatched aliases"
            raise ValueError(msg)
        types = list(column_types) if column_types is not None else None
        if types is not None and len(types) != len(names):
            msg = f"Dataset '{dataset_name}' has mismatched column types"
            raise ValueError(msg)
        trailing_shape = tuple(int(value) for value in (shape or (len(names),)))
        if not trailing_shape or trailing_shape[0] != len(names):
            msg = f"Dataset '{dataset_name}' has an incompatible shape"
            raise ValueError(msg)
        if any(value <= 0 for value in trailing_shape):
            msg = f"Dataset '{dataset_name}' has non-positive dimensions"
            raise ValueError(msg)

        metadata: dict[str, Any] = {
            "column_names": names,
            "aliased_names": alias_names,
            "shape": [-1, *trailing_shape],
        }
        if types is not None:
            metadata["column_types"] = types
        if serialized is not None:
            metadata["serialized"] = bool(serialized)
        if service_metadata:
            metadata["trustyai_service"] = dict(service_metadata)

        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            existing = await self._get_reference_in_connection(connection, dataset_name)
            if existing is None:
                result = await connection.execute(
                    self.tables.reference.insert().values(
                        dataset_name=dataset_name,
                        metadata=metadata,
                        n_rows=0,
                    )
                )
                table_idx = int(result.inserted_primary_key[0])
                table = _build_dataset_table(table_idx, len(names))
                await connection.run_sync(table.create, checkfirst=True)
                return

            existing_metadata = dict(existing["metadata"])
            if existing_metadata.get("column_names") != names:
                msg = f"Dataset '{dataset_name}' has incompatible column names"
                raise ValueError(msg)
            if tuple(existing_metadata.get("shape", [])[1:]) != trailing_shape:
                msg = f"Dataset '{dataset_name}' has incompatible shape"
                raise ValueError(msg)
            existing_aliases = existing_metadata.get("aliased_names", names)
            if aliases is not None and existing_aliases != alias_names:
                msg = f"Dataset '{dataset_name}' has incompatible aliases"
                raise ValueError(msg)
            existing_types = existing_metadata.get("column_types")
            if types is not None and existing_types not in (None, types):
                msg = f"Dataset '{dataset_name}' has incompatible column types"
                raise ValueError(msg)
            existing_service_metadata = existing_metadata.get("trustyai_service")
            if service_metadata and existing_service_metadata not in (
                None,
                service_metadata,
            ):
                msg = f"Dataset '{dataset_name}' has incompatible service metadata"
                raise ValueError(msg)
            table = _build_dataset_table(
                int(existing["table_idx"]),
                len(existing_metadata["column_names"]),
            )
            await connection.run_sync(
                lambda sync_connection: sync_connection.execute(
                    CreateTable(table, if_not_exists=True)
                )
            )
            merged = dict(existing_metadata)
            if aliases is not None:
                merged["aliased_names"] = alias_names
            if types is not None:
                merged["column_types"] = types
            if serialized is not None:
                merged["serialized"] = bool(serialized)
            if service_metadata:
                merged["trustyai_service"] = dict(service_metadata)
            await connection.execute(
                self.tables.reference.update()
                .where(self.tables.reference.c.dataset_name == dataset_name)
                .values(metadata=merged)
            )

    async def write_data(
        self, dataset_name: str, new_rows: np.ndarray, column_names: list[str]
    ) -> None:
        """Append rows using the canonical per-column blob representation."""
        rows = _normalize_rows(new_rows)
        if not await self.dataset_exists(dataset_name):
            await self.ensure_dataset(
                dataset_name,
                column_names,
                shape=rows.shape[1:],
            )
        await self.append_data(dataset_name, rows, column_names)

    async def append_data(
        self,
        dataset_name: str,
        rows: np.ndarray,
        column_names: Sequence[str],
    ) -> None:
        """Append one non-empty migration batch to an existing dataset."""
        normalized = _normalize_rows(rows)
        if len(normalized) == 0:
            msg = "No data provided for a canonical dataset write"
            raise ValueError(msg)
        reference = await self._get_reference(dataset_name)
        metadata = dict(reference["metadata"])
        stored_shape = tuple(int(value) for value in metadata["shape"][1:])
        if list(column_names) != metadata["column_names"]:
            msg = f"Dataset '{dataset_name}' has incompatible column names"
            raise ValueError(msg)
        if tuple(normalized.shape[1:]) != stored_shape:
            msg = (
                f"Dataset '{dataset_name}' has incompatible shape: "
                f"expected {stored_shape}, got {normalized.shape[1:]}"
            )
            raise ValueError(msg)

        table = _build_dataset_table(
            int(reference["table_idx"]), len(metadata["column_names"])
        )
        parameters = [
            {
                f"column_{index}": gzip.compress(
                    json.dumps(value, default=json_encoder).encode("utf-8"),
                    compresslevel=1,
                )
                for index, value in enumerate(row)
            }
            for row in normalized
        ]
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.execute(insert(table), parameters)
            await connection.execute(
                self.tables.reference.update()
                .where(self.tables.reference.c.dataset_name == dataset_name)
                .values(n_rows=self.tables.reference.c.n_rows + len(parameters))
            )

    async def dataset_exists(self, dataset_name: str) -> bool:
        """Return whether a canonical dataset is registered."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.reference.c.dataset_name).where(
                    self.tables.reference.c.dataset_name == dataset_name
                )
            )
            return result.first() is not None

    async def list_all_datasets(self) -> list[str]:
        """Return canonical datasets in stable order."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.reference.c.dataset_name).order_by(
                    self.tables.reference.c.dataset_name
                )
            )
            return list(result.scalars())

    async def dataset_rows(self, dataset_name: str) -> int:
        """Return the canonical registry row count."""
        reference = await self._get_reference(dataset_name)
        return int(reference["n_rows"])

    async def dataset_shape(self, dataset_name: str) -> tuple[int, ...]:
        """Return the actual row count followed by stored dimensions."""
        reference = await self._get_reference(dataset_name)
        shape = list(reference["metadata"]["shape"])
        shape[0] = int(reference["n_rows"])
        return tuple(int(value) for value in shape)

    async def read_data(
        self,
        dataset_name: str,
        start_row: int | None = None,
        n_rows: int | None = None,
    ) -> np.ndarray:
        """Read and decode canonical per-column gzip/JSON cells."""
        start = 0 if start_row is None else start_row
        if start < 0 or (n_rows is not None and n_rows < 0):
            msg = "row offsets and limits must be non-negative"
            raise ValueError(msg)
        reference = await self._get_reference(dataset_name)
        metadata = dict(reference["metadata"])
        column_count = len(metadata["column_names"])
        shape = tuple(int(value) for value in metadata["shape"][1:])
        table = _build_dataset_table(int(reference["table_idx"]), column_count)
        query = select(table).order_by(table.c.row_idx.asc()).offset(start)
        if n_rows is not None:
            query = query.limit(n_rows)
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(query)
            rows: list[list[Any]] = []
            dtypes: set[np.dtype[Any]] = set()
            has_object_cells = False
            for row in result:
                values: list[Any] = []
                for cell in row[1:]:
                    decoded = json.loads(
                        safe_gzip_decompress(bytes(cell)).decode("utf-8"),
                        object_hook=json_decoder_hook,
                    )
                    value = np.asarray(decoded)
                    dtypes.add(value.dtype)
                    if value.ndim == 0:
                        scalar = value.item()
                        has_object_cells |= isinstance(scalar, (bytes, dict)) or (
                            scalar is None
                        )
                        values.append(scalar)
                    else:
                        has_object_cells |= len(shape) == 1
                        values.append(decoded)
                rows.append(values)
        if rows:
            dtype = object if has_object_cells or len(dtypes) != 1 else dtypes.pop()
            return _decoded_rows_array(rows, shape, dtype)
        return np.empty((0, *shape), dtype=object)

    async def get_original_column_names(self, dataset_name: str) -> list[str]:
        """Return original logical column names."""
        reference = await self._get_reference(dataset_name)
        return list(reference["metadata"]["column_names"])

    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        """Return current logical aliases."""
        reference = await self._get_reference(dataset_name)
        metadata = reference["metadata"]
        return list(metadata.get("aliased_names", metadata["column_names"]))

    async def get_column_types(self, dataset_name: str) -> list[str] | None:
        """Return preserved source types when the migration supplied them."""
        reference = await self._get_reference(dataset_name)
        types = reference["metadata"].get("column_types")
        return list(types) if types is not None else None

    async def get_service_metadata(self, dataset_name: str) -> dict[str, Any] | None:
        """Return preserved service-level metadata for a migrated dataset."""
        reference = await self._get_reference(dataset_name)
        value = reference["metadata"].get("trustyai_service")
        return dict(value) if value is not None else None

    async def set_column_types(
        self, dataset_name: str, column_types: list[str]
    ) -> None:
        """Preserve type labels as optional migration metadata."""
        reference = await self._get_reference(dataset_name)
        metadata = dict(reference["metadata"])
        if len(column_types) != len(metadata["column_names"]):
            msg = f"Dataset '{dataset_name}' has mismatched column types"
            raise ValueError(msg)
        metadata["column_types"] = list(column_types)
        await self._update_metadata(dataset_name, metadata)

    async def apply_name_mapping(
        self, dataset_name: str, name_mapping: dict[str, str]
    ) -> None:
        """Update aliases while retaining original names."""
        names = await self.get_original_column_names(dataset_name)
        await self._update_metadata(
            dataset_name,
            {
                **(await self._get_reference(dataset_name))["metadata"],
                "aliased_names": [name_mapping.get(name, name) for name in names],
            },
        )

    async def clear_name_mapping(self, dataset_name: str) -> None:
        """Reset aliases to original names."""
        names = await self.get_original_column_names(dataset_name)
        metadata = dict((await self._get_reference(dataset_name))["metadata"])
        metadata["aliased_names"] = names
        await self._update_metadata(dataset_name, metadata)

    async def delete_dataset(self, dataset_name: str) -> None:
        """Delete a dataset reference and its dynamic table."""
        reference = await self._get_reference(dataset_name)
        table = _build_dataset_table(
            int(reference["table_idx"]),
            len(reference["metadata"]["column_names"]),
        )
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.execute(
                delete(self.tables.reference).where(
                    self.tables.reference.c.dataset_name == dataset_name
                )
            )
            await connection.run_sync(table.drop, checkfirst=True)

    async def clear_dataset_rows(self, dataset_name: str) -> None:
        """Remove rows while retaining the canonical dataset definition."""
        reference = await self._get_reference(dataset_name)
        table = _build_dataset_table(
            int(reference["table_idx"]),
            len(reference["metadata"]["column_names"]),
        )
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.execute(delete(table))
            await connection.execute(
                self.tables.reference.update()
                .where(self.tables.reference.c.dataset_name == dataset_name)
                .values(n_rows=0)
            )

    async def reset_database(self) -> None:
        """Remove all canonical datasets and partial payloads."""
        await self.reset_for_test()

    async def persist_partial_payload(
        self,
        payload: PartialPayload | KServeInferenceRequest | KServeInferenceResponse,
        payload_id: str,
        *,
        is_input: bool,
    ) -> None:
        """Upsert one partial payload using the canonical three-column table."""
        if not payload_id or len(payload_id) > _MAX_DATASET_NAME_LENGTH:
            msg = "partial payload IDs must contain 1 to 255 characters"
            raise ValueError(msg)
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.execute(
                delete(self.tables.partial_payloads).where(
                    (self.tables.partial_payloads.c.payload_id == payload_id)
                    & (self.tables.partial_payloads.c.is_input == is_input)
                )
            )
            await connection.execute(
                self.tables.partial_payloads.insert().values(
                    payload_id=payload_id,
                    is_input=is_input,
                    payload_data=serialize_model(payload),
                )
            )

    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool, is_modelmesh: bool
    ) -> PartialPayload | KServeInferenceRequest | KServeInferenceResponse | None:
        """Read one partial payload and decode it using the source classification."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.partial_payloads.c.payload_data).where(
                    (self.tables.partial_payloads.c.payload_id == payload_id)
                    & (self.tables.partial_payloads.c.is_input == is_input)
                )
            )
            row = result.first()
        if row is None:
            return None
        target_class = (
            PartialPayload
            if is_modelmesh
            else KServeInferenceRequest
            if is_input
            else KServeInferenceResponse
        )
        return deserialize_model(bytes(row[0]), target_class)

    async def delete_partial_payload(self, payload_id: str, *, is_input: bool) -> None:
        """Delete both possible representations for one payload ID."""
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.execute(
                delete(self.tables.partial_payloads).where(
                    (self.tables.partial_payloads.c.payload_id == payload_id)
                    & (self.tables.partial_payloads.c.is_input == is_input)
                )
            )

    async def persist_modelmesh_payload(
        self, payload: PartialPayload, request_id: str, *, is_input: bool
    ) -> None:
        """Persist one ModelMesh payload."""
        await self.persist_partial_payload(payload, request_id, is_input=is_input)

    async def get_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> PartialPayload | None:
        """Retrieve one ModelMesh payload."""
        payload = await self.get_partial_payload(
            request_id, is_input=is_input, is_modelmesh=True
        )
        return payload if isinstance(payload, PartialPayload) else None

    async def delete_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> None:
        """Delete one ModelMesh payload."""
        await self.delete_partial_payload(request_id, is_input=is_input)

    async def get_known_models(self) -> list[str]:
        """Return model IDs inferred from canonical dataset suffixes."""
        datasets = await self.list_all_datasets()
        models: set[str] = set()
        for dataset_name in datasets:
            for suffix in ("_inputs", "_outputs", "_metadata"):
                if dataset_name.endswith(suffix):
                    models.add(dataset_name[: -len(suffix)])
                    break
        return sorted(models)

    async def get_metadata(self, model_id: str) -> None:
        """Migration verification does not need the service metadata projection."""
        _ = model_id

    async def _get_reference(self, dataset_name: str) -> dict[str, Any]:
        """Return one canonical reference row or raise a stable error."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            reference = await self._get_reference_in_connection(
                connection, dataset_name
            )
        if reference is None:
            msg = f"Dataset '{dataset_name}' does not exist"
            raise ValueError(msg)
        return reference

    async def _get_reference_in_connection(
        self, connection: AsyncConnection, dataset_name: str
    ) -> dict[str, Any] | None:
        """Read a reference row using an existing async connection."""
        result = await connection.execute(
            select(self.tables.reference).where(
                self.tables.reference.c.dataset_name == dataset_name
            )
        )
        row = result.mappings().first()
        return dict(row) if row is not None else None

    async def _list_references(self) -> list[tuple[int, str]]:
        """Return table indexes and names for cleanup."""
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(
                    self.tables.reference.c.table_idx,
                    self.tables.reference.c.dataset_name,
                )
            )
            return [(int(row[0]), str(row[1])) for row in result]

    async def _column_count(self, table_idx: int) -> int:
        """Read a dynamic table's column count from its reference metadata."""
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.reference.c.metadata).where(
                    self.tables.reference.c.table_idx == table_idx
                )
            )
            metadata = result.scalar_one()
        return len(metadata["column_names"])

    async def _update_metadata(
        self, dataset_name: str, metadata: dict[str, Any]
    ) -> None:
        """Replace one reference metadata document atomically."""
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.execute(
                update(self.tables.reference)
                .where(self.tables.reference.c.dataset_name == dataset_name)
                .values(metadata=metadata)
            )


def _decoded_rows_array(
    rows: list[list[Any]], shape: tuple[int, ...], dtype: np.dtype[Any] | type[object]
) -> np.ndarray:
    """Retain opaque cells in a declared matrix without expanding their contents."""
    if len(shape) == 1 and dtype is object:
        batch = np.empty((len(rows), *shape), dtype=object)
        for index, row in enumerate(rows):
            for column, value in enumerate(row):
                batch[index, column] = value
        return batch
    return np.array(rows, dtype=dtype)


def _validate_dataset_name(dataset_name: str) -> None:
    """Validate logical names before writing the service registry."""
    if not dataset_name or len(dataset_name) > _MAX_DATASET_NAME_LENGTH:
        msg = "dataset names must contain 1 to 255 characters"
        raise ValueError(msg)


def _normalize_rows(rows: np.ndarray | list) -> np.ndarray:
    """Normalize a row batch without flattening tensor dimensions."""
    normalized = np.asarray(rows, dtype=object)
    if normalized.ndim < _MIN_MATRIX_DIMENSIONS:
        normalized = normalized.reshape(-1, 1)
    return normalized
