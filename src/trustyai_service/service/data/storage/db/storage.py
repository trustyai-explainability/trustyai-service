"""Async SQLAlchemy storage implementing the TrustyAI storage contract."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import delete, func, select

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.data.metadata.storage_metadata import (
    StorageMetadata,
    StorageMetadataConfig,
)
from trustyai_service.service.data.modelmesh_parser import PartialPayload
from trustyai_service.service.data.storage.storage_interface import StorageInterface
from trustyai_service.service.payloads.service.schema import Schema
from trustyai_service.service.payloads.service.schema_item import SchemaItem
from trustyai_service.service.payloads.values.data_type import DataType
from trustyai_service.service.serialization import deserialize_model, serialize_model
from trustyai_service.service.serialization.detection import safe_gzip_decompress
from trustyai_service.service.serialization.encoders import (
    json_decoder_hook,
    json_encoder,
)

from .schema import MigrationTables, build_tables

logger = logging.getLogger(__name__)
_MATRIX_DIMENSIONS = 2


class DBStorage(StorageInterface):
    """Store logical TrustyAI data in dialect-neutral SQL tables."""

    def __init__(self, engine: AsyncEngine, *, namespace: str = "default") -> None:
        """Initialize storage for one isolated namespace."""
        self.engine = engine
        self.namespace = namespace
        self.tables: MigrationTables = build_tables(namespace)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Create namespaced tables exactly once per storage object."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self.engine.begin() as connection:
                await connection.run_sync(self.tables.metadata.create_all)
            self._initialized = True

    async def initialize(self) -> None:
        """Create tables before a migration begins."""
        await self._ensure_initialized()

    async def reset_for_test(self) -> None:
        """Drop only this namespace's tables.

        This method is intentionally explicit and exists for test/maintenance
        setup. Normal migrations never drop target data.
        """
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.run_sync(self.tables.metadata.drop_all)
        self._initialized = False

    async def dataset_exists(self, dataset_name: str) -> bool:
        """Return whether a logical dataset is registered."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(func.count())
                .select_from(self.tables.datasets)
                .where(self.tables.datasets.c.dataset_name == dataset_name)
            )
            return result.scalar_one() > 0

    async def list_all_datasets(self) -> list[str]:
        """Return all logical dataset names in stable order."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.datasets.c.dataset_name).order_by(
                    self.tables.datasets.c.dataset_name
                )
            )
            return list(result.scalars())

    async def dataset_rows(self, dataset_name: str) -> int:
        """Return the stored row count for a dataset."""
        row = await self._get_dataset(dataset_name)
        return int(row["n_rows"])

    async def dataset_shape(self, dataset_name: str) -> tuple[int]:
        """Return the row and column dimensions for a dataset."""
        row = await self._get_dataset(dataset_name)
        shape = row["metadata"].get("shape", [-1, 0])
        return (int(row["n_rows"]), *[int(value) for value in shape[1:]])

    async def ensure_dataset(
        self,
        dataset_name: str,
        column_names: Sequence[str],
        *,
        aliases: Sequence[str] | None = None,
        column_types: Sequence[str] | None = None,
        serialized: bool | None = None,
    ) -> None:
        """Register a dataset without requiring its first row to exist."""
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

        metadata = {
            "column_names": names,
            "aliased_names": alias_names,
            "column_types": types,
            "shape": [-1, len(names)],
            "serialized": bool(serialized) if serialized is not None else False,
        }
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            existing = await self._get_dataset_in_connection(connection, dataset_name)
            if existing is None:
                await connection.execute(
                    self.tables.datasets.insert().values(
                        dataset_name=dataset_name,
                        metadata=metadata,
                        n_rows=0,
                    )
                )
                return

            existing_metadata = existing["metadata"]
            if existing_metadata.get("column_names") != names:
                msg = f"Dataset '{dataset_name}' has incompatible column names"
                raise ValueError(msg)
            existing_types = existing_metadata.get("column_types")
            if types is not None and existing_types not in (None, types):
                msg = f"Dataset '{dataset_name}' has incompatible column types"
                raise ValueError(msg)

            merged = dict(existing_metadata)
            if types is not None:
                merged["column_types"] = types
            if aliases is not None:
                merged["aliased_names"] = alias_names
            if serialized is not None:
                merged["serialized"] = bool(serialized)
            await connection.execute(
                self.tables.datasets.update()
                .where(self.tables.datasets.c.dataset_name == dataset_name)
                .values(metadata=merged)
            )

    async def write_data(
        self,
        dataset_name: str,
        new_rows: np.ndarray | list,
        column_names: list[str],
    ) -> None:
        """Append rows using one transaction and stable row indexes."""
        rows = np.asarray(new_rows, dtype=object)
        if rows.ndim == 1:
            rows = rows.reshape(-1, 1)
        if rows.ndim != _MATRIX_DIMENSIONS:
            msg = f"Expected a two-dimensional row matrix, got shape {rows.shape}"
            raise ValueError(msg)
        if rows.shape[1] != len(column_names):
            msg = (
                f"Shape mismatch: {rows.shape[1]} columns supplied, "
                f"but {len(column_names)} names were provided"
            )
            raise ValueError(msg)
        await self.ensure_dataset(dataset_name, column_names)
        if len(rows) == 0:
            return

        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            existing = await self._get_dataset_in_connection(connection, dataset_name)
            if existing is None:
                msg = f"Dataset '{dataset_name}' does not exist"
                raise ValueError(msg)
            start = int(existing["n_rows"])
            encoded = [
                {
                    "dataset_name": dataset_name,
                    "row_idx": start + index,
                    "row_data": gzip.compress(
                        json.dumps(row.tolist(), default=json_encoder).encode("utf-8"),
                        compresslevel=1,
                    ),
                }
                for index, row in enumerate(rows)
            ]
            await connection.execute(self.tables.data.insert(), encoded)
            await connection.execute(
                self.tables.datasets.update()
                .where(self.tables.datasets.c.dataset_name == dataset_name)
                .values(n_rows=start + len(encoded))
            )

    async def read_data(
        self,
        dataset_name: str,
        start_row: int | None = None,
        n_rows: int | None = None,
    ) -> np.ndarray:
        """Read and decode canonical JSON+gzip rows."""
        await self._ensure_initialized()
        start = 0 if start_row is None else start_row
        if start < 0 or (n_rows is not None and n_rows < 0):
            msg = "row offsets and limits must be non-negative"
            raise ValueError(msg)
        query = (
            select(self.tables.data.c.row_data)
            .where(self.tables.data.c.dataset_name == dataset_name)
            .order_by(self.tables.data.c.row_idx)
            .offset(start)
        )
        if n_rows is not None:
            query = query.limit(n_rows)
        async with self.engine.connect() as connection:
            result = await connection.execute(query)
            rows = [
                json.loads(
                    safe_gzip_decompress(bytes(row[0])).decode("utf-8"),
                    object_hook=json_decoder_hook,
                )
                for row in result
            ]
        if rows:
            return np.asarray(rows, dtype=object)
        metadata = await self._get_dataset(dataset_name)
        column_count = len(metadata["metadata"].get("column_names", []))
        return np.empty((0, column_count), dtype=object)

    async def delete_dataset(self, dataset_name: str) -> None:
        """Delete one dataset and its rows from this target namespace."""
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.execute(
                delete(self.tables.data).where(
                    self.tables.data.c.dataset_name == dataset_name
                )
            )
            await connection.execute(
                delete(self.tables.datasets).where(
                    self.tables.datasets.c.dataset_name == dataset_name
                )
            )

    async def clear_dataset_rows(self, dataset_name: str) -> None:
        """Remove rows while retaining the dataset schema for resumability."""
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            await connection.execute(
                delete(self.tables.data).where(
                    self.tables.data.c.dataset_name == dataset_name
                )
            )
            await connection.execute(
                self.tables.datasets.update()
                .where(self.tables.datasets.c.dataset_name == dataset_name)
                .values(n_rows=0)
            )

    async def get_original_column_names(self, dataset_name: str) -> list[str]:
        """Return original source column names."""
        row = await self._get_dataset(dataset_name)
        return list(row["metadata"]["column_names"])

    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        """Return current column aliases."""
        row = await self._get_dataset(dataset_name)
        metadata = row["metadata"]
        return list(metadata.get("aliased_names", metadata["column_names"]))

    async def apply_name_mapping(
        self, dataset_name: str, name_mapping: dict[str, str]
    ) -> None:
        """Apply aliases without changing original names."""
        await self._update_metadata(
            dataset_name,
            lambda metadata: {
                **metadata,
                "aliased_names": [
                    name_mapping.get(name, name) for name in metadata["column_names"]
                ],
            },
        )

    async def clear_name_mapping(self, dataset_name: str) -> None:
        """Reset aliases to original names."""
        await self._update_metadata(
            dataset_name,
            lambda metadata: {**metadata, "aliased_names": metadata["column_names"]},
        )

    async def set_column_types(
        self, dataset_name: str, column_types: list[str]
    ) -> None:
        """Store canonical column type labels."""
        await self._update_metadata(
            dataset_name,
            lambda metadata: {**metadata, "column_types": list(column_types)},
        )

    async def get_column_types(self, dataset_name: str) -> list[str] | None:
        """Return canonical column type labels when present."""
        row = await self._get_dataset(dataset_name)
        types = row["metadata"].get("column_types")
        return list(types) if types is not None else None

    async def get_known_models(self) -> list[str]:
        """Return model IDs inferred from logical dataset suffixes."""
        datasets = await self.list_all_datasets()
        models: set[str] = set()
        for dataset_name in datasets:
            for suffix in ("_inputs", "_outputs", "_metadata"):
                if dataset_name.endswith(suffix):
                    models.add(dataset_name[: -len(suffix)])
                    break
        return sorted(models)

    async def get_metadata(self, model_id: str) -> StorageMetadata | None:
        """Build the service metadata view for one model."""
        input_name = f"{model_id}_inputs"
        if not await self.dataset_exists(input_name):
            return None

        def schema_for(names: list[str]) -> Schema:
            items = {
                name: SchemaItem(DataType.UNKNOWN, name, index)
                for index, name in enumerate(names)
            }
            return Schema(items=items)

        input_names = await self.get_original_column_names(input_name)
        output_name = f"{model_id}_outputs"
        output_names = (
            await self.get_original_column_names(output_name)
            if await self.dataset_exists(output_name)
            else []
        )
        return StorageMetadata(
            StorageMetadataConfig(
                model_id=model_id,
                input_schema=schema_for(input_names),
                output_schema=schema_for(output_names),
                observations=await self.dataset_rows(input_name),
                recorded_inferences=await self.dataset_rows(input_name) > 0,
            )
        )

    async def persist_partial_payload(
        self,
        payload: PartialPayload | KServeInferenceRequest | KServeInferenceResponse,
        payload_id: str,
        *,
        is_input: bool,
    ) -> None:
        """Upsert a serialized KServe or ModelMesh partial payload."""
        await self._ensure_initialized()
        serialized = serialize_model(payload)
        is_modelmesh = isinstance(payload, PartialPayload)
        async with self.engine.begin() as connection:
            await connection.execute(
                delete(self.tables.partial_payloads).where(
                    (self.tables.partial_payloads.c.payload_id == payload_id)
                    & (self.tables.partial_payloads.c.is_input == is_input)
                    & (self.tables.partial_payloads.c.is_modelmesh == is_modelmesh)
                )
            )
            await connection.execute(
                self.tables.partial_payloads.insert().values(
                    payload_id=payload_id,
                    is_input=is_input,
                    is_modelmesh=is_modelmesh,
                    payload_data=serialized,
                )
            )

    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool, is_modelmesh: bool
    ) -> PartialPayload | KServeInferenceRequest | KServeInferenceResponse | None:
        """Read and deserialize a partial payload."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.partial_payloads.c.payload_data).where(
                    (self.tables.partial_payloads.c.payload_id == payload_id)
                    & (self.tables.partial_payloads.c.is_input == is_input)
                    & (self.tables.partial_payloads.c.is_modelmesh == is_modelmesh)
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
        """Delete both possible representations of one payload ID."""
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
        """Persist a ModelMesh partial payload."""
        await self.persist_partial_payload(payload, request_id, is_input=is_input)

    async def get_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> PartialPayload | None:
        """Read a ModelMesh partial payload."""
        result = await self.get_partial_payload(
            request_id,
            is_input=is_input,
            is_modelmesh=True,
        )
        return result if isinstance(result, PartialPayload) else None

    async def delete_modelmesh_payload(
        self, request_id: str, *, is_input: bool
    ) -> None:
        """Delete a ModelMesh partial payload."""
        await self.delete_partial_payload(request_id, is_input=is_input)

    async def get_migration_run(self, migration_id: str) -> dict[str, Any] | None:
        """Read durable migration-run state."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.runs).where(
                    self.tables.runs.c.migration_id == migration_id
                )
            )
            row = result.mappings().first()
            return dict(row) if row is not None else None

    async def save_migration_run(self, values: dict[str, Any]) -> None:
        """Insert or update durable migration-run state."""
        await self._ensure_initialized()
        migration_id = values["migration_id"]
        async with self.engine.begin() as connection:
            existing = await connection.execute(
                select(self.tables.runs.c.migration_id).where(
                    self.tables.runs.c.migration_id == migration_id
                )
            )
            if existing.first() is None:
                await connection.execute(self.tables.runs.insert().values(**values))
            else:
                await connection.execute(
                    self.tables.runs.update()
                    .where(self.tables.runs.c.migration_id == migration_id)
                    .values(
                        **{
                            key: value
                            for key, value in values.items()
                            if key != "migration_id"
                        }
                    )
                )

    async def get_migration_dataset_state(
        self, migration_id: str, dataset_name: str
    ) -> dict[str, Any] | None:
        """Read durable state for one logical dataset."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            result = await connection.execute(
                select(self.tables.dataset_states).where(
                    (self.tables.dataset_states.c.migration_id == migration_id)
                    & (self.tables.dataset_states.c.dataset_name == dataset_name)
                )
            )
            row = result.mappings().first()
            return dict(row) if row is not None else None

    async def save_migration_dataset_state(self, values: dict[str, Any]) -> None:
        """Insert or update durable state for one dataset."""
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            where = (
                self.tables.dataset_states.c.migration_id == values["migration_id"]
            ) & (self.tables.dataset_states.c.dataset_name == values["dataset_name"])
            existing = await connection.execute(
                select(self.tables.dataset_states.c.status).where(where)
            )
            if existing.first() is None:
                await connection.execute(
                    self.tables.dataset_states.insert().values(**values)
                )
            else:
                await connection.execute(
                    self.tables.dataset_states.update()
                    .where(where)
                    .values(
                        **{
                            key: value
                            for key, value in values.items()
                            if key not in {"migration_id", "dataset_name"}
                        }
                    )
                )

    async def _get_dataset(self, dataset_name: str) -> dict[str, Any]:
        """Read one dataset registry row or raise a stable error."""
        await self._ensure_initialized()
        async with self.engine.connect() as connection:
            row = await self._get_dataset_in_connection(connection, dataset_name)
        if row is None:
            msg = f"Dataset '{dataset_name}' does not exist"
            raise ValueError(msg)
        return row

    async def _get_dataset_in_connection(
        self, connection: AsyncConnection, dataset_name: str
    ) -> dict[str, Any] | None:
        """Read one dataset using an existing connection."""
        result = await connection.execute(
            select(self.tables.datasets).where(
                self.tables.datasets.c.dataset_name == dataset_name
            )
        )
        row = result.mappings().first()
        return dict(row) if row is not None else None

    async def _update_metadata(
        self,
        dataset_name: str,
        updater: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        """Apply an atomic metadata update."""
        await self._ensure_initialized()
        async with self.engine.begin() as connection:
            row = await self._get_dataset_in_connection(connection, dataset_name)
            if row is None:
                msg = f"Dataset '{dataset_name}' does not exist"
                raise ValueError(msg)
            await connection.execute(
                self.tables.datasets.update()
                .where(self.tables.datasets.c.dataset_name == dataset_name)
                .values(metadata=updater(dict(row["metadata"])))
            )
