"""Unified database storage backend using SQLAlchemy Core async.

Supports MariaDB and PostgreSQL through dialect-agnostic queries.
Replaces the MariaDB-specific raw SQL implementation.
"""

import asyncio
import gzip
import json
import logging

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from src.endpoints.consumer import KServeInferenceRequest, KServeInferenceResponse
from src.service.data.storage.db.schema import (
    data_table,
    datasets_table,
    metadata_obj,
    partial_payloads_table,
)
from src.service.data.storage.storage_interface import StorageInterface
from src.service.serialization.encoders import json_decoder_hook, json_encoder

logger = logging.getLogger(__name__)

_MIN_MATRIX_NDIM = 2


class DBStorage(StorageInterface):
    """StorageInterface implementation backed by SQLAlchemy Core async engine."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize with an async SQLAlchemy engine."""
        self.engine = engine
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Create tables on first use (lazy initialization)."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(metadata_obj.create_all)
            self._initialized = True
            logger.info("Database tables initialized")

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        await self._ensure_initialized()

    # === DATASET OPERATIONS ===

    async def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists in the registry."""
        await self._ensure_initialized()
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(func.count())
                .select_from(datasets_table)
                .where(datasets_table.c.dataset_name == dataset_name)
            )
            return result.scalar_one() > 0

    async def list_all_datasets(self) -> list[str]:
        """List all dataset names in the registry."""
        await self._ensure_initialized()
        async with self.engine.connect() as conn:
            result = await conn.execute(select(datasets_table.c.dataset_name))
            return [row.dataset_name for row in result]

    async def dataset_rows(self, dataset_name: str) -> int:
        """Get the number of rows in a dataset."""
        meta = await self._get_dataset_metadata(dataset_name)
        return meta["n_rows"]

    async def dataset_shape(self, dataset_name: str) -> tuple[int]:
        """Get the shape of a dataset."""
        meta = await self._get_dataset_metadata(dataset_name)
        shape = meta["metadata"]["shape"]
        n_rows = meta["n_rows"]
        return (n_rows, *shape[1:])

    async def write_data(
        self, dataset_name: str, new_rows: np.ndarray, column_names: list[str]
    ) -> None:
        """Write new rows to a dataset."""
        await self._ensure_initialized()
        if len(new_rows) == 0:
            msg = f"No data provided! `new_rows`=={new_rows}."
            raise ValueError(msg)

        if new_rows.ndim < _MIN_MATRIX_NDIM:
            new_rows = new_rows.reshape(-1, 1)

        if new_rows.shape[1] != len(column_names):
            msg = (
                f"Shape mismatch: Number of provided column names ({len(column_names)}) "
                f"does not match number of columns in provided array ({new_rows.shape[1]})."
            )
            raise ValueError(msg)

        async with self.engine.begin() as conn:
            exists = await self._dataset_exists_in_conn(conn, dataset_name)

            if not exists:
                ds_metadata = {
                    "column_names": column_names,
                    "aliased_names": column_names,
                    "shape": (-1, *new_rows.shape[1:]),
                }
                await conn.execute(
                    datasets_table.insert().values(
                        dataset_name=dataset_name,
                        metadata=ds_metadata,
                        n_rows=0,
                    )
                )
                current_rows = 0
            else:
                row = await self._get_dataset_row(conn, dataset_name)
                stored_shape = row.metadata["shape"]
                current_rows = row.n_rows

                if len(stored_shape) > 1 and list(stored_shape[1:]) != list(
                    new_rows.shape[1:]
                ):
                    msg = (
                        f"Shape mismatch: new_rows.shape[1:] ({new_rows.shape[1:]}) does not"
                        f" match shape of existing database ({stored_shape[1:]})."
                    )
                    raise ValueError(msg)

            blobs = []
            for i, row_data in enumerate(new_rows):
                row_list = (
                    row_data.tolist()
                    if isinstance(row_data, np.ndarray)
                    else list(row_data)
                )
                json_bytes = json.dumps(row_list, default=json_encoder).encode("utf-8")
                blobs.append(
                    {
                        "dataset_name": dataset_name,
                        "row_idx": current_rows + i,
                        "row_data": gzip.compress(json_bytes),
                    }
                )

            if blobs:
                await conn.execute(data_table.insert(), blobs)

            await conn.execute(
                update(datasets_table)
                .where(datasets_table.c.dataset_name == dataset_name)
                .values(n_rows=current_rows + len(new_rows))
            )

    async def read_data(
        self,
        dataset_name: str,
        start_row: int | None = None,
        n_rows: int | None = None,
    ) -> np.ndarray:
        """Read data from a dataset with optional row range."""
        if start_row is None:
            start_row = 0

        async with self.engine.connect() as conn:
            query = (
                select(data_table.c.row_data)
                .where(data_table.c.dataset_name == dataset_name)
                .order_by(data_table.c.row_idx)
                .offset(start_row)
            )
            if n_rows is not None:
                query = query.limit(n_rows)

            result = await conn.execute(query)
            rows = []
            for row in result:
                decompressed = gzip.decompress(row.row_data)
                rows.append(json.loads(decompressed, object_hook=json_decoder_hook))

            if not rows:
                return np.array([], dtype="O")
            return np.array(rows, dtype="O")

    async def delete_dataset(self, dataset_name: str) -> None:
        """Delete a dataset and all its rows."""
        async with self.engine.begin() as conn:
            await conn.execute(
                delete(data_table).where(data_table.c.dataset_name == dataset_name)
            )
            await conn.execute(
                delete(datasets_table).where(
                    datasets_table.c.dataset_name == dataset_name
                )
            )

    # === COLUMN NAME MAPPING ===

    async def get_original_column_names(self, dataset_name: str) -> list[str]:
        """Get the original column names from the raw payloads."""
        meta = await self._get_dataset_metadata(dataset_name)
        return meta["metadata"]["column_names"]

    async def get_aliased_column_names(self, dataset_name: str) -> list[str]:
        """Get the current aliased column names after name mapping."""
        meta = await self._get_dataset_metadata(dataset_name)
        ds_meta = meta["metadata"]
        return ds_meta.get("aliased_names", ds_meta["column_names"])

    async def apply_name_mapping(
        self, dataset_name: str, name_mapping: dict[str, str]
    ) -> None:
        """Apply column name aliases to a dataset."""
        async with self.engine.begin() as conn:
            row = await self._get_dataset_row(conn, dataset_name)
            ds_meta = dict(row.metadata)
            original = ds_meta["column_names"]
            ds_meta["aliased_names"] = [
                name_mapping.get(name, name) for name in original
            ]
            await conn.execute(
                update(datasets_table)
                .where(datasets_table.c.dataset_name == dataset_name)
                .values(metadata=ds_meta)
            )

    async def clear_name_mapping(self, dataset_name: str) -> None:
        """Clear all column name aliases from a dataset."""
        async with self.engine.begin() as conn:
            row = await self._get_dataset_row(conn, dataset_name)
            ds_meta = dict(row.metadata)
            ds_meta["aliased_names"] = ds_meta["column_names"]
            await conn.execute(
                update(datasets_table)
                .where(datasets_table.c.dataset_name == dataset_name)
                .values(metadata=ds_meta)
            )

    # === MODEL DISCOVERY ===

    async def get_known_models(self) -> list[str]:
        """Get a list of all model IDs that have inference data stored."""
        all_datasets = await self.list_all_datasets()
        model_ids: set[str] = set()

        for name in all_datasets:
            if name.startswith("trustyai_"):
                continue
            for suffix in ("_inputs", "_outputs", "_metadata"):
                if name.endswith(suffix):
                    model_ids.add(name[: -len(suffix)])
                    break

        return list(model_ids)

    async def get_metadata(self, model_id: str) -> dict:
        """Get metadata for a specific model."""
        input_dataset = f"{model_id}_inputs"
        output_dataset = f"{model_id}_outputs"
        metadata_dataset = f"{model_id}_metadata"

        result: dict = {
            "modelId": model_id,
            "inputData": None,
            "outputData": None,
            "metadataData": None,
        }

        for key, ds_name, include_aliases in [
            ("inputData", input_dataset, True),
            ("outputData", output_dataset, True),
            ("metadataData", metadata_dataset, False),
        ]:
            if await self.dataset_exists(ds_name):
                try:
                    shape = await self.dataset_shape(ds_name)
                    names = await self.get_original_column_names(ds_name)
                    entry: dict = {
                        "shape": list(shape),
                        "columnNames": list(names),
                    }
                    if include_aliases:
                        aliased = await self.get_aliased_column_names(ds_name)
                        entry["aliasedNames"] = list(aliased)
                    result[key] = entry
                except (
                    Exception
                ) as e:  # Intentional: per-dataset errors should not break metadata
                    logger.warning(
                        "Error getting %s metadata for %s: %s", key, model_id, e
                    )

        return result

    # === PARTIAL PAYLOADS (KServe reconciliation) ===

    async def persist_partial_payload(
        self,
        payload: KServeInferenceRequest | KServeInferenceResponse,
        payload_id: str,
        *,
        is_input: bool,
    ) -> None:
        """Persist a partial payload before reconciliation."""
        await self._ensure_initialized()
        serialized = gzip.compress(payload.model_dump_json().encode("utf-8"))
        async with self.engine.begin() as conn:
            await conn.execute(
                partial_payloads_table.insert().values(
                    payload_id=payload_id,
                    is_input=is_input,
                    payload_data=serialized,
                )
            )

    async def get_partial_payload(
        self, payload_id: str, *, is_input: bool
    ) -> KServeInferenceRequest | KServeInferenceResponse | None:
        """Retrieve a stored partial payload."""
        async with self.engine.begin() as conn:
            result = await conn.execute(
                select(partial_payloads_table.c.payload_data).where(
                    (partial_payloads_table.c.payload_id == payload_id)
                    & (partial_payloads_table.c.is_input == is_input)
                )
            )
            row = result.first()
            if row is None:
                return None

            payload_bytes = bytes(row.payload_data)
            json_str = gzip.decompress(payload_bytes).decode("utf-8")
            target_class = (
                KServeInferenceRequest if is_input else KServeInferenceResponse
            )
            parsed = target_class.model_validate_json(json_str)

            await conn.execute(
                delete(partial_payloads_table).where(
                    (partial_payloads_table.c.payload_id == payload_id)
                    & (partial_payloads_table.c.is_input == is_input)
                )
            )

        return parsed

    async def delete_partial_payload(self, payload_id: str, *, is_input: bool) -> None:
        """Delete a stored partial payload."""
        async with self.engine.begin() as conn:
            await conn.execute(
                delete(partial_payloads_table).where(
                    (partial_payloads_table.c.payload_id == payload_id)
                    & (partial_payloads_table.c.is_input == is_input)
                )
            )

    # === ModelMesh methods (removed by PR #216, stubs for interface compat) ===

    async def persist_modelmesh_payload(  # type: ignore[override]
        self, *_args: object, **_kwargs: object
    ) -> None:
        """Not implemented — ModelMesh support removed."""
        msg = "ModelMesh support has been removed"
        raise NotImplementedError(msg)

    async def get_modelmesh_payload(  # type: ignore[override]
        self, *_args: object, **_kwargs: object
    ) -> None:
        """Not implemented — ModelMesh support removed."""
        msg = "ModelMesh support has been removed"
        raise NotImplementedError(msg)

    async def delete_modelmesh_payload(  # type: ignore[override]
        self, *_args: object, **_kwargs: object
    ) -> None:
        """Not implemented — ModelMesh support removed."""
        msg = "ModelMesh support has been removed"
        raise NotImplementedError(msg)

    # === INTERNAL HELPERS ===

    async def _dataset_exists_in_conn(self, conn: object, dataset_name: str) -> bool:
        result = await conn.execute(
            select(func.count())
            .select_from(datasets_table)
            .where(datasets_table.c.dataset_name == dataset_name)
        )
        return result.scalar_one() > 0

    async def _get_dataset_row(self, conn: object, dataset_name: str) -> object:
        result = await conn.execute(
            select(datasets_table).where(datasets_table.c.dataset_name == dataset_name)
        )
        row = result.first()
        if row is None:
            msg = f"Dataset '{dataset_name}' does not exist."
            raise ValueError(msg)
        return row

    async def _get_dataset_metadata(self, dataset_name: str) -> dict:
        async with self.engine.connect() as conn:
            row = await self._get_dataset_row(conn, dataset_name)
            return {"metadata": row.metadata, "n_rows": row.n_rows}
