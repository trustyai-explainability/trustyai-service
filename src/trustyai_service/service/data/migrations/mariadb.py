"""MariaDB source adapter for current and legacy TrustyAI schemas."""

from __future__ import annotations

import hashlib
import json
import uuid
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from sqlalchemy import inspect, text

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.serialization.detection import safe_gzip_decompress
from trustyai_service.service.serialization.encoders import json_decoder_hook

from .contracts import (
    DatasetDescriptor,
    MigrationSource,
    PartialPayloadRecord,
    SourceManifest,
    decode_partial_payload,
    partial_payload_fingerprint,
)
from .errors import MigrationPreflightError, MigrationValidationError

_CONTENT_HASH_PAGE_SIZE = 1_024
_MIN_SHAPE_DIMENSIONS = 2


class _Digest(Protocol):
    """Small protocol for hashlib-compatible digest objects."""

    def update(self, data: bytes, /) -> None:
        """Add bytes to the digest."""


@dataclass(frozen=True)
class _LegacyDatasetSpec:
    """Additional information needed to stream one legacy logical dataset."""

    model_id: str
    kind: str
    raw_columns: tuple[str, ...]
    selected_indices: tuple[int, ...]
    aliases: tuple[str, ...]


class MariaDBSource(MigrationSource):
    """Read current TrustyAI or legacy Java/MariaDB data through SQLAlchemy."""

    source_type = "MariaDB"

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        schema: str = "auto",
        quiesced: bool,
    ) -> None:
        """Initialize a read-only MariaDB source."""
        if schema not in {"auto", "legacy", "trustyai_v2"}:
            msg = f"unsupported MariaDB source schema: {schema}"
            raise ValueError(msg)
        self.engine = engine
        self.schema = schema
        self.quiesced = quiesced
        self._mode: str | None = None
        self._current_specs: dict[str, tuple[int, str]] = {}
        self._legacy_specs: dict[str, _LegacyDatasetSpec] = {}
        self._last_manifest: SourceManifest | None = None

    async def preflight(self) -> SourceManifest:
        """Detect and validate the source schema without writing to it."""
        table_names = await self._table_names()
        has_legacy = "DataframeMetadata" in table_names
        has_current = "trustyai_v2_table_reference" in table_names
        if self.schema == "auto" and has_legacy and has_current:
            msg = (
                "MariaDB source contains both legacy and trustyai_v2 schemas; "
                "set schema='legacy' or schema='trustyai_v2' explicitly"
            )
            raise MigrationPreflightError(msg)
        if self.schema == "legacy" and not has_legacy:
            msg = "missing MariaDB legacy schema"
            raise MigrationPreflightError(msg)
        if self.schema == "trustyai_v2" and not has_current:
            msg = "missing MariaDB trustyai_v2 schema"
            raise MigrationPreflightError(msg)
        if self.schema == "legacy" or (self.schema == "auto" and has_legacy):
            self._mode = "legacy"
            manifest = await self._build_legacy_manifest()
        elif self.schema == "trustyai_v2" or (self.schema == "auto" and has_current):
            self._mode = "trustyai_v2"
            manifest = await self._build_current_manifest(table_names)
        else:
            msg = "MariaDB source does not contain a supported TrustyAI schema"
            raise MigrationPreflightError(msg)
        self._last_manifest = manifest
        return manifest

    async def read_rows(
        self, dataset: DatasetDescriptor, start_row: int, n_rows: int
    ) -> np.ndarray:
        """Read one bounded logical row batch."""
        if start_row < 0 or n_rows < 0:
            msg = "row offsets and limits must be non-negative"
            raise ValueError(msg)
        if self._mode is None:
            await self.preflight()
        if self._mode == "legacy":
            return await self._read_legacy_rows(dataset, start_row, n_rows)
        if self._mode == "trustyai_v2":
            return await self._read_current_rows(dataset, start_row, n_rows)
        msg = "MariaDB source mode was not initialized"
        raise MigrationPreflightError(msg)

    async def partial_payloads(self) -> tuple[PartialPayloadRecord, ...]:
        """Return partial payloads from the current TrustyAI schema."""
        manifest = self._last_manifest or await self.preflight()
        return manifest.partial_payloads

    async def _table_names(self) -> set[str]:
        """Inspect source tables through SQLAlchemy's synchronous inspector."""
        async with self.engine.connect() as connection:
            return set(await connection.run_sync(inspect_table_names))

    async def _build_current_manifest(self, table_names: set[str]) -> SourceManifest:
        """Read the current TrustyAI v2 dataset registry."""
        reference = "trustyai_v2_table_reference"
        if reference not in table_names:
            msg = f"missing MariaDB table: {reference}"
            raise MigrationPreflightError(msg)
        self._current_specs.clear()
        query = text(
            "SELECT table_idx, dataset_name, metadata, n_rows "
            "FROM `trustyai_v2_table_reference` ORDER BY dataset_name"
        )
        descriptors: list[DatasetDescriptor] = []
        content_digest = hashlib.sha256()
        async with self.engine.connect() as connection:
            result = await connection.execute(query)
            rows = result.mappings().all()
            for row in rows:
                metadata = _json_object(row["metadata"])
                names = _string_list(metadata.get("column_names"))
                if not names:
                    msg = f"MariaDB dataset '{row['dataset_name']}' has no column metadata"
                    raise MigrationValidationError(msg)
                aliases = _string_list(metadata.get("aliased_names", names))
                types = _string_list(
                    metadata.get("column_types", ["UNKNOWN"] * len(names))
                )
                if len(aliases) != len(names) or len(types) != len(names):
                    msg = f"MariaDB dataset '{row['dataset_name']}' has mismatched column metadata"
                    raise MigrationValidationError(msg)
                try:
                    table_idx = int(row["table_idx"])
                    declared_count = int(row["n_rows"])
                except (TypeError, ValueError) as error:
                    msg = f"MariaDB dataset '{row['dataset_name']}' has invalid registry metadata"
                    raise MigrationValidationError(msg) from error
                table_name = _current_table_name(table_idx)
                try:
                    actual_count = int(
                        (
                            await connection.execute(
                                text(f"SELECT COUNT(*) FROM `{table_name}`")  # noqa: S608 -- table name is an integer-derived internal identifier
                            )
                        ).scalar_one()
                    )
                    await self._hash_current_rows(
                        connection,
                        table_name,
                        table_idx,
                        len(names),
                        content_digest,
                    )
                except MigrationValidationError:
                    raise
                except Exception as error:
                    msg = f"unable to read MariaDB dataset '{row['dataset_name']}'"
                    raise MigrationValidationError(msg) from error
                if actual_count != declared_count:
                    msg = (
                        f"MariaDB dataset '{row['dataset_name']}' row count is inconsistent: "
                        f"registry={declared_count}, table={actual_count}"
                    )
                    raise MigrationValidationError(msg)
                descriptor = DatasetDescriptor(
                    name=str(row["dataset_name"]),
                    column_names=tuple(names),
                    aliases=tuple(aliases),
                    column_types=tuple(types),
                    row_count=declared_count,
                    shape=_shape_from_metadata(metadata, len(names)),
                    serialized=False,
                    source_file=str(table_idx),
                )
                descriptors.append(descriptor)
                self._current_specs[descriptor.name] = (
                    table_idx,
                    descriptor.name,
                )
            partials = await self._read_current_partials(connection, table_names)
        return self._manifest(
            descriptors,
            partials,
            mode="trustyai_v2",
            content_digest=content_digest.hexdigest(),
        )

    async def _build_legacy_manifest(self) -> SourceManifest:
        """Translate the legacy Java schema into logical datasets."""
        self._legacy_specs.clear()
        model_query = text("SELECT DISTINCT id FROM DataframeMetadata ORDER BY id")
        descriptors: list[DatasetDescriptor] = []
        content_digest = hashlib.sha256()
        async with self.engine.connect() as connection:
            model_result = await connection.execute(model_query)
            model_ids = [str(row[0]) for row in model_result]
            for model_id in model_ids:
                descriptors.extend(await self._legacy_descriptors(connection, model_id))
                await self._hash_legacy_rows(connection, model_id, content_digest)
            partials = await self._read_legacy_partials(connection)
        return self._manifest(
            descriptors,
            partials,
            mode="legacy",
            content_digest=content_digest.hexdigest(),
        )

    async def _legacy_descriptors(
        self, connection: AsyncConnection, model_id: str
    ) -> list[DatasetDescriptor]:
        """Read legacy schemas and build input/output/metadata descriptors."""
        names_result = await connection.execute(
            text(
                "SELECT names FROM DataframeMetadata_names "
                "WHERE DataframeMetadata_id=:model_id"
            ),
            {"model_id": model_id},
        )
        raw_names = [str(row[0]) for row in names_result]
        input_result = await connection.execute(
            text(
                "SELECT inputs FROM DataframeMetadata_inputs "
                "WHERE DataframeMetadata_id=:model_id"
            ),
            {"model_id": model_id},
        )
        flags = [row[0] for row in input_result]
        if len(raw_names) != len(flags):
            msg = f"legacy dataset '{model_id}' has incomplete input metadata"
            raise MigrationValidationError(msg)
        input_indices = tuple(
            index for index, flag in enumerate(flags) if _is_input_flag(flag)
        )
        output_indices = tuple(
            index for index, flag in enumerate(flags) if not _is_input_flag(flag)
        )
        input_names = tuple(raw_names[index] for index in input_indices)
        output_names = tuple(raw_names[index] for index in output_indices)
        input_mapping, output_mapping = await self._legacy_name_mappings(
            connection,
            model_id,
            set(input_names),
            set(output_names),
        )
        row_count_result = await connection.execute(
            text("SELECT COUNT(*) FROM DataframeRow WHERE modelId=:model_id"),
            {"model_id": model_id},
        )
        row_count = int(row_count_result.scalar_one())
        specs = (
            _LegacyDatasetSpec(
                model_id,
                "inputs",
                tuple(raw_names),
                input_indices,
                tuple(input_mapping.get(name, name) for name in input_names),
            ),
            _LegacyDatasetSpec(
                model_id,
                "outputs",
                tuple(raw_names),
                output_indices,
                tuple(output_mapping.get(name, name) for name in output_names),
            ),
            _LegacyDatasetSpec(
                model_id,
                "metadata",
                ("timestamp", "rowId", "tag"),
                (0, 1, 2),
                ("timestamp", "rowId", "tag"),
            ),
        )
        descriptors: list[DatasetDescriptor] = []
        for spec in specs:
            name = f"{model_id}_{spec.kind}"
            self._legacy_specs[name] = spec
            descriptors.append(
                DatasetDescriptor(
                    name=name,
                    column_names=(
                        tuple(raw_names[index] for index in spec.selected_indices)
                        if spec.kind != "metadata"
                        else spec.raw_columns
                    ),
                    aliases=spec.aliases,
                    column_types=tuple(
                        ["UNKNOWN"]
                        * len(
                            spec.selected_indices
                            if spec.kind != "metadata"
                            else spec.raw_columns
                        )
                    ),
                    row_count=row_count,
                    shape=(
                        len(spec.selected_indices)
                        if spec.kind != "metadata"
                        else len(spec.raw_columns),
                    ),
                    source_file=f"legacy:{model_id}:{spec.kind}",
                )
            )
        return descriptors

    async def _legacy_name_mappings(
        self,
        connection: AsyncConnection,
        model_id: str,
        input_names: set[str],
        output_names: set[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Read legacy input/output alias mappings."""
        query = text(
            "SELECT items_KEY, mappedItems_KEY "
            "FROM StorageMetadata_StorageSchema "
            "JOIN storageSchema_mappedItems "
            "ON storageSchema_mappedItems.Schema_id = StorageMetadata_StorageSchema.schemas_id "
            "JOIN storageSchema_originalItems "
            "ON storageSchema_originalItems.items_id = storageSchema_mappedItems.mappedItems_id "
            "WHERE StorageMetadata_modelId=:model_id"
        )
        result = await connection.execute(query, {"model_id": model_id})
        input_mapping: dict[str, str] = {}
        output_mapping: dict[str, str] = {}
        for original, mapped in result:
            original_name = str(original)
            if original_name in input_names:
                input_mapping[original_name] = str(mapped)
            elif original_name in output_names:
                output_mapping[original_name] = str(mapped)
        return input_mapping, output_mapping

    async def _read_legacy_rows(
        self, descriptor: DatasetDescriptor, start_row: int, n_rows: int
    ) -> np.ndarray:
        """Read legacy Java rows and select input/output/metadata values."""
        try:
            import javaobj  # noqa: PLC0415 -- legacy decoding is an optional extra
        except ModuleNotFoundError as error:
            msg = "legacy MariaDB migration requires the 'javaobj-py3' extra"
            raise MigrationPreflightError(msg) from error

        spec = self._legacy_specs.get(descriptor.name)
        if spec is None:
            msg = f"legacy dataset '{descriptor.name}' was not discovered"
            raise MigrationValidationError(msg)
        if spec.kind == "metadata":
            return await self._read_legacy_metadata(spec, start_row, n_rows)

        separator = f",-{uuid.uuid4()}-,"
        query = text(
            "SELECT DataframeRow.timestamp, DataframeRow.modelId, "  # noqa: S608 -- separator is an internal UUID
            "DataframeRow.rowId, DataframeRow.tag, "
            "GROUP_CONCAT(DataframeRow_Values.serializableObject "
            f"ORDER BY DataframeRow_Values.order_column SEPARATOR '{separator}') "
            "FROM DataframeRow "
            "LEFT JOIN DataframeRow_Values "
            "ON DataframeRow.dbId = DataframeRow_Values.DataframeRow_dbId "
            "WHERE DataframeRow.modelId=:model_id "
            "GROUP BY DataframeRow.dbId "
            "ORDER BY DataframeRow.dbId "
            "LIMIT :n_rows OFFSET :start_row"
        )
        async with self.engine.connect() as connection:
            result = await connection.execute(
                query,
                {
                    "model_id": spec.model_id,
                    "n_rows": n_rows,
                    "start_row": start_row,
                },
            )
            rows = []
            for row in result:
                serialized_values = row[4]
                if serialized_values is None:
                    values: list[object] = []
                else:
                    raw = (
                        serialized_values
                        if isinstance(serialized_values, bytes)
                        else str(serialized_values).encode()
                    )
                    try:
                        values = [
                            _legacy_value(javaobj.loads(item).object)
                            for item in raw.split(separator.encode())
                        ]
                    except Exception as error:
                        msg = (
                            f"unable to decode legacy MariaDB row for '{spec.model_id}'"
                        )
                        raise MigrationValidationError(msg) from error
                rows.append([values[index] for index in spec.selected_indices])
        return np.asarray(rows, dtype=object)

    async def _read_legacy_metadata(
        self, spec: _LegacyDatasetSpec, start_row: int, n_rows: int
    ) -> np.ndarray:
        """Read the row-level metadata subset of the legacy table."""
        query = text(
            "SELECT timestamp, rowId, tag FROM DataframeRow "
            "WHERE modelId=:model_id ORDER BY dbId "
            "LIMIT :n_rows OFFSET :start_row"
        )
        async with self.engine.connect() as connection:
            result = await connection.execute(
                query,
                {
                    "model_id": spec.model_id,
                    "n_rows": n_rows,
                    "start_row": start_row,
                },
            )
            return np.asarray([list(row) for row in result], dtype=object)

    async def _read_current_rows(
        self, descriptor: DatasetDescriptor, start_row: int, n_rows: int
    ) -> np.ndarray:
        """Read current v2 per-dataset blob columns."""
        table_spec = self._current_specs.get(descriptor.name)
        if table_spec is None:
            msg = f"current dataset '{descriptor.name}' was not discovered"
            raise MigrationValidationError(msg)
        table_name = _current_table_name(table_spec[0])
        columns = ", ".join(
            f"column_{index}" for index in range(len(descriptor.column_names))
        )
        query = text(
            f"SELECT {columns} FROM `{table_name}` "  # noqa: S608 -- table name is an integer-derived internal identifier
            "ORDER BY row_idx LIMIT :n_rows OFFSET :start_row"
        )
        async with self.engine.connect() as connection:
            result = await connection.execute(
                query,
                {"n_rows": n_rows, "start_row": start_row},
            )
            rows = []
            for row in result:
                try:
                    rows.append(
                        [
                            json.loads(
                                _decompress_json(cell),
                                object_hook=json_decoder_hook,
                            )
                            for cell in row
                        ]
                    )
                except MigrationValidationError:
                    raise
                except Exception as error:
                    msg = f"unable to decode MariaDB row for '{descriptor.name}'"
                    raise MigrationValidationError(msg) from error
        return np.asarray(rows, dtype=object)

    async def _read_current_partials(
        self, connection: AsyncConnection, table_names: set[str]
    ) -> list[PartialPayloadRecord]:
        """Read current KServe and ModelMesh partial payloads when the table exists."""
        table = "trustyai_v2_partial_payloads"
        if table not in table_names:
            return []
        result = await connection.execute(
            text(
                "SELECT payload_id, is_input, payload_data "
                "FROM `trustyai_v2_partial_payloads`"
            )
        )
        records: list[PartialPayloadRecord] = []
        for row in result:
            is_input = bool(row[1])
            try:
                payload, is_modelmesh = decode_partial_payload(
                    bytes(row[2]),
                    KServeInferenceRequest if is_input else KServeInferenceResponse,
                )
            except (TypeError, ValueError, OSError) as error:
                msg = "unable to decode MariaDB partial payload"
                raise MigrationValidationError(msg) from error
            records.append(
                PartialPayloadRecord(
                    payload_id=str(row[0]),
                    is_input=is_input,
                    is_modelmesh=is_modelmesh,
                    payload=payload,
                )
            )
        return records

    async def _read_legacy_partials(
        self, connection: AsyncConnection
    ) -> list[PartialPayloadRecord]:
        """Legacy TrustyAI did not have a supported partial-payload table."""
        _ = connection
        return []

    def _manifest(
        self,
        descriptors: list[DatasetDescriptor],
        partials: list[PartialPayloadRecord],
        *,
        mode: str,
        content_digest: str,
    ) -> SourceManifest:
        """Create a stable manifest from discovered logical data."""
        descriptors.sort(key=lambda descriptor: descriptor.name)
        partials.sort(key=lambda payload: (payload.payload_id, payload.is_input))
        url = self.engine.url.render_as_string(hide_password=True)
        source_identity = hashlib.sha256(f"{url}:{mode}".encode()).hexdigest()
        digest_payload = {
            "source_identity": source_identity,
            "mode": mode,
            "datasets": [
                {
                    "name": descriptor.name,
                    "columns": descriptor.column_names,
                    "aliases": descriptor.aliases,
                    "types": descriptor.column_types,
                    "shape": descriptor.shape,
                    "rows": descriptor.row_count,
                }
                for descriptor in descriptors
            ],
            "partials": [partial_payload_fingerprint(payload) for payload in partials],
            "content_digest": content_digest,
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return SourceManifest(
            source_type=self.source_type,
            source_identity=source_identity,
            digest=digest,
            total_files=1,
            datasets=tuple(descriptors),
            partial_payloads=tuple(partials),
        )

    async def _hash_current_rows(
        self,
        connection: AsyncConnection,
        table_name: str,
        table_idx: int,
        column_count: int,
        digest: _Digest,
    ) -> None:
        """Hash current-schema row blobs in bounded result pages."""
        columns = ", ".join(
            ["row_idx", *(f"column_{index}" for index in range(column_count))]
        )
        query = text(
            f"SELECT {columns} FROM `{table_name}` "  # noqa: S608 -- table name is an integer-derived internal identifier
            "ORDER BY row_idx LIMIT :page_size OFFSET :offset"
        )
        offset = 0
        while True:
            result = await connection.execute(
                query,
                {"page_size": _CONTENT_HASH_PAGE_SIZE, "offset": offset},
            )
            rows = result.fetchall()
            if not rows:
                return
            for row in rows:
                _hash_values(digest, "current", table_idx, *row)
            offset += len(rows)

    async def _hash_legacy_rows(
        self,
        connection: AsyncConnection,
        model_id: str,
        digest: _Digest,
    ) -> None:
        """Hash legacy row metadata and serialized values in bounded pages."""
        query = text(
            "SELECT DataframeRow.dbId, DataframeRow.timestamp, "
            "DataframeRow.modelId, DataframeRow.rowId, DataframeRow.tag, "
            "DataframeRow_Values.order_column, "
            "DataframeRow_Values.serializableObject "
            "FROM DataframeRow "
            "LEFT JOIN DataframeRow_Values "
            "ON DataframeRow.dbId = DataframeRow_Values.DataframeRow_dbId "
            "WHERE DataframeRow.modelId=:model_id "
            "ORDER BY DataframeRow.dbId, DataframeRow_Values.order_column "
            "LIMIT :page_size OFFSET :offset"
        )
        offset = 0
        while True:
            result = await connection.execute(
                query,
                {
                    "model_id": model_id,
                    "page_size": _CONTENT_HASH_PAGE_SIZE,
                    "offset": offset,
                },
            )
            rows = result.fetchall()
            if not rows:
                return
            for row in rows:
                _hash_values(digest, "legacy", model_id, *row)
            offset += len(rows)


def inspect_table_names(connection: Connection) -> list[str]:
    """Sync inspector callback kept module-level for type-checking."""
    return inspect(connection).get_table_names()


def _hash_values(digest: _Digest, *values: object) -> None:
    """Hash typed, length-delimited values without ambiguous concatenation."""
    for value in values:
        if value is None:
            encoded = b"N"
        elif isinstance(value, (bytes, bytearray, memoryview)):
            encoded = b"B" + bytes(value)
        else:
            encoded = b"S" + str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def _json_object(value: object) -> dict[str, object]:
    """Normalize a SQL JSON column returned as text or a mapping."""
    try:
        parsed = json.loads(value) if isinstance(value, str) else dict(value)
    except (TypeError, ValueError) as error:
        msg = "MariaDB dataset metadata is not a JSON object"
        raise MigrationValidationError(msg) from error
    if not isinstance(parsed, dict):
        msg = "MariaDB dataset metadata is not a JSON object"
        raise MigrationValidationError(msg)
    return parsed


def _string_list(value: object) -> list[str]:
    """Normalize a JSON list of names."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        msg = "MariaDB dataset metadata contains a non-list value"
        raise MigrationValidationError(msg)
    return [str(item) for item in value]


def _shape_from_metadata(
    metadata: dict[str, object], column_count: int
) -> tuple[int, ...]:
    """Return the non-row dimensions from a canonical metadata document."""
    raw_shape = metadata.get("shape")
    if raw_shape is None:
        return (column_count,)
    if isinstance(raw_shape, (str, bytes)) or not isinstance(raw_shape, (list, tuple)):
        msg = "MariaDB dataset metadata contains a non-list shape"
        raise MigrationValidationError(msg)
    try:
        shape = tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError) as error:
        msg = "MariaDB dataset metadata contains an invalid shape"
        raise MigrationValidationError(msg) from error
    if (
        len(shape) < _MIN_SHAPE_DIMENSIONS
        or shape[1] != column_count
        or any(value <= 0 for value in shape[1:])
    ):
        msg = "MariaDB dataset metadata contains an incompatible shape"
        raise MigrationValidationError(msg)
    return shape[1:]


def _is_input_flag(value: object) -> bool:
    """Normalize MariaDB boolean/bit representations."""
    return value in (True, 1, b"\x01", "1", "true", "True")


def _legacy_value(value: object) -> object:
    """Unwrap javaobj primitive wrappers while preserving rich values."""
    return value.value if hasattr(value, "value") else value


def _decompress_json(value: object) -> str:
    """Decode a gzip-compressed JSON cell from MariaDB."""
    try:
        return safe_gzip_decompress(bytes(value)).decode("utf-8")
    except (
        EOFError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        zlib.error,
    ) as error:
        msg = "unable to decode MariaDB compressed JSON cell"
        raise MigrationValidationError(msg) from error


def _current_table_name(table_idx: int) -> str:
    """Build a current-schema table name from a validated integer index."""
    return f"trustyai_v2_dataset_{table_idx}"
