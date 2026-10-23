"""Reader for the CSV-on-PVC storage layout and metadata sidecars."""

from __future__ import annotations

import asyncio
import base64
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import struct
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .contracts import (
    DatasetDescriptor,
    MigrationSource,
    PartialPayloadRecord,
    SourceManifest,
)
from .errors import MigrationPreflightError, MigrationValidationError
from .java_values import tensor_value

if TYPE_CHECKING:
    from collections.abc import Iterator

_DATA_FILE = "data.csv"
_INTERNAL_FILE = "internal_data.csv"
_METADATA_FILE = "metadata.json"
_INTERNAL_COLUMN_COUNT = 3
# Large enough for serialized tensors, but not an unbounded process-wide limit.
_MAX_CSV_FIELD_SIZE = 16 * 1024 * 1024
_CSV_LIMIT_LOCK = threading.RLock()
_INTEGER_BOUNDS = {
    "INT32": (-(2**31), 2**31 - 1),
    "INT64": (-(2**63), 2**63 - 1),
}


@contextmanager
def _bounded_csv_fields() -> Iterator[None]:
    """Scope the stdlib's global CSV limit to one serialized parsing pass."""
    with _CSV_LIMIT_LOCK:
        original = csv.field_size_limit()
        try:
            csv.field_size_limit(_MAX_CSV_FIELD_SIZE)
            yield
        finally:
            csv.field_size_limit(original)


class PVCCSVSource(MigrationSource):
    """Read per-model CSV data and metadata sidecars from a PVC."""

    source_type = "PVC_CSV"

    def __init__(
        self,
        directory: str | Path,
        *,
        quiesced: bool,
        data_file: str = _DATA_FILE,
    ) -> None:
        """Initialize a source directory and Java data filename."""
        if not data_file or Path(data_file).name != data_file or "\x00" in data_file:
            msg = "data_file must be a non-empty file name"
            raise ValueError(msg)
        self.directory = Path(directory)
        self.quiesced = quiesced
        self.data_file = data_file
        self._indices_by_dataset: dict[str, tuple[int, ...]] = {}
        self._types_by_dataset: dict[str, tuple[str, ...]] = {}
        # Text seek cookies preserve CSV record boundaries, including embedded
        # newlines. Keep only one cursor per dataset, not one offset per row.
        self._read_cursors: dict[str, tuple[int, int]] = {}

    async def preflight(self) -> SourceManifest:
        """Validate Java sidecars and return a content-addressed manifest."""
        if not self.directory.exists() or not self.directory.is_dir():
            msg = f"source directory does not exist: {self.directory}"
            raise MigrationPreflightError(msg)
        try:
            manifest = await asyncio.to_thread(self._build_manifest)
        except OSError as error:
            msg = f"unable to inspect CSV source directory: {error}"
            raise MigrationPreflightError(msg) from error
        return manifest

    async def read_rows(
        self, dataset: DatasetDescriptor, start_row: int, n_rows: int
    ) -> np.ndarray:
        """Read a bounded logical batch and convert Java cell encodings."""
        if start_row < 0 or n_rows < 0:
            msg = "row offsets and limits must be non-negative"
            raise ValueError(msg)
        if dataset.source_file is None:
            msg = f"CSV dataset '{dataset.name}' has no source file"
            raise MigrationValidationError(msg)
        indices = self._indices_by_dataset.get(dataset.name)
        types = self._types_by_dataset.get(dataset.name)
        if indices is None or types is None:
            msg = f"CSV dataset '{dataset.name}' was not discovered by preflight"
            raise MigrationValidationError(msg)
        return await asyncio.to_thread(
            self._read_rows_sync,
            dataset,
            indices,
            types,
            start_row,
            n_rows,
        )

    async def partial_payloads(self) -> tuple[PartialPayloadRecord, ...]:
        """CSV PVC storage has no separate partial-payload representation."""
        return ()

    @_bounded_csv_fields()
    def _build_manifest(self) -> SourceManifest:
        """Discover models, validate row pairs, and hash every source file."""
        self._indices_by_dataset.clear()
        self._types_by_dataset.clear()
        self._read_cursors.clear()
        data_paths = self._data_paths()
        if not data_paths:
            msg = f"no CSV data files ending in '-{self.data_file}' found"
            raise MigrationValidationError(msg)

        descriptors: list[DatasetDescriptor] = []
        file_hashes: dict[str, str] = {}
        for data_path in data_paths:
            model_id = data_path.name[: -(len(self.data_file) + 1)]
            if not model_id:
                msg = f"invalid model ID in CSV filename: {data_path.name}"
                raise MigrationValidationError(msg)
            internal_path = self._required_file(model_id, _INTERNAL_FILE)
            metadata_path = self._required_file(model_id, _METADATA_FILE)
            metadata = self._read_model_metadata(metadata_path, model_id)
            input_fields = self._schema_fields(metadata, "inputSchema", model_id)
            output_fields = self._schema_fields(metadata, "outputSchema", model_id)
            all_fields = [*input_fields, *output_fields]
            self._validate_column_indices(all_fields, model_id)
            row_count = self._validate_row_files(
                data_path,
                internal_path,
                all_fields,
                model_id,
            )
            observations = metadata.get("observations")
            if (
                isinstance(observations, bool)
                or not isinstance(observations, int)
                or observations != row_count
            ):
                msg = (
                    f"Java metadata observation count for '{model_id}' does not "
                    f"match CSV row count ({observations!r} != {row_count})"
                )
                raise MigrationValidationError(msg)

            service_metadata = {
                "input_tensor_name": self._metadata_string(
                    metadata, "inputTensorName", "input", model_id
                ),
                "output_tensor_name": self._metadata_string(
                    metadata, "outputTensorName", "output", model_id
                ),
                "recorded_inferences": self._metadata_boolean(
                    metadata, "recordedInferences", model_id, default=False
                ),
            }
            descriptors.extend(
                self._descriptors_for_schema(
                    model_id,
                    data_path.name,
                    "inputs",
                    input_fields,
                    row_count,
                    service_metadata,
                )
            )
            descriptors.extend(
                self._descriptors_for_schema(
                    model_id,
                    data_path.name,
                    "outputs",
                    output_fields,
                    row_count,
                    {},
                )
            )
            metadata_name = f"{model_id}_metadata"
            metadata_descriptor = DatasetDescriptor(
                name=metadata_name,
                column_names=("timestamp", "rowId", "tag"),
                aliases=("timestamp", "rowId", "tag"),
                column_types=("UNKNOWN", "STRING", "STRING"),
                row_count=row_count,
                shape=(3,),
                source_file=data_path.name,
            )
            descriptors.append(metadata_descriptor)
            self._indices_by_dataset[metadata_name] = ()
            self._types_by_dataset[metadata_name] = ()

            for path in (data_path, internal_path, metadata_path):
                file_hashes[path.name] = self._hash_file(path)

        # A successful subset must never authorize deleting a PVC containing
        # models written under another historical filename or storage format.
        for path in self.directory.iterdir():
            if path.name not in file_hashes and (
                path.suffix.lower() in {".csv", ".hdf5", ".h5"}
                or path.name.endswith(f"-{_METADATA_FILE}")
            ):
                msg = f"unselected storage file '{path.name}'; isolate the complete source snapshot or correct --data-file"
                raise MigrationValidationError(msg)
        descriptors.sort(key=lambda dataset: dataset.name)
        source_identity = hashlib.sha256(
            str(self.directory.resolve()).encode("utf-8")
        ).hexdigest()
        digest_payload = {
            "source_identity": source_identity,
            "files": file_hashes,
            "datasets": [asdict(dataset) for dataset in descriptors],
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return SourceManifest(
            source_type=self.source_type,
            source_identity=source_identity,
            digest=digest,
            total_files=len(file_hashes),
            datasets=tuple(descriptors),
        )

    def _data_paths(self) -> list[Path]:
        """Select only exact Java data-file suffix matches in this directory."""
        suffix = f"-{self.data_file}"
        paths: list[Path] = []
        for path in self.directory.iterdir():
            if not path.name.endswith(suffix):
                continue
            if path.is_symlink():
                msg = f"PVC CSV source file must not be a symlink: {path.name}"
                raise MigrationValidationError(msg)
            if path.is_file():
                paths.append(path)
        return sorted(paths)

    def _required_file(self, model_id: str, suffix: str) -> Path:
        """Resolve and validate a required per-model sidecar."""
        path = self.directory / f"{model_id}-{suffix}"
        if path.is_symlink():
            msg = f"PVC CSV source file must not be a symlink: {path.name}"
            raise MigrationValidationError(msg)
        if path.parent != self.directory or not path.is_file():
            msg = f"missing CSV source file: {path.name}"
            raise MigrationValidationError(msg)
        return path

    @staticmethod
    def _read_model_metadata(path: Path, model_id: str) -> dict[str, Any]:
        """Read strict JSON without accepting duplicate keys or non-finite values."""
        try:
            document = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            msg = f"invalid Java metadata file '{path.name}': {error}"
            raise MigrationValidationError(msg) from error
        if not isinstance(document, dict):
            msg = f"Java metadata for '{model_id}' must be a JSON object"
            raise MigrationValidationError(msg)
        stored_model_id = document.get("modelId")
        if stored_model_id is not None and stored_model_id != model_id:
            msg = f"Java metadata model ID does not match filename for '{model_id}'"
            raise MigrationValidationError(msg)
        return document

    @staticmethod
    def _schema_fields(
        metadata: dict[str, Any], schema_name: str, model_id: str
    ) -> list[dict[str, Any]]:
        """Normalize one Java Schema to ordered, validated source fields."""
        schema = metadata.get(schema_name)
        if not isinstance(schema, dict) or not isinstance(schema.get("items"), dict):
            msg = f"Java metadata for '{model_id}' has invalid {schema_name}.items"
            raise MigrationValidationError(msg)
        raw_items = schema["items"]
        if not raw_items:
            msg = f"Java metadata for '{model_id}' has an empty {schema_name}"
            raise MigrationValidationError(msg)

        mappings = schema.get("nameMapping", {})
        if not isinstance(mappings, dict):
            msg = (
                f"Java metadata for '{model_id}' has invalid {schema_name}.nameMapping"
            )
            raise MigrationValidationError(msg)
        fields: list[dict[str, Any]] = []
        for raw_name, item in raw_items.items():
            if (
                not isinstance(raw_name, str)
                or not raw_name
                or not isinstance(item, dict)
            ):
                msg = f"Java metadata for '{model_id}' has an invalid schema item"
                raise MigrationValidationError(msg)
            name = item.get("name", raw_name)
            if name != raw_name:
                msg = f"Java schema item name does not match key '{raw_name}'"
                raise MigrationValidationError(msg)
            data_type = item.get("type")
            if not isinstance(data_type, str) or data_type not in {
                "BOOL",
                "FLOAT",
                "DOUBLE",
                "INT32",
                "INT64",
                "STRING",
                "TENSOR",
                "MAP",
                "UNKNOWN",
            }:
                msg = f"Java schema item '{raw_name}' has an unsupported type"
                raise MigrationValidationError(msg)
            index = _column_index(item, raw_name)
            alias = mappings.get(raw_name, raw_name)
            if not isinstance(alias, str) or not alias:
                msg = f"Java schema alias for '{raw_name}' must be a non-empty string"
                raise MigrationValidationError(msg)
            fields.append(
                {"name": raw_name, "alias": alias, "type": data_type, "index": index}
            )
        aliases = [field["alias"] for field in fields]
        if len(aliases) != len(set(aliases)):
            msg = f"Java metadata for '{model_id}' has duplicate {schema_name} aliases"
            raise MigrationValidationError(msg)
        return sorted(fields, key=lambda field: field["index"])

    @staticmethod
    def _validate_column_indices(fields: list[dict[str, Any]], model_id: str) -> None:
        """Require schemas to partition every physical CSV column exactly once."""
        indices = [field["index"] for field in fields]
        if sorted(indices) != list(range(len(fields))):
            msg = (
                f"Java metadata for '{model_id}' has duplicate, missing, or invalid "
                "column index values"
            )
            raise MigrationValidationError(msg)

    def _validate_row_files(
        self,
        data_path: Path,
        internal_path: Path,
        fields: list[dict[str, Any]],
        model_id: str,
    ) -> int:
        """Validate each CSV row and require exact data/metadata alignment."""
        count = 0
        try:
            with (
                data_path.open(encoding="utf-8-sig", newline="") as data_file,
                internal_path.open(encoding="utf-8-sig", newline="") as internal_file,
            ):
                data_rows = csv.reader(data_file, strict=True)
                internal_rows = csv.reader(internal_file, strict=True)
                for count, pair in enumerate(
                    itertools.zip_longest(data_rows, internal_rows), start=1
                ):
                    data_row, internal_row = pair
                    _validate_row_pair(data_row, internal_row, fields, model_id, count)
        except MigrationValidationError:
            raise
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as error:
            msg = f"unable to validate CSV rows for '{model_id}': {error}"
            raise MigrationValidationError(msg) from error
        return count

    def _descriptors_for_schema(  # noqa: PLR0913 -- descriptor projects schema and source context
        self,
        model_id: str,
        source_file: str,
        kind: str,
        fields: list[dict[str, Any]],
        row_count: int,
        service_metadata: dict[str, Any],
    ) -> list[DatasetDescriptor]:
        """Create one logical model descriptor and remember its physical columns."""
        indices = tuple(field["index"] for field in fields)
        types = tuple(field["type"] for field in fields)
        name = f"{model_id}_{kind}"
        self._indices_by_dataset[name] = indices
        self._types_by_dataset[name] = types
        return [
            DatasetDescriptor(
                name=name,
                column_names=tuple(field["name"] for field in fields),
                aliases=tuple(field["alias"] for field in fields),
                column_types=types,
                row_count=row_count,
                shape=(len(fields),),
                source_file=source_file,
                service_metadata=service_metadata,
            )
        ]

    @_bounded_csv_fields()
    def _read_rows_sync(
        self,
        dataset: DatasetDescriptor,
        indices: tuple[int, ...],
        types: tuple[str, ...],
        start_row: int,
        n_rows: int,
    ) -> np.ndarray:
        """Decode one logical dataset batch from the paired source CSV files."""
        suffix = next(
            (
                f"_{kind}"
                for kind in ("inputs", "outputs", "metadata")
                if dataset.name.endswith(f"_{kind}")
            ),
            None,
        )
        if suffix is None:
            msg = f"CSV dataset '{dataset.name}' has an invalid logical name"
            raise MigrationValidationError(msg)
        model_id = dataset.name[: -len(suffix)]
        expected_filename = f"{model_id}-{self.data_file}"
        if dataset.source_file != expected_filename:
            msg = f"CSV dataset '{dataset.name}' references an unexpected source file"
            raise MigrationValidationError(msg)
        data_path = self.directory / dataset.source_file
        internal_path = self.directory / f"{model_id}-{_INTERNAL_FILE}"
        if data_path.is_symlink() or internal_path.is_symlink():
            msg = f"CSV source files for '{model_id}' must not be symlinks"
            raise MigrationValidationError(msg)

        result = np.empty((n_rows, len(dataset.column_names)), dtype=object)
        try:
            is_metadata = dataset.name.endswith("_metadata")
            path = internal_path if is_metadata else data_path
            with path.open(encoding="utf-8-sig", newline="") as stream:
                cursor = self._read_cursors.get(dataset.name)
                skip = start_row
                if cursor is not None and cursor[0] == start_row:
                    stream.seek(cursor[1])
                    skip = 0
                # Iterating TextIOWrapper directly disables tell(); readline
                # iteration keeps valid seek cookies after whole CSV records.
                reader = csv.reader(iter(stream.readline, ""), strict=True)
                rows = _read_batch_rows(
                    reader,
                    indices,
                    types,
                    model_id,
                    skip,
                    n_rows,
                    is_metadata=is_metadata,
                )
                for offset, row in enumerate(rows):
                    for column, value in enumerate(row):
                        result[offset, column] = value
                self._read_cursors[dataset.name] = (
                    start_row + len(rows),
                    stream.tell(),
                )
        except MigrationValidationError:
            raise
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as error:
            msg = f"unable to read CSV dataset '{dataset.name}': {error}"
            raise MigrationValidationError(msg) from error
        if len(rows) != n_rows:
            msg = f"CSV dataset '{dataset.name}' ended before requested batch"
            raise MigrationValidationError(msg)
        return result

    @staticmethod
    def _metadata_string(
        metadata: dict[str, Any], key: str, default: str, model_id: str
    ) -> str:
        value = metadata.get(key, default)
        if not isinstance(value, str) or not value:
            msg = f"Java metadata field '{key}' for '{model_id}' must be a non-empty string"
            raise MigrationValidationError(msg)
        return value

    @staticmethod
    def _metadata_boolean(
        metadata: dict[str, Any], key: str, model_id: str, *, default: bool
    ) -> bool:
        value = metadata.get(key, default)
        if not isinstance(value, bool):
            msg = f"Java metadata field '{key}' for '{model_id}' must be a boolean"
            raise MigrationValidationError(msg)
        return value

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Hash a source file with bounded memory use."""
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            msg = f"unable to hash CSV source file '{path.name}': {error}"
            raise MigrationValidationError(msg) from error
        return digest.hexdigest()


def _validate_row_pair(
    data_row: list[str] | None,
    internal_row: list[str] | None,
    fields: list[dict[str, Any]],
    model_id: str,
    row_number: int,
) -> None:
    """Validate paired Java rows and decode cells during source preflight."""
    if data_row is None or internal_row is None:
        msg = f"CSV row count mismatch for '{model_id}'"
        raise MigrationValidationError(msg)
    if len(data_row) != len(fields):
        msg = (
            f"Java data row {row_number} for '{model_id}' has {len(data_row)} "
            f"columns; expected {len(fields)}"
        )
        raise MigrationValidationError(msg)
    if len(internal_row) != _INTERNAL_COLUMN_COUNT:
        msg = (
            f"Java internal data row {row_number} for '{model_id}' must have "
            f"{_INTERNAL_COLUMN_COUNT} columns"
        )
        raise MigrationValidationError(msg)
    for field in fields:
        _parse_value(data_row[field["index"]], field["type"], model_id)
    _parse_internal_row(internal_row, model_id, row_number)


def _read_batch_rows(  # noqa: PLR0913 -- bounded projection needs its source selection
    reader: Iterator[list[str]],
    indices: tuple[int, ...],
    types: tuple[str, ...],
    model_id: str,
    start_row: int,
    n_rows: int,
    *,
    is_metadata: bool,
) -> list[list[object] | tuple[str, str, str]]:
    """Decode a bounded page from either the data or internal CSV file."""
    rows = itertools.islice(reader, start_row, start_row + n_rows)
    decoded: list[list[object] | tuple[str, str, str]] = []
    for row_number, row in enumerate(rows, start=start_row + 1):
        if is_metadata:
            decoded.append(_parse_internal_row(row, model_id, row_number))
            continue
        if max(indices, default=-1) >= len(row):
            msg = f"Java data row {row_number} for '{model_id}' is incomplete"
            raise MigrationValidationError(msg)
        decoded.append(
            [
                _parse_value(row[index], data_type, model_id)
                for index, data_type in zip(indices, types, strict=True)
            ]
        )
    return decoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated JSON keys rather than silently selecting one value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate JSON key: {key}"
            raise ValueError(msg)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject non-standard NaN and Infinity JSON tokens."""
    msg = f"invalid JSON numeric constant: {value}"
    raise ValueError(msg)


def _column_index(item: dict[str, Any], name: str) -> int:
    """Read the current Java field name or its legacy JSON alias."""
    index = item.get("columnIndex", item.get("index"))
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or ("columnIndex" in item and "index" in item and item["index"] != index)
    ):
        msg = f"Java schema item '{name}' has an invalid column index"
        raise MigrationValidationError(msg)
    return index


def _parse_value(value: str, data_type: str, model_id: str) -> object:
    """Convert one CSV cell without coercing malformed values."""
    if data_type == "BOOL":
        normalized = value.lower()
        if normalized not in {"true", "false"}:
            msg = f"invalid BOOL value in CSV row for '{model_id}'"
            raise MigrationValidationError(msg)
        return normalized == "true"
    if data_type in {"FLOAT", "DOUBLE"}:
        return _parse_float(value, data_type, model_id)
    if data_type in _INTEGER_BOUNDS:
        return _parse_integer(value, data_type, model_id)
    if data_type == "TENSOR":
        return _decode_tensor(value)
    return value


def _parse_float(value: str, data_type: str, model_id: str) -> float:
    """Parse Java floating-point text with Java FLOAT precision when needed."""
    try:
        number = float(value)
    except ValueError as error:
        msg = f"invalid {data_type} value in CSV row for '{model_id}'"
        raise MigrationValidationError(msg) from error
    if data_type == "FLOAT":
        try:
            number = struct.unpack("!f", struct.pack("!f", number))[0]
        except (OverflowError, struct.error) as error:
            if not math.isfinite(number):
                msg = f"invalid FLOAT value in CSV row for '{model_id}'"
                raise MigrationValidationError(msg) from error
            number = math.copysign(math.inf, number)
    return number


def _parse_integer(value: str, data_type: str, model_id: str) -> int:
    """Parse Java integer text and enforce its declared width."""
    if value != value.strip():
        msg = f"invalid {data_type} value in CSV row for '{model_id}'"
        raise MigrationValidationError(msg)
    try:
        number = int(value)
    except ValueError as error:
        msg = f"invalid {data_type} value in CSV row for '{model_id}'"
        raise MigrationValidationError(msg) from error
    lower, upper = _INTEGER_BOUNDS[data_type]
    if not lower <= number <= upper:
        msg = f"invalid {data_type} value in CSV row for '{model_id}'"
        raise MigrationValidationError(msg)
    return number


def _decode_tensor(value: str) -> object:
    """Safely parse Java Tensor object-stream structure without Java deserialization."""
    try:
        import javaobj  # noqa: PLC0415 -- required only when CSV contains tensors
    except ModuleNotFoundError as error:
        msg = "Tensor migration requires the 'migration' extra"
        raise MigrationPreflightError(msg) from error
    try:
        encoded = base64.b64decode(value, validate=True)
        tensor = javaobj.loads(encoded)
    except Exception as error:
        msg = "invalid Java-serialized Tensor cell"
        raise MigrationValidationError(msg) from error

    return tensor_value(tensor)


def _parse_internal_row(
    row: list[str], model_id: str, row_number: int
) -> tuple[str, str, str]:
    """Convert Java's tag,id,local-timestamp sidecar row."""
    if len(row) != _INTERNAL_COLUMN_COUNT:
        msg = (
            f"Java internal data row {row_number} for '{model_id}' must have "
            f"{_INTERNAL_COLUMN_COUNT} columns"
        )
        raise MigrationValidationError(msg)
    try:
        parsed_timestamp = dt.datetime.fromisoformat(row[2])
    except ValueError as error:
        msg = (
            f"invalid timestamp in Java internal data row {row_number} for '{model_id}'"
        )
        raise MigrationValidationError(msg) from error
    if parsed_timestamp.tzinfo is not None:
        msg = f"Java internal timestamp for '{model_id}' must be local time without an offset"
        raise MigrationValidationError(msg)
    return row[2], row[1], row[0]
