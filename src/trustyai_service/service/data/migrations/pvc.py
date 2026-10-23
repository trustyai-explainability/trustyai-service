"""PVC/HDF5 source adapter for the canonical migration contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from trustyai_service.endpoints.consumer import (
    KServeInferenceRequest,
    KServeInferenceResponse,
)
from trustyai_service.service.constants import PROTECTED_DATASET_SUFFIX
from trustyai_service.service.data.storage.pvc import (
    BYTES_ATTRIBUTE,
    COLUMN_ALIAS_ATTRIBUTE,
    COLUMN_NAMES_ATTRIBUTE,
    COLUMN_TYPES_ATTRIBUTE,
    PARTIAL_INPUT_NAME,
    PARTIAL_OUTPUT_NAME,
)
from trustyai_service.service.serialization import deserialize_rows

from .contracts import (
    DatasetDescriptor,
    MigrationSource,
    PartialPayloadRecord,
    SourceManifest,
    decode_partial_payload,
    partial_payload_fingerprint,
)
from .errors import MigrationPreflightError, MigrationValidationError


class PVCSource(MigrationSource):
    """Read one quiesced PVC directory without modifying it."""

    source_type = "PVC"

    def __init__(
        self,
        directory: str | Path,
        *,
        quiesced: bool,
        data_file: str = "trustyai.hdf5",
    ) -> None:
        """Initialize an HDF5 source directory."""
        if not data_file or Path(data_file).name != data_file or "\x00" in data_file:
            msg = "data_file must be a non-empty file name"
            raise ValueError(msg)
        self.directory = Path(directory)
        self.quiesced = quiesced
        self.data_file = data_file
        self._last_manifest: SourceManifest | None = None

    async def preflight(self) -> SourceManifest:
        """Validate the directory and return its current content manifest."""
        if not self.directory.exists() or not self.directory.is_dir():
            msg = f"source directory does not exist: {self.directory}"
            raise MigrationPreflightError(msg)
        manifest = await asyncio.to_thread(self._build_manifest)
        self._last_manifest = manifest
        return manifest

    async def read_rows(
        self, dataset: DatasetDescriptor, start_row: int, n_rows: int
    ) -> np.ndarray:
        """Read and decode one bounded HDF5 row batch."""
        if start_row < 0 or n_rows < 0:
            msg = "row offsets and limits must be non-negative"
            raise ValueError(msg)
        if dataset.source_file is None:
            msg = f"PVC dataset '{dataset.name}' has no source file"
            raise MigrationValidationError(msg)
        return await asyncio.to_thread(
            self._read_rows_sync,
            dataset,
            start_row,
            n_rows,
        )

    async def partial_payloads(self) -> tuple[PartialPayloadRecord, ...]:
        """Return partial payloads from the current PVC snapshot."""
        manifest = self._last_manifest or await self.preflight()
        return manifest.partial_payloads

    def _build_manifest(self) -> SourceManifest:
        """Build and validate a source manifest in a worker thread."""
        paths = self._source_paths()
        datasets: list[DatasetDescriptor] = []
        partials: list[PartialPayloadRecord] = []
        file_hashes: dict[str, str] = {}
        for path in paths:
            file_hashes[path.name] = self._hash_file(path)
            dataset_name = self._dataset_name(path)
            if dataset_name in {PARTIAL_INPUT_NAME, PARTIAL_OUTPUT_NAME}:
                partials.extend(self._read_partial_file(path, dataset_name))
                continue
            if dataset_name.startswith(PROTECTED_DATASET_SUFFIX):
                continue
            datasets.append(self._describe_dataset(path, dataset_name))

        datasets.sort(key=lambda dataset: dataset.name)
        partials.sort(key=lambda payload: (payload.payload_id, payload.is_input))
        source_identity = hashlib.sha256(
            str(self.directory.resolve()).encode("utf-8")
        ).hexdigest()
        digest_payload = {
            "source_identity": source_identity,
            "files": file_hashes,
            "datasets": [asdict(dataset) for dataset in datasets],
            "partials": [partial_payload_fingerprint(payload) for payload in partials],
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return SourceManifest(
            source_type=self.source_type,
            source_identity=source_identity,
            digest=digest,
            total_files=len(paths),
            datasets=tuple(datasets),
            partial_payloads=tuple(partials),
        )

    def _source_paths(self) -> list[Path]:
        """Return only files owned by PVCStorage's one-file-per-dataset layout."""
        suffix = f"_{self.data_file}"
        paths = []
        for path in self.directory.iterdir():
            if not path.name.endswith(suffix):
                if path.suffix.lower() in {".hdf5", ".h5", ".csv"} or (
                    not path.is_symlink() and path.is_file() and h5py.is_hdf5(path)
                ):
                    msg = f"unselected storage file '{path.name}'; isolate the complete source snapshot or correct --data-file"
                    raise MigrationValidationError(msg)
                continue
            if path.parent != self.directory:
                msg = f"PVC file is outside the source directory: {path.name}"
                raise MigrationValidationError(msg)
            if path.is_symlink():
                msg = f"PVC source file must not be a symlink: {path.name}"
                raise MigrationValidationError(msg)
            if path.is_file():
                paths.append(path)
        return sorted(paths)

    def _dataset_name(self, path: Path) -> str:
        """Derive the logical dataset name from a PVC file name."""
        suffix = f"_{self.data_file}"
        if not path.name.endswith(suffix):
            msg = f"unrecognized PVC file: {path.name}"
            raise MigrationValidationError(msg)
        return path.name[: -len(suffix)]

    def _describe_dataset(self, path: Path, dataset_name: str) -> DatasetDescriptor:
        """Read and validate logical dataset metadata."""
        try:
            with h5py.File(path, "r") as h5_file:
                if dataset_name not in h5_file:
                    msg = f"PVC file '{path.name}' does not contain '{dataset_name}'"
                    raise MigrationValidationError(msg)
                dataset = h5_file[dataset_name]
                column_names = self._string_sequence(
                    dataset.attrs.get(COLUMN_NAMES_ATTRIBUTE),
                    attribute=COLUMN_NAMES_ATTRIBUTE,
                    dataset_name=dataset_name,
                )
                aliases = self._string_sequence(
                    dataset.attrs.get(COLUMN_ALIAS_ATTRIBUTE, column_names),
                    attribute=COLUMN_ALIAS_ATTRIBUTE,
                    dataset_name=dataset_name,
                )
                column_types = self._string_sequence(
                    dataset.attrs.get(
                        COLUMN_TYPES_ATTRIBUTE,
                        ["UNKNOWN"] * len(column_names),
                    ),
                    attribute=COLUMN_TYPES_ATTRIBUTE,
                    dataset_name=dataset_name,
                )
                if len(aliases) != len(column_names):
                    msg = f"PVC dataset '{dataset_name}' has mismatched column aliases"
                    raise MigrationValidationError(msg)
                if len(column_types) != len(column_names):
                    msg = f"PVC dataset '{dataset_name}' has mismatched column types"
                    raise MigrationValidationError(msg)
                serialized = bool(dataset.attrs.get(BYTES_ATTRIBUTE, False))
                shape = self._logical_shape(
                    dataset,
                    column_count=len(column_names),
                    serialized=serialized,
                    dataset_name=dataset_name,
                )
                return DatasetDescriptor(
                    name=dataset_name,
                    column_names=tuple(column_names),
                    aliases=tuple(aliases),
                    column_types=tuple(column_types),
                    row_count=int(dataset.shape[0]),
                    shape=shape,
                    serialized=serialized,
                    source_file=path.name,
                )
        except (OSError, ValueError) as error:
            msg = f"unable to read PVC dataset '{dataset_name}': {error}"
            raise MigrationValidationError(msg) from error

    @staticmethod
    def _logical_shape(
        dataset: h5py.Dataset,
        *,
        column_count: int,
        serialized: bool,
        dataset_name: str,
    ) -> tuple[int, ...]:
        """Return logical row dimensions, including serialized HDF5 rows."""
        if not serialized:
            return tuple(int(value) for value in dataset.shape[1:])
        if dataset.shape[0] == 0:
            return (column_count,)
        try:
            decoded = deserialize_rows(dataset[0:1])
        except ValueError:
            # Keep preflight non-destructive for corrupt rows. The normal
            # bounded read path reports the corruption as an unsafe migration.
            return (column_count,)
        shape = tuple(int(value) for value in decoded.shape[1:])
        if not shape or shape[0] != column_count or any(value <= 0 for value in shape):
            msg = f"PVC dataset '{dataset_name}' has an incompatible serialized shape"
            raise MigrationValidationError(msg)
        return shape

    def _read_rows_sync(
        self,
        descriptor: DatasetDescriptor,
        start_row: int,
        n_rows: int,
    ) -> np.ndarray:
        """Read rows synchronously for ``asyncio.to_thread``."""
        source_file = str(descriptor.source_file)
        if Path(source_file).name != source_file:
            msg = f"PVC dataset '{descriptor.name}' has an invalid source file"
            raise MigrationValidationError(msg)
        path = self.directory / source_file
        if path.is_symlink():
            msg = f"PVC source file must not be a symlink: {source_file}"
            raise MigrationValidationError(msg)
        try:
            if path.resolve(strict=True).parent != self.directory.resolve(strict=True):
                msg = f"PVC source file is outside the source directory: {source_file}"
                raise MigrationValidationError(msg)
        except OSError as error:
            msg = f"unable to resolve PVC source file '{source_file}'"
            raise MigrationValidationError(msg) from error
        with h5py.File(path, "r") as h5_file:
            raw = h5_file[descriptor.name][start_row : start_row + n_rows]
        if descriptor.serialized:
            if len(raw) == 0:
                return np.empty((0, len(descriptor.column_names)), dtype=object)
            try:
                return deserialize_rows(raw)
            except ValueError as error:
                msg = f"unable to decode PVC dataset '{descriptor.name}' at row {start_row}"
                raise MigrationValidationError(msg) from error
        return np.asarray(raw)

    def _read_partial_file(
        self,
        path: Path,
        dataset_name: str,
    ) -> list[PartialPayloadRecord]:
        """Decode persisted KServe and ModelMesh partial payload attributes."""
        kserve_class = (
            KServeInferenceRequest
            if dataset_name == PARTIAL_INPUT_NAME
            else KServeInferenceResponse
        )
        records: list[PartialPayloadRecord] = []
        with h5py.File(path, "r") as h5_file:
            if dataset_name not in h5_file:
                msg = f"PVC partial-payload file '{path.name}' is malformed"
                raise MigrationValidationError(msg)
            dataset = h5_file[dataset_name]
            for payload_id in dataset.attrs:
                raw = dataset.attrs[payload_id]
                try:
                    payload, is_modelmesh = decode_partial_payload(
                        self._bytes(raw), kserve_class
                    )
                except Exception as error:
                    msg = f"unable to decode partial payload '{payload_id}'"
                    raise MigrationValidationError(msg) from error
                records.append(
                    PartialPayloadRecord(
                        payload_id=str(payload_id),
                        is_input=dataset_name == PARTIAL_INPUT_NAME,
                        is_modelmesh=is_modelmesh,
                        payload=payload,
                    )
                )
        return records

    @staticmethod
    def _bytes(value: object) -> bytes:
        """Normalize HDF5 attribute scalar types to bytes."""
        if isinstance(value, bytes):
            return value
        if isinstance(value, np.void):
            return bytes(value)
        if isinstance(value, np.ndarray):
            return value.tobytes()
        if isinstance(value, (bytearray, memoryview)):
            return bytes(value)
        msg = f"unsupported HDF5 byte attribute type: {type(value).__name__}"
        raise TypeError(msg)

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Hash one source file without loading the whole file into memory."""
        digest = hashlib.sha256()
        try:
            with path.open("rb") as file:
                while chunk := file.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as error:
            msg = f"unable to hash PVC source file '{path.name}'"
            raise MigrationValidationError(msg) from error
        return digest.hexdigest()

    @staticmethod
    def _string_sequence(
        value: object,
        *,
        attribute: str,
        dataset_name: str,
    ) -> list[str]:
        """Normalize HDF5 string attributes and reject missing metadata."""
        if value is None:
            msg = (
                f"PVC dataset '{dataset_name}' is missing '{attribute}' column metadata"
            )
            raise MigrationValidationError(msg)
        values = value.tolist() if isinstance(value, np.ndarray) else value
        if isinstance(values, (str, bytes)):
            values = [values]
        return [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in values
        ]
