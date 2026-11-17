"""Convert legacy row metadata to the service's ingestion schema."""

from __future__ import annotations

import datetime as dt

from .errors import MigrationValidationError

METADATA_COLUMNS = ("id", "iso_time", "unix_timestamp", "tags")
METADATA_TYPES = ("STRING", "STRING", "DOUBLE", "UNKNOWN")


def legacy_metadata_row(timestamp: object, row_id: object, tag: object) -> list[object]:
    """Preserve timestamp text and interpret Java local time as UTC for epoch seconds."""
    if (
        not isinstance(timestamp, (str, dt.datetime))
        or not isinstance(row_id, str)
        or (tag is not None and not isinstance(tag, str))
    ):
        msg = (
            "invalid legacy row metadata: expected a local timestamp and string ID/tag"
        )
        raise MigrationValidationError(msg)
    iso_time = (
        timestamp.isoformat() if isinstance(timestamp, dt.datetime) else timestamp
    )
    try:
        parsed = dt.datetime.fromisoformat(iso_time)
    except ValueError as error:
        msg = "invalid legacy row metadata timestamp"
        raise MigrationValidationError(msg) from error
    if parsed.tzinfo is not None:
        msg = "legacy timestamp must be local time without an offset"
        raise MigrationValidationError(msg)
    unix_timestamp = parsed.replace(tzinfo=dt.UTC).timestamp()
    return [row_id, iso_time, unix_timestamp, [tag] if tag else []]
