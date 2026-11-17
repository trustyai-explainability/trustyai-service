"""Contract tests for legacy row metadata consumed after cutover."""

from __future__ import annotations

import datetime as dt

import pytest

from trustyai_service.service.data.migrations.errors import MigrationValidationError
from trustyai_service.service.data.migrations.row_metadata import legacy_metadata_row


@pytest.mark.parametrize("tag", ["organic", "", None])
@pytest.mark.parametrize(
    "timestamp",
    [
        "2024-02-03T04:05:06.123456",
        dt.datetime(2024, 2, 3, 4, 5, 6, 123456),  # noqa: DTZ001 -- Java LocalDateTime is timezone-naive
    ],
)
def test_legacy_metadata_preserves_ids_timestamps_and_tags(timestamp, tag):
    """SQL datetime values and CSV strings produce the same consumer metadata."""
    assert legacy_metadata_row(timestamp, "row-1", tag) == [
        "row-1",
        "2024-02-03T04:05:06.123456",
        1706933106.123456,
        ["organic"] if tag else [],
    ]


@pytest.mark.parametrize(
    ("timestamp", "row_id", "tag"),
    [
        (None, "id", "organic"),
        ("not-a-date", "id", "organic"),
        ("2024-02-03T04:05:06Z", "id", "organic"),
        (dt.datetime(2024, 2, 3, tzinfo=dt.UTC), "id", "organic"),
        ("2024-02-03T04:05:06", None, "organic"),
        ("2024-02-03T04:05:06", "id", ["organic"]),
    ],
)
def test_legacy_metadata_rejects_unsupported_values(timestamp, row_id, tag):
    """Malformed historical rows must not be misrepresented as valid metadata."""
    with pytest.raises(MigrationValidationError):
        legacy_metadata_row(timestamp, row_id, tag)
