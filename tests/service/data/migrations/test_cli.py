"""Tests for the migration command-line boundary."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_cli_requires_quiescence() -> None:
    """The CLI must make the source consistency acknowledgement explicit."""
    from trustyai_service.service.data.migrations.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--source",
                "pvc",
                "--source-path",
                "source",
                "--target-url",
                "sqlite+aiosqlite:///:memory:",
            ]
        )


def test_cli_accepts_pvc_to_database_arguments() -> None:
    """PVC source arguments map to the reusable runner configuration."""
    from trustyai_service.service.data.migrations.cli import parse_args

    args = parse_args(
        [
            "--source",
            "pvc",
            "--source-path",
            "source",
            "--target-url",
            "sqlite+aiosqlite:///:memory:",
            "--batch-size",
            "17",
            "--namespace",
            "cutover",
            "--quiesced",
            "--dry-run",
        ]
    )

    assert args.source == "pvc"
    assert args.source_path.name == "source"
    assert args.batch_size == 17
    assert args.namespace == "cutover"
    assert args.dry_run is True


def test_cli_accepts_operator_pvc_environment(monkeypatch) -> None:
    """PVC migration defaults match the operator's migration deployment env."""
    from trustyai_service.service.data.migrations.cli import parse_args

    monkeypatch.setenv("STORAGE_DATA_FOLDER", "/var/lib/trustyai")
    monkeypatch.setenv("STORAGE_DATA_FILENAME", "legacy.hdf5")

    args = parse_args(
        [
            "--source",
            "pvc",
            "--target-url",
            "sqlite+aiosqlite:///:memory:",
            "--quiesced",
        ]
    )

    assert args.source_path == Path("/var/lib/trustyai")
    assert args.data_file == "legacy.hdf5"


def test_cli_accepts_mariadb_source_arguments() -> None:
    """MariaDB source arguments expose schema selection without importing a driver."""
    from trustyai_service.service.data.migrations.cli import parse_args

    args = parse_args(
        [
            "--source",
            "mariadb",
            "--source-url",
            "mysql+asyncmy://user@localhost/trustyai",
            "--target-url",
            "postgresql+asyncpg://user@localhost/trustyai",
            "--source-schema",
            "legacy",
            "--quiesced",
        ]
    )

    assert args.source == "mariadb"
    assert args.source_schema == "legacy"


def test_cli_accepts_database_urls_from_environment(monkeypatch) -> None:
    """Database URLs can be injected without shell arguments."""
    from trustyai_service.service.data.migrations.cli import parse_args

    monkeypatch.setenv(
        "TRUSTYAI_MIGRATION_SOURCE_URL",
        "mysql+asyncmy://user@localhost/trustyai",
    )
    monkeypatch.setenv(
        "TRUSTYAI_MIGRATION_TARGET_URL",
        "postgresql+asyncpg://user@localhost/trustyai",
    )

    args = parse_args(["--source", "mariadb", "--quiesced"])

    assert args.source_url.endswith("/trustyai")
    assert args.target_url.endswith("/trustyai")


def test_cli_rejects_non_positive_batch_size() -> None:
    """Invalid transfer bounds fail at argument parsing."""
    from trustyai_service.service.data.migrations.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--source",
                "pvc",
                "--source-path",
                "source",
                "--target-url",
                "sqlite+aiosqlite:///:memory:",
                "--batch-size",
                "0",
                "--quiesced",
            ]
        )
