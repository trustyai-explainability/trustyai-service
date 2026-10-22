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


def test_cli_accepts_csv_pvc_source() -> None:
    """The CSV source uses its default filename."""
    from trustyai_service.service.data.migrations.cli import parse_args

    args = parse_args(
        [
            "--source",
            "pvc-csv",
            "--source-path",
            "source",
            "--target-url",
            "sqlite+aiosqlite:///:memory:",
            "--quiesced",
        ]
    )

    assert args.source == "pvc-csv"
    assert args.source_path == Path("source")
    assert args.data_file == "data.csv"


def test_cli_allows_csv_filename_override(monkeypatch) -> None:
    """The Java source honors the mounted storage filename configuration."""
    from trustyai_service.service.data.migrations.cli import parse_args

    monkeypatch.setenv("STORAGE_DATA_FILENAME", "inference.csv")
    args = parse_args(
        [
            "--source",
            "pvc-csv",
            "--source-path",
            "source",
            "--target-url",
            "sqlite+aiosqlite:///:memory:",
            "--quiesced",
        ]
    )

    assert args.data_file == "inference.csv"


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


def test_cli_defaults_database_format_to_mariadb(monkeypatch) -> None:
    """DATABASE uses the service's MariaDB default when no kind is set."""
    from sqlalchemy.engine import make_url

    from trustyai_service.service.data.migrations.cli import target_url_from_environment

    for name in (
        "TRUSTYAI_MIGRATION_TARGET_URL",
        "QUARKUS_DATASOURCE_DB_KIND",
        "DATABASE_USERNAME",
        "DATABASE_PASSWORD",
        "DATABASE_HOST",
        "DATABASE_SERVICE",
        "DATABASE_DATABASE",
        "DATABASE_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERVICE_STORAGE_FORMAT", "DATABASE")
    monkeypatch.setenv("DATABASE_USERNAME", "trustyai")
    monkeypatch.setenv("DATABASE_PASSWORD", "secret")
    monkeypatch.setenv("DATABASE_HOST", "mariadb.example")
    monkeypatch.setenv("DATABASE_DATABASE", "trustyai")

    parsed = make_url(target_url_from_environment())
    assert parsed.drivername == "mariadb+asyncmy"
    assert parsed.username == "trustyai"
    assert parsed.host == "mariadb.example"
    assert parsed.port == 3306
    assert parsed.database == "trustyai"


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
