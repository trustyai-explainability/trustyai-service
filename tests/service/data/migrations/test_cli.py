"""Tests for the migration command-line boundary."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [
        ("COMPLETE", 0),
        ("DRY_RUN", 0),
        ("PARTIAL", 1),
        ("FAILED", 1),
        ("IN_PROGRESS", 1),
    ],
)
def test_main_status_and_json_contract(monkeypatch, capsys, status, exit_code):
    """Only verified completion or a dry-run returns zero to automation."""
    import json

    from trustyai_service.service.data.migrations import cli

    monkeypatch.setattr(cli, "parse_args", lambda _arguments: object())

    async def run(_args):
        return cli.MigrationResult(
            status=cli.MigrationStatus(status),
            safe_to_remove_source=status == "COMPLETE",
        )

    monkeypatch.setattr(cli, "run_from_args", run)
    assert cli.main([]) == exit_code
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == status
    assert result["safe_to_remove_source"] is (status == "COMPLETE")


def test_main_database_failure_produces_safe_json(monkeypatch, capsys):
    """Driver errors have a nonzero exit code without exposing SQL parameters."""
    import json

    from sqlalchemy.exc import OperationalError

    from trustyai_service.service.data.migrations import cli

    monkeypatch.setattr(cli, "parse_args", lambda _arguments: object())

    async def run(_args):
        statement = "SELECT secret"
        # Synthetic credentials exercise error redaction, not authentication.
        raise OperationalError(
            statement,
            {"password": "sensitive"},  # pragma: allowlist secret
            Exception("sensitive"),
        )

    monkeypatch.setattr(cli, "run_from_args", run)
    assert cli.main([]) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "FAILED"
    assert result["safe_to_remove_source"] is False
    assert "sensitive" not in captured.out + captured.err
    assert "Traceback" not in captured.err


def test_empty_source_never_authorizes_removal(tmp_path, capsys):
    """An empty or incorrectly selected snapshot is not a successful cutover."""
    import json

    from trustyai_service.service.data.migrations.cli import main

    target = tmp_path / "target.sqlite"
    assert (
        main(
            [
                "--source",
                "pvc",
                "--source-path",
                str(tmp_path),
                "--target-url",
                f"sqlite+aiosqlite:///{target}",
                "--quiesced",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["safe_to_remove_source"] is False
    assert result["status"] == "FAILED"
    assert not target.exists()


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


def test_cli_accepts_compatibility_and_checksum_flags():
    """Operational flags are parsed explicitly rather than silently ignored."""
    from trustyai_service.service.data.migrations.cli import parse_args

    args = parse_args(
        [
            "--source",
            "pvc",
            "--source-path",
            "source",
            "--target-url",
            "sqlite+aiosqlite:///target.sqlite",
            "--quiesced",
            "--no-resume",
            "--no-checksums",
        ]
    )
    assert args.no_resume is True
    assert args.no_checksums is True


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
            "sqlite+aiosqlite:///cutover.sqlite",
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
            "sqlite+aiosqlite:///cutover.sqlite",
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
            "sqlite+aiosqlite:///cutover.sqlite",
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


@pytest.mark.parametrize(
    "url",
    [
        "sqlite+aiosqlite://",
        "sqlite+aiosqlite:///",
        "sqlite+aiosqlite:///:memory:",
        "sqlite+aiosqlite:///file::memory:?uri=true",
        "sqlite+aiosqlite:///file:%3Amemory%3A?uri=true",
        "sqlite+aiosqlite:///file:?uri=true",
        "sqlite+aiosqlite:///file:cutover?vfs=memdb&uri=true",
        "sqlite+aiosqlite:///file:cutover?mode=memory&cache=shared&uri=true",
        "sqlite+aiosqlite:///file:cutover?mode=rw&mode=memory&uri=true",
    ],
)
def test_cli_rejects_transient_targets_except_for_dry_run(url) -> None:
    """An ephemeral database must never be offered as a durable cutover target."""
    from trustyai_service.service.data.migrations.cli import parse_args

    arguments = [
        "--source",
        "pvc",
        "--source-path",
        "source",
        "--target-url",
        url,
        "--quiesced",
    ]
    with pytest.raises(SystemExit):
        parse_args(arguments)
    assert parse_args([*arguments, "--dry-run"]).dry_run


@pytest.mark.parametrize("insecure", ["false", "0", "off", "invalid"])
def test_cli_requires_default_ca_when_insecure_tls_is_prohibited(
    monkeypatch, tmp_path, insecure
) -> None:
    """Strict TLS cannot fall back when the default CA mount is absent."""
    from trustyai_service.service.data import storage
    from trustyai_service.service.data.migrations.cli import (
        engine_options_from_environment,
    )

    monkeypatch.setattr(storage, "DEFAULT_TLS_CA_CERT", str(tmp_path / "missing.crt"))
    monkeypatch.delenv("DATABASE_TLS_CA_CERT", raising=False)
    monkeypatch.setenv("DATABASE_ALLOW_INSECURE_TLS", insecure)
    with pytest.raises(ValueError, match="usable CA"):
        engine_options_from_environment("mariadb+asyncmy://user@localhost/db")


def test_cli_rejects_explicit_missing_ca_even_with_insecure_opt_out(
    monkeypatch, tmp_path
) -> None:
    """An explicitly configured CA is never silently ignored."""
    from trustyai_service.service.data.migrations.cli import (
        engine_options_from_environment,
    )

    monkeypatch.setenv("DATABASE_TLS_CA_CERT", str(tmp_path / "missing.crt"))
    monkeypatch.setenv("DATABASE_ALLOW_INSECURE_TLS", "true")
    with pytest.raises(ValueError, match="usable CA"):
        engine_options_from_environment("postgresql+asyncpg://user@localhost/db")


def test_sqlite_cli_does_not_require_network_tls(monkeypatch) -> None:
    """Network certificate policy must not prevent local SQLite migrations."""
    from trustyai_service.service.data.migrations.cli import (
        engine_options_from_environment,
    )

    monkeypatch.setenv("DATABASE_TLS_CA_CERT", "/missing/certificate.crt")
    monkeypatch.setenv("DATABASE_ALLOW_INSECURE_TLS", "false")
    assert engine_options_from_environment("sqlite+aiosqlite:///cutover.db") == {
        "connect_args": {}
    }


def test_cli_warns_when_ca_verification_is_unavailable(monkeypatch, tmp_path, caplog):
    """The compatibility fallback must never silently drop CA verification."""
    from trustyai_service.service.data import storage
    from trustyai_service.service.data.migrations.cli import (
        engine_options_from_environment,
    )

    monkeypatch.setattr(storage, "DEFAULT_TLS_CA_CERT", str(tmp_path / "missing.crt"))
    monkeypatch.delenv("DATABASE_TLS_CA_CERT", raising=False)
    monkeypatch.setenv("DATABASE_ALLOW_INSECURE_TLS", "true")
    # Synthetic credentials verify that warnings do not expose connection URLs.
    engine_options_from_environment(
        "postgresql+asyncpg://user:password@db/test"  # pragma: allowlist secret
    )
    assert "no configured CA verification" in caplog.text
    assert "password" not in caplog.text


def test_cli_rejects_ignoring_a_ca_for_an_unsupported_driver(monkeypatch):
    """A configured CA must never be ignored for a driver without TLS wiring."""
    from trustyai_service.service.data.migrations import cli

    monkeypatch.setattr(cli, "_resolve_tls_ca", lambda: ("/mounted/ca", "/mounted/ca"))
    with pytest.raises(ValueError, match="does not support driver"):
        cli.engine_options_from_environment("postgresql+psycopg://user@db/test")


def test_cli_requires_persistent_sqlite_path_from_environment(monkeypatch) -> None:
    """SQLITE without a configured path cannot silently choose a transient target."""
    from trustyai_service.service.data.migrations.cli import parse_args

    monkeypatch.setenv("SERVICE_STORAGE_FORMAT", "SQLITE")
    monkeypatch.delenv("STORAGE_DATABASE_PATH", raising=False)
    monkeypatch.delenv("TRUSTYAI_MIGRATION_TARGET_URL", raising=False)
    with pytest.raises(SystemExit):
        parse_args(["--source", "pvc", "--source-path", "source", "--quiesced"])


@pytest.mark.asyncio
async def test_cli_file_target_survives_engine_disposal(pvc_fixture, tmp_path) -> None:
    """The CLI result is checked against a new production reader after exit."""
    from trustyai_service.service.data.migrations.cli import parse_args, run_from_args
    from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage

    path = tmp_path / "persistent.sqlite"
    args = parse_args(
        [
            "--source",
            "pvc",
            "--source-path",
            str(pvc_fixture.directory),
            "--target-url",
            f"sqlite+aiosqlite:///{path}",
            "--quiesced",
        ]
    )
    result = await run_from_args(args)
    assert result.safe_to_remove_source is True
    reader = SQLiteStorage(str(path))
    try:
        assert (await reader.read_data(pvc_fixture.datasets[0].name)).tolist() == [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    finally:
        reader._engine.dispose()
