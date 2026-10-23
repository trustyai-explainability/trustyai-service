"""Command-line entry point for canonical storage migrations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import ssl
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

from trustyai_service.service.data.storage import (
    _insecure_tls_allowed,
    _resolve_tls_ca,
)

from .errors import MigrationError, error_detail
from .mariadb import MariaDBSource
from .pvc import PVCSource
from .pvc_csv import PVCCSVSource
from .runner import MigrationConfig, MigrationResult, MigrationRunner, MigrationStatus
from .target import DatabaseTarget

_MAX_NAMESPACE_LENGTH = 255
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine


def target_url_from_environment() -> str | None:
    """Build a migration target URL from the operator database contract."""
    explicit = os.environ.get("TRUSTYAI_MIGRATION_TARGET_URL")
    if explicit:
        return explicit

    database_kind = _configured_database_kind()
    if database_kind is None:
        return None
    if database_kind in {"maria", "mariadb"}:
        driver = "mariadb+asyncmy"
        default_port = 3306
    elif database_kind in {"postgres", "postgresql", "pgsql"}:
        driver = "postgresql+asyncpg"
        default_port = 5432
    elif database_kind == "sqlite":
        path = os.environ.get("STORAGE_DATABASE_PATH", ":memory:")
        if path == ":memory:":
            return "sqlite+aiosqlite:///:memory:"
        return URL.create("sqlite+aiosqlite", database=path).render_as_string(
            hide_password=False
        )
    else:
        return None

    return _build_network_database_url(driver, default_port)


def _configured_database_kind() -> str | None:
    """Normalize the operator's storage format and database kind."""
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "").strip().upper()
    if storage_format not in {
        "DATABASE",
        "MARIA",
        "MARIADB",
        "POSTGRES",
        "POSTGRESQL",
        "SQLITE",
    }:
        return None
    if storage_format in {"MARIA", "MARIADB"}:
        return "mariadb"
    if storage_format in {"POSTGRES", "POSTGRESQL"}:
        return "postgresql"
    if storage_format == "SQLITE":
        return "sqlite"
    return os.environ.get("QUARKUS_DATASOURCE_DB_KIND", "").strip().lower() or "mariadb"


def _build_network_database_url(driver: str, default_port: int) -> str:
    """Build a SQLAlchemy URL from operator Secret and Service values."""
    user = os.environ.get("DATABASE_USERNAME") or os.environ.get(
        "QUARKUS_DATASOURCE_USERNAME"
    )
    password = os.environ.get("DATABASE_PASSWORD") or os.environ.get(
        "QUARKUS_DATASOURCE_PASSWORD"
    )
    host = os.environ.get("DATABASE_HOST") or os.environ.get("DATABASE_SERVICE")
    database = os.environ.get("DATABASE_DATABASE") or os.environ.get("DATABASE_NAME")
    port = os.environ.get("DATABASE_PORT", str(default_port))
    missing = [
        name
        for name, value in (
            ("DATABASE_USERNAME or QUARKUS_DATASOURCE_USERNAME", user),
            ("DATABASE_PASSWORD or QUARKUS_DATASOURCE_PASSWORD", password),
            ("DATABASE_HOST or DATABASE_SERVICE", host),
            ("DATABASE_DATABASE or DATABASE_NAME", database),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "operator database configuration is incomplete: " + ", ".join(missing)
        )
    try:
        port_number = int(port)
    except ValueError as error:
        message = f"Invalid DATABASE_PORT={port!r}: must be an integer"
        raise ValueError(message) from error
    return URL.create(
        driver,
        username=user,
        password=password,
        host=host,
        port=port_number,
        database=database,
    ).render_as_string(hide_password=False)


def engine_options_from_environment(url: str) -> dict[str, object]:
    """Translate operator tuning and mounted TLS settings to async SQLAlchemy."""
    parsed = make_url(url)
    options = _pool_options(parsed.drivername)
    options["connect_args"] = _connect_args(parsed.drivername)
    return options


def _pool_options(drivername: str) -> dict[str, object]:
    """Return validated SQLAlchemy pool settings for non-SQLite drivers."""
    if drivername.startswith("sqlite"):
        return {}
    pool_size = _integer_env("DATABASE_POOL_SIZE", 5, minimum=1)
    max_overflow = _integer_env("DATABASE_MAX_OVERFLOW", 10, minimum=0)
    pool_timeout = _integer_env("DATABASE_POOL_TIMEOUT", 30, minimum=1)
    pool_recycle = _integer_env("DATABASE_POOL_RECYCLE", 1800, minimum=-1)
    budget = os.environ.get("DATABASE_MAX_CONNECTIONS_BUDGET")
    if budget is not None:
        budget_value = _integer_env("DATABASE_MAX_CONNECTIONS_BUDGET", 1, minimum=1)
        ceiling = pool_size + max_overflow
        if ceiling > budget_value:
            msg = (
                f"DATABASE_MAX_CONNECTIONS_BUDGET={budget_value}: configured pool "
                f"ceiling {ceiling} exceeds the per-pod budget"
            )
            raise ValueError(msg)
    return {
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_timeout": pool_timeout,
        "pool_recycle": pool_recycle,
        "pool_pre_ping": True,
    }


def _connect_args(drivername: str) -> dict[str, object]:
    """Return TLS and PostgreSQL session settings for one async driver."""
    args = _tls_connect_args(drivername)
    if drivername.startswith("postgresql+asyncpg"):
        args.update(_postgres_connect_args())
    elif drivername.startswith(("mariadb+", "mysql+")):
        application_name = os.environ.get("DATABASE_APPLICATION_NAME")
        if application_name is not None and "\x00" in application_name:
            message = "DATABASE_APPLICATION_NAME must not contain NUL characters"
            raise ValueError(message)
        if application_name is not None:
            args["program_name"] = application_name
    return args


def _tls_connect_args(drivername: str) -> dict[str, object]:
    """Build verified TLS arguments from the mounted CA contract."""
    if drivername.startswith("sqlite"):
        return {}
    ca_path, usable_ca = _resolve_tls_ca()
    ca_configured = "DATABASE_TLS_CA_CERT" in os.environ
    if usable_ca is None:
        if ca_configured or not _insecure_tls_allowed():
            message = f"database TLS requires a usable CA certificate at '{ca_path}'"
            raise ValueError(message)
        logger.warning(
            "Database CA is unavailable; migration connection has no configured CA verification. Mount a CA or set DATABASE_ALLOW_INSECURE_TLS=false to require one."
        )
        return {}
    if drivername.startswith("postgresql+asyncpg"):
        return {"ssl": ssl.create_default_context(cafile=ca_path)}
    if drivername.startswith(("mariadb+", "mysql+")):
        return {
            "ssl": {
                "ca": ca_path,
                "check_hostname": True,
                "verify_mode": ssl.CERT_REQUIRED,
            }
        }
    message = f"migration TLS configuration does not support driver '{drivername}'"
    raise ValueError(message)


def _validate_target_url(url: str, *, dry_run: bool) -> None:
    """Do not advertise transient SQLite data as a durable migration result."""
    parsed = make_url(url)
    if dry_run or parsed.get_backend_name() != "sqlite":
        return
    database = unquote(parsed.database or "")
    if (
        database in {"", ":memory:", "file:"}
        or database.startswith("file::memory:")
        or "memory" in parsed.normalized_query.get("mode", ())
        or "memdb" in parsed.normalized_query.get("vfs", ())
    ):
        msg = "migration target must be a persistent SQLite file, not an in-memory database"
        raise ValueError(msg)


def _postgres_connect_args() -> dict[str, object]:
    """Build asyncpg session settings from the operator ConfigMap."""
    application_name = os.environ.get("DATABASE_APPLICATION_NAME")
    if application_name is not None and "\x00" in application_name:
        message = "DATABASE_APPLICATION_NAME must not contain NUL characters"
        raise ValueError(message)
    statement_timeout = _optional_integer_env(
        "DATABASE_STATEMENT_TIMEOUT_MS", minimum=0
    )
    synchronous_commit = os.environ.get("DATABASE_SYNCHRONOUS_COMMIT", "on").lower()
    sync_value = {
        "1": "on",
        "true": "on",
        "yes": "on",
        "on": "on",
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
    }.get(synchronous_commit)
    if sync_value is None:
        message = "DATABASE_SYNCHRONOUS_COMMIT must be a boolean (true or false)"
        raise ValueError(message)
    server_settings: dict[str, str] = {"synchronous_commit": sync_value}
    if application_name is not None:
        server_settings["application_name"] = application_name
    if statement_timeout is not None:
        server_settings["statement_timeout"] = str(statement_timeout)
    return {"server_settings": server_settings}


def _integer_env(name: str, default: int, *, minimum: int) -> int:
    """Read one bounded integer from an operator ConfigMap value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        message = f"Invalid {name}={raw!r}: must be an integer"
        raise ValueError(message) from error
    if value < minimum:
        message = f"Invalid {name}={raw!r}: must be at least {minimum}"
        raise ValueError(message)
    return value


def _optional_integer_env(name: str, *, minimum: int) -> int | None:
    """Read an optional bounded integer."""
    if name not in os.environ:
        return None
    return _integer_env(name, 0, minimum=minimum)


def build_parser() -> argparse.ArgumentParser:
    """Build the migration CLI parser."""
    parser = argparse.ArgumentParser(
        description="Transfer TrustyAI storage through the canonical migration format."
    )
    parser.add_argument(
        "--source",
        choices=("pvc", "pvc-csv", "mariadb"),
        required=True,
        help="Source adapter to read.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=(
            Path(os.environ["STORAGE_DATA_FOLDER"])
            if os.environ.get("STORAGE_DATA_FOLDER")
            else None
        ),
        help=(
            "PVC directory containing TrustyAI HDF5 or CSV files; may be "
            "supplied through STORAGE_DATA_FOLDER."
        ),
    )
    parser.add_argument(
        "--source-url",
        default=os.environ.get("TRUSTYAI_MIGRATION_SOURCE_URL"),
        help="Async SQLAlchemy URL for a MariaDB source; may be supplied through TRUSTYAI_MIGRATION_SOURCE_URL.",
    )
    parser.add_argument(
        "--source-schema",
        choices=("auto", "legacy", "trustyai_v2"),
        default="auto",
        help="MariaDB schema to read; default: auto.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="PVC data filename; defaults to STORAGE_DATA_FILENAME, trustyai.hdf5 for pvc, or data.csv for pvc-csv.",
    )
    parser.add_argument(
        "--target-url",
        default=os.environ.get("TRUSTYAI_MIGRATION_TARGET_URL"),
        help="Async SQLAlchemy URL for the destination database; may be supplied through TRUSTYAI_MIGRATION_TARGET_URL or the operator database environment.",
    )
    parser.add_argument(
        "--namespace",
        default="migration",
        help="Destination namespace for migration tables; default: migration.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Maximum logical rows per transfer batch; default: 10000.",
    )
    parser.add_argument(
        "--quiesced",
        action="store_true",
        help="Confirm that the source is quiesced or snapshotted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and decode the source without writing the target.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Compatibility flag: incomplete runs replay whole datasets; "
            "completed migrations are always audit-only."
        ),
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip the source re-hash during post-write verification.",
    )
    return parser


def _default_data_file(source: str, configured: str | None) -> str:
    """Resolve source-specific defaults while honoring the mounted config."""
    if configured:
        return configured
    return "data.csv" if source == "pvc-csv" else "trustyai.hdf5"


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:  # noqa: C901 -- boundary validations report precise CLI errors
    """Parse and validate command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(arguments)
    args.data_file = _default_data_file(
        args.source,
        args.data_file
        if args.data_file is not None
        else os.environ.get("STORAGE_DATA_FILENAME"),
    )
    if not args.quiesced:
        parser.error("--quiesced is required before a source can be read")
    if not args.target_url:
        try:
            args.target_url = target_url_from_environment()
        except ValueError as error:
            parser.error(str(error))
    if not args.target_url:
        parser.error(
            "--target-url, TRUSTYAI_MIGRATION_TARGET_URL, or a complete operator "
            "database environment is required"
        )
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if not args.namespace or len(args.namespace) > _MAX_NAMESPACE_LENGTH:
        parser.error("--namespace must contain 1 to 255 characters")
    if args.source in {"pvc", "pvc-csv"} and args.source_path is None:
        parser.error(f"--source-path is required for --source {args.source}")
    if args.source in {"pvc", "pvc-csv"} and (
        not args.data_file or Path(args.data_file).name != args.data_file
    ):
        parser.error("--data-file must be a non-empty file name")
    if args.source == "mariadb" and not args.source_url:
        parser.error("--source-url is required for --source mariadb")
    try:
        _validate_target_url(args.target_url, dry_run=args.dry_run)
    except ValueError as error:
        parser.error(str(error))
    return args


async def run_from_args(args: argparse.Namespace) -> MigrationResult:
    """Run one migration described by parsed CLI arguments."""
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

    _validate_target_url(args.target_url, dry_run=args.dry_run)
    source_engine: AsyncEngine | None = None
    target_engine = create_async_engine(
        args.target_url,
        **engine_options_from_environment(args.target_url),
    )
    try:
        if args.source == "pvc":
            source = PVCSource(
                args.source_path,
                quiesced=args.quiesced,
                data_file=args.data_file,
            )
        elif args.source == "pvc-csv":
            source = PVCCSVSource(
                args.source_path,
                quiesced=args.quiesced,
                data_file=args.data_file,
            )
        else:
            source_engine = create_async_engine(
                args.source_url,
                **engine_options_from_environment(args.source_url),
            )
            source = MariaDBSource(
                source_engine,
                schema=args.source_schema,
                quiesced=args.quiesced,
            )
        target = DatabaseTarget(
            target_engine,
            migration_namespace=args.namespace,
        )
        runner = MigrationRunner(
            source=source,
            target=target,
            config=MigrationConfig(
                batch_size=args.batch_size,
                resume=not args.no_resume,
                verify_checksums=not args.no_checksums,
                dry_run=args.dry_run,
            ),
        )
        return await runner.run()
    finally:
        if source_engine is not None:
            await source_engine.dispose()
        await target_engine.dispose()


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the migration CLI and return a process exit code."""
    args = parse_args(arguments)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = asyncio.run(run_from_args(args))
    except (MigrationError, ValueError, OSError, SQLAlchemyError) as error:
        # Driver exceptions can contain credentials, SQL and bound inference
        # values. Report their class, not their potentially sensitive payload.
        detail = error_detail(error)
        sys.stderr.write(f"migration failed: {detail}\n")
        result = MigrationResult(status=MigrationStatus.FAILED, error=detail)
        exit_code = 2
    else:
        exit_code = (
            0
            if result.status in {MigrationStatus.COMPLETE, MigrationStatus.DRY_RUN}
            else 1
        )
    sys.stdout.write(
        json.dumps(asdict(result), default=_json_default, sort_keys=True) + "\n"
    )
    return exit_code


def _json_default(value: object) -> object:
    """Serialize enum values in the result JSON."""
    if isinstance(value, MigrationStatus):
        return value.value
    msg = f"cannot serialize {type(value).__name__}"
    raise TypeError(msg)


if __name__ == "__main__":  # pragma: no cover - exercised through the script
    raise SystemExit(main())
