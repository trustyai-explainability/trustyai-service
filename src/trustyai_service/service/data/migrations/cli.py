"""Command-line entry point for canonical storage migrations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import MigrationError
from .mariadb import MariaDBSource
from .pvc import PVCSource
from .runner import MigrationConfig, MigrationResult, MigrationRunner, MigrationStatus
from .target import DatabaseTarget

_MAX_NAMESPACE_LENGTH = 255

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncEngine


def build_parser() -> argparse.ArgumentParser:
    """Build the migration CLI parser."""
    parser = argparse.ArgumentParser(
        description="Transfer TrustyAI storage through the canonical migration format."
    )
    parser.add_argument(
        "--source",
        choices=("pvc", "mariadb"),
        required=True,
        help="Source adapter to read.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        help="PVC directory containing TrustyAI HDF5 files.",
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
        default="trustyai.hdf5",
        help="PVC data-file suffix; default: trustyai.hdf5.",
    )
    parser.add_argument(
        "--target-url",
        default=os.environ.get("TRUSTYAI_MIGRATION_TARGET_URL"),
        help="Async SQLAlchemy URL for the destination database; may be supplied through TRUSTYAI_MIGRATION_TARGET_URL.",
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
        help="Do not reuse completed dataset checkpoints.",
    )
    parser.add_argument(
        "--no-checksums",
        action="store_true",
        help="Skip the source re-hash during post-write verification.",
    )
    return parser


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(arguments)
    if not args.quiesced:
        parser.error("--quiesced is required before a source can be read")
    if not args.target_url:
        parser.error("--target-url or TRUSTYAI_MIGRATION_TARGET_URL is required")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if not args.namespace or len(args.namespace) > _MAX_NAMESPACE_LENGTH:
        parser.error("--namespace must contain 1 to 255 characters")
    if args.source == "pvc" and args.source_path is None:
        parser.error("--source-path is required for --source pvc")
    if args.source == "pvc" and (
        not args.data_file or Path(args.data_file).name != args.data_file
    ):
        parser.error("--data-file must be a non-empty file name")
    if args.source == "mariadb" and not args.source_url:
        parser.error("--source-url is required for --source mariadb")
    return args


async def run_from_args(args: argparse.Namespace) -> MigrationResult:
    """Run one migration described by parsed CLI arguments."""
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

    source_engine: AsyncEngine | None = None
    target_engine = create_async_engine(args.target_url)
    try:
        if args.source == "pvc":
            source = PVCSource(
                args.source_path,
                quiesced=args.quiesced,
                data_file=args.data_file,
            )
        else:
            source_engine = create_async_engine(args.source_url)
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
    try:
        result = asyncio.run(run_from_args(args))
    except MigrationError as error:
        sys.stderr.write(f"migration failed: {error}\n")
        return 2
    sys.stdout.write(
        json.dumps(asdict(result), default=_json_default, sort_keys=True) + "\n"
    )
    return (
        0 if result.status in {MigrationStatus.COMPLETE, MigrationStatus.DRY_RUN} else 1
    )


def _json_default(value: object) -> object:
    """Serialize enum values in the result JSON."""
    if isinstance(value, MigrationStatus):
        return value.value
    msg = f"cannot serialize {type(value).__name__}"
    raise TypeError(msg)


if __name__ == "__main__":  # pragma: no cover - exercised through the script
    raise SystemExit(main())
