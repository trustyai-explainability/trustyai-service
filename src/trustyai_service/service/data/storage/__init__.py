"""Storage backend implementations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trustyai_service.service.data.storage.maria.maria import MariaDBStorage
    from trustyai_service.service.data.storage.postgres.postgres import (
        PostgreSQLStorage,
    )
    from trustyai_service.service.data.storage.sqlite.sqlite import SQLiteStorage

from trustyai_service.service.data.storage.pvc import PVCStorage

# Default location the operator mounts the database CA certificate at.
DEFAULT_TLS_CA_CERT = "/etc/tls/db/ca.crt"

# Optional opt-out of the default no-CA connection behavior. When a CA file is
# available, database connections verify the server certificate.
INSECURE_TLS_ENV_VAR = "DATABASE_ALLOW_INSECURE_TLS"

_TRUTHY = ("1", "true", "yes", "on")


def _insecure_tls_allowed() -> bool:
    """Return True unless the deployment explicitly requires an available CA."""
    value = os.environ.get(INSECURE_TLS_ENV_VAR)
    return value is None or value.lower() in _TRUTHY


def _resolve_tls_ca() -> tuple[str, str | None]:
    """Return the configured CA path and the same path only if it is a real file.

    An empty ``DATABASE_TLS_CA_CERT`` resolves to the current directory, and a
    directory is not a usable CA bundle, so neither is handed to the driver.
    Validation separately rejects an unusable path when the variable was
    explicitly configured and insecure connections were not explicitly allowed.
    """
    ca_path = os.environ.get("DATABASE_TLS_CA_CERT", DEFAULT_TLS_CA_CERT)
    usable = bool(ca_path) and Path(ca_path).is_file()
    return ca_path, ca_path if usable else None


def _tls_error(backend: str, ca_path: str) -> str:
    """Build the error message for a configured but unusable CA certificate."""
    return (
        f"{backend} storage requires a usable CA certificate at '{ca_path}'. Mount "
        f"a valid CA certificate there or unset DATABASE_TLS_CA_CERT. A configured "
        f"CA path will not fall back to an insecure connection."
    )


class GlobalStorageInterface:
    """Singleton holder for global storage interface."""

    _instance: (
        MariaDBStorage | PVCStorage | PostgreSQLStorage | SQLiteStorage | None
    ) = None

    @classmethod
    def get(
        cls, *, force_reload: bool = False
    ) -> MariaDBStorage | PVCStorage | PostgreSQLStorage | SQLiteStorage:
        """Get or create the global storage interface singleton.

        :param force_reload: If True, force recreation of the storage interface
        :return: Storage interface instance (PVCStorage, MariaDBStorage, PostgreSQLStorage, or SQLiteStorage)
        """
        if cls._instance is None or force_reload:
            cls._instance = get_storage_interface()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None


def get_global_storage_interface(
    *, force_reload: bool = False
) -> MariaDBStorage | PVCStorage | PostgreSQLStorage | SQLiteStorage:
    """Get or create the global storage interface singleton.

    :param force_reload: If True, force recreation of the storage interface
    :return: Storage interface instance (PVCStorage, MariaDBStorage, PostgreSQLStorage, or SQLiteStorage)
    """
    return GlobalStorageInterface.get(force_reload=force_reload)


class MariaDBConfig:
    """MariaDB connection configuration read from environment variables.

    Supports both operator (Quarkus) and direct deployment env vars.
    """

    def __init__(self) -> None:
        """Read MariaDB connection parameters from environment variables."""
        self.user = os.environ.get("DATABASE_USERNAME") or os.environ.get(
            "QUARKUS_DATASOURCE_USERNAME"
        )
        self.password = os.environ.get("DATABASE_PASSWORD") or os.environ.get(
            "QUARKUS_DATASOURCE_PASSWORD"
        )
        self.host = os.environ.get("DATABASE_HOST") or os.environ.get(
            "DATABASE_SERVICE"
        )
        self.database = os.environ.get("DATABASE_DATABASE") or os.environ.get(
            "DATABASE_NAME"
        )
        port_str = os.environ.get("DATABASE_PORT", "3306")
        try:
            self.port = int(port_str)
        except ValueError as e:
            msg = f"Invalid DATABASE_PORT value '{port_str}': must be a valid integer"
            raise ValueError(msg) from e

        self.ssl_ca_path, self.ssl_ca = _resolve_tls_ca()
        self.tls_ca_configured = "DATABASE_TLS_CA_CERT" in os.environ
        self.allow_insecure_tls = _insecure_tls_allowed()

    def validate(self) -> None:
        """Raise ValueError if required env vars are missing or TLS is unusable."""
        from trustyai_service.service.data.storage.sql.engine import (  # noqa: PLC0415 -- SQL dependencies are optional
            pool_kwargs_from_env,
        )

        missing = []
        if not self.user:
            missing.append("DATABASE_USERNAME or QUARKUS_DATASOURCE_USERNAME")
        if not self.password:
            missing.append("DATABASE_PASSWORD or QUARKUS_DATASOURCE_PASSWORD")
        if not self.host:
            missing.append("DATABASE_HOST or DATABASE_SERVICE")
        if not self.database:
            missing.append("DATABASE_DATABASE or DATABASE_NAME")
        if missing:
            msg = (
                f"MariaDB storage requires environment variables: {', '.join(missing)}"
            )
            raise ValueError(msg)
        # Fail readiness on malformed pool values or a per-pod connection budget
        # that the configured pool could exceed.
        pool_kwargs_from_env()
        if self.ssl_ca is None and (
            not self.allow_insecure_tls or self.tls_ca_configured
        ):
            raise ValueError(_tls_error("MariaDB", self.ssl_ca_path))


class PostgreSQLConfig:
    """PostgreSQL connection configuration read from environment variables.

    Supports both operator (Quarkus) and direct deployment env vars. Env vars
    are identical to MariaDB's except that DATABASE_PORT defaults to 5432.
    """

    def __init__(self) -> None:
        """Read PostgreSQL connection parameters from environment variables."""
        self.user = os.environ.get("DATABASE_USERNAME") or os.environ.get(
            "QUARKUS_DATASOURCE_USERNAME"
        )
        self.password = os.environ.get("DATABASE_PASSWORD") or os.environ.get(
            "QUARKUS_DATASOURCE_PASSWORD"
        )
        self.host = os.environ.get("DATABASE_HOST") or os.environ.get(
            "DATABASE_SERVICE"
        )
        self.database = os.environ.get("DATABASE_DATABASE") or os.environ.get(
            "DATABASE_NAME"
        )
        port_str = os.environ.get("DATABASE_PORT", "5432")
        try:
            self.port = int(port_str)
        except ValueError as e:
            msg = f"Invalid DATABASE_PORT value '{port_str}': must be a valid integer"
            raise ValueError(msg) from e

        self.ssl_ca_path, self.ssl_ca = _resolve_tls_ca()
        self.tls_ca_configured = "DATABASE_TLS_CA_CERT" in os.environ
        self.allow_insecure_tls = _insecure_tls_allowed()

    def validate(self) -> None:
        """Raise ValueError if required env vars are missing or TLS is unusable."""
        from trustyai_service.service.data.storage.sql.engine import (  # noqa: PLC0415 -- SQL dependencies are optional
            pool_kwargs_from_env,
            postgres_tuning_from_env,
        )

        missing = []
        if not self.user:
            missing.append("DATABASE_USERNAME or QUARKUS_DATASOURCE_USERNAME")
        if not self.password:
            missing.append("DATABASE_PASSWORD or QUARKUS_DATASOURCE_PASSWORD")
        if not self.host:
            missing.append("DATABASE_HOST or DATABASE_SERVICE")
        if not self.database:
            missing.append("DATABASE_DATABASE or DATABASE_NAME")
        if missing:
            msg = (
                "PostgreSQL storage requires environment variables: "
                f"{', '.join(missing)}"
            )
            raise ValueError(msg)
        pool_kwargs_from_env()
        postgres_tuning_from_env()
        if self.ssl_ca is None and (
            not self.allow_insecure_tls or self.tls_ca_configured
        ):
            raise ValueError(_tls_error("PostgreSQL", self.ssl_ca_path))


def _database_storage_format() -> str:
    """Map the configured database kind to its storage format."""
    database_kind = os.environ.get("QUARKUS_DATASOURCE_DB_KIND", "mariadb")
    normalized_kind = database_kind.strip().lower()
    if normalized_kind in ("mariadb", "maria"):
        return "MARIA"
    if normalized_kind in ("postgresql", "postgres", "pgsql"):
        return "POSTGRESQL"
    if normalized_kind == "sqlite":
        return "SQLITE"
    msg = (
        f"Database kind={database_kind} not supported for "
        "storage format=DATABASE. "
        "Supported kinds are mariadb, postgresql, and sqlite."
    )
    raise ValueError(msg)


def get_storage_interface() -> (
    MariaDBStorage | PVCStorage | PostgreSQLStorage | SQLiteStorage
):
    """Create a new storage interface based on environment configuration.

    :return: Storage interface instance (PVCStorage, MariaDBStorage, PostgreSQLStorage, or SQLiteStorage)
    :raises ValueError: If storage format is unsupported or dependencies missing
    """
    storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC")
    if storage_format == "DATABASE":
        storage_format = _database_storage_format()
    if storage_format == "PVC":
        return PVCStorage(
            data_directory=os.environ.get("STORAGE_DATA_FOLDER", "/tmp"),  # noqa: S108 -- fallback default for STORAGE_DATA_FOLDER env var
            data_file=os.environ.get("STORAGE_DATA_FILENAME", "trustyai.hdf5"),
        )
    if storage_format == "SQLITE":
        try:
            # Import SQLite storage only when needed (optional dependency: sqlalchemy)
            from trustyai_service.service.data.storage.sqlite.sqlite import (  # noqa: PLC0415 -- lazy import: sqlalchemy is optional
                SQLiteStorage,
            )

            # STORAGE_DATABASE_PATH may be ":memory:" or a filesystem path.
            return SQLiteStorage(
                path=os.environ.get("STORAGE_DATABASE_PATH", ":memory:")
            )
        except ImportError as e:
            msg = (
                "SQLite storage requires optional dependencies. "
                "Install with: pip install trustyai-service[sqlite]. "
                f"Error: {e}"
            )
            raise ValueError(msg) from e
    if storage_format in ("MARIA", "MARIADB"):
        try:
            # Import MariaDB storage only when needed (optional dependency)
            from trustyai_service.service.data.storage.maria.maria import (  # noqa: PLC0415 -- lazy import: mariadb is optional
                MariaDBStorage,
            )

            migration_str = os.environ.get("DATABASE_ATTEMPT_MIGRATION", "0").lower()
            attempt_migration = migration_str in ("1", "true", "yes", "on")

            config = MariaDBConfig()
            config.validate()

            return MariaDBStorage(
                user=config.user,
                password=config.password,
                host=config.host,
                port=config.port,
                database=config.database,
                ssl_ca=config.ssl_ca,
                attempt_migration=attempt_migration,
            )
        except ImportError as e:
            msg = (
                "MariaDB storage requires optional dependencies. "
                "Install with: pip install trustyai-service[mariadb]. "
                f"Error: {e}"
            )
            raise ValueError(msg) from e
    if storage_format in ("POSTGRESQL", "POSTGRES"):
        try:
            # Import PostgreSQL storage only when needed (optional dependency)
            from trustyai_service.service.data.storage.postgres.postgres import (  # noqa: PLC0415 -- lazy import: psycopg is optional
                PostgreSQLStorage,
            )

            config = PostgreSQLConfig()
            config.validate()

            return PostgreSQLStorage(
                user=config.user,
                password=config.password,
                host=config.host,
                port=config.port,
                database=config.database,
                ssl_ca=config.ssl_ca,
            )
        except ImportError as e:
            msg = (
                "PostgreSQL storage requires optional dependencies. "
                "Install with: pip install trustyai-service[postgres]. "
                f"Error: {e}"
            )
            raise ValueError(msg) from e
    else:
        msg = f"Storage format={storage_format} not yet supported by the Python implementation of the service."
        raise ValueError(msg)
