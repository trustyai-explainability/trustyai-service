"""Async SQLAlchemy engine factory for MariaDB and PostgreSQL."""

import logging
import os
import ssl
from pathlib import Path
from urllib.parse import quote_plus

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

_MARIADB_DEFAULT_PORT = 3306
_POSTGRESQL_DEFAULT_PORT = 5432


def _read_db_config(storage_format: str) -> dict:
    """Read database configuration from environment variables."""
    user = os.environ.get("DATABASE_USERNAME") or os.environ.get(
        "QUARKUS_DATASOURCE_USERNAME"
    )
    password = os.environ.get("DATABASE_PASSWORD") or os.environ.get(
        "QUARKUS_DATASOURCE_PASSWORD"
    )
    host = os.environ.get("DATABASE_HOST") or os.environ.get("DATABASE_SERVICE")
    database = os.environ.get("DATABASE_DATABASE") or os.environ.get("DATABASE_NAME")
    ssl_ca = os.environ.get("DATABASE_SSL_CA")

    default_port = (
        _POSTGRESQL_DEFAULT_PORT
        if storage_format in ("POSTGRES", "POSTGRESQL")
        else _MARIADB_DEFAULT_PORT
    )
    port_str = os.environ.get("DATABASE_PORT", str(default_port))
    try:
        port = int(port_str)
    except ValueError:
        msg = f"DATABASE_PORT must be an integer, got '{port_str}'"
        raise ValueError(msg) from None

    missing = []
    if not user:
        missing.append("DATABASE_USERNAME or QUARKUS_DATASOURCE_USERNAME")
    if not password:
        missing.append("DATABASE_PASSWORD or QUARKUS_DATASOURCE_PASSWORD")
    if not host:
        missing.append("DATABASE_HOST or DATABASE_SERVICE")
    if not database:
        missing.append("DATABASE_DATABASE or DATABASE_NAME")
    if missing:
        msg = f"Database storage requires environment variables: {', '.join(missing)}"
        raise ValueError(msg)

    return {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "database": database,
        "ssl_ca": ssl_ca,
    }


def _build_ssl_args(ssl_ca: str | None, storage_format: str) -> dict:
    """Build SSL connection arguments for the database driver."""
    if not ssl_ca:
        return {}
    if not Path(ssl_ca).exists():
        msg = f"DATABASE_SSL_CA is set to '{ssl_ca}' but the file does not exist"
        raise FileNotFoundError(msg)

    if storage_format in ("POSTGRES", "POSTGRESQL"):
        ctx = ssl.create_default_context(cafile=ssl_ca)
        return {"ssl": ctx}

    ctx = ssl.create_default_context(cafile=ssl_ca)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    return {"ssl": ctx}


def create_db_engine(storage_format: str | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine based on environment configuration."""
    if storage_format is None:
        storage_format = os.environ.get("SERVICE_STORAGE_FORMAT", "PVC")

    config = _read_db_config(storage_format)
    ssl_args = _build_ssl_args(config["ssl_ca"], storage_format)

    user = quote_plus(config["user"])
    password = quote_plus(config["password"])

    if storage_format in ("MARIA", "DATABASE"):
        url = (
            f"mysql+asyncmy://{user}:{password}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )
    elif storage_format in ("POSTGRES", "POSTGRESQL"):
        url = (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )
    else:
        msg = f"Unsupported storage format for DB engine: {storage_format}"
        raise ValueError(msg)

    logger.info(
        "Creating async DB engine for %s at %s:%s/%s",
        storage_format,
        config["host"],
        config["port"],
        config["database"],
    )

    return create_async_engine(
        url,
        connect_args=ssl_args,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
