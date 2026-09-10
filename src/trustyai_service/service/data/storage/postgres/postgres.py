"""PostgreSQL storage backend for TrustyAI inference data.

Thin :class:`SQLStorage` subclass: all control flow lives in the shared
SQLAlchemy Core base; this adapter only builds the PostgreSQL engine (psycopg 3
driver, ``verify-full`` TLS) and supplies the backend name. Schema names,
metadata shape, method signatures, error messages, and the ``trustyai_v2_*``
layout match the historical raw-SQL backend so existing data stays readable.
"""

from __future__ import annotations

from trustyai_service.service.data.storage.sql.base import SQLStorage
from trustyai_service.service.data.storage.sql.engine import (
    build_engine,
    postgres_connect_args,
    postgres_url,
)


class PostgreSQLStorage(SQLStorage):
    """PostgreSQL backend built on the shared SQLAlchemy Core base."""

    _backend_name = "PostgreSQL"

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        port: int,
        database: str,
        *,
        ssl_ca: str | None = None,
    ) -> None:
        """Initialize PostgreSQL storage and create schema tables."""
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

        engine = build_engine(
            postgres_url(user, password, host, port, database),
            connect_args=postgres_connect_args(ssl_ca),
        )
        super().__init__(engine)
