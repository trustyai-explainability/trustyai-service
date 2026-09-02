"""Utility functions and decorators for PostgreSQL storage operations.

These helpers intentionally duplicate the dialect-agnostic MariaDB equivalents
(`get_clean_column_names`, `require_existing_dataset`) so that the ``postgres``
package stays fully decoupled from the ``maria`` package.
"""

from collections.abc import Callable, Coroutine
from types import TracebackType
from typing import Any

import psycopg


def require_existing_dataset[**P, R](
    func: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Annotation to assert that a given function requires a valid dataset name as the first non-self argument."""

    async def validate_dataset_exists(*args: P.args, **kwargs: P.kwargs) -> R:
        storage, dataset_name = args[0], args[1]
        if not await storage.dataset_exists(dataset_name):
            msg = f"Error when calling {func.__name__}: Dataset '{dataset_name}' does not exist."
            raise ValueError(msg)
        return await func(*args, **kwargs)

    return validate_dataset_exists


def get_clean_column_names(column_names: list[str]) -> list[str]:
    """Programmatically generate the column names in a model data table.

    This avoids possible SQL injection from the real column names coming
    from the mode.
    """
    return [f"column_{i}" for i in range(len(column_names))]


class PostgresConnectionManager:
    """Context manager for PostgreSQL (psycopg 3) database connections."""

    def __init__(
        self,
        user: str | None,
        password: str | None,
        host: str | None,
        port: int,
        database: str | None,
        ssl_ca: str | None = None,
        connect_timeout: int | None = None,
    ) -> None:
        """Initialize connection manager with database credentials.

        :param user: Database user
        :param password: Database password
        :param host: Database host
        :param port: Database port
        :param database: Database name
        :param ssl_ca: Path to CA certificate for TLS connection
        :param connect_timeout: Connection timeout in seconds (None = driver default)
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.ssl_ca = ssl_ca
        self.connect_timeout = connect_timeout

    def __enter__(self) -> tuple[psycopg.Connection, psycopg.Cursor]:
        """Enter context manager and establish database connection."""
        connect_kwargs: dict[str, str | int | bool | None] = {
            "dbname": self.database,
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
        }
        if self.ssl_ca:
            connect_kwargs["sslmode"] = "verify-full"
            connect_kwargs["sslrootcert"] = self.ssl_ca
        if self.connect_timeout is not None:
            connect_kwargs["connect_timeout"] = self.connect_timeout
        self.conn = psycopg.connect(**connect_kwargs)
        return self.conn, self.conn.cursor()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit context manager and close database connection."""
        self.conn.close()
