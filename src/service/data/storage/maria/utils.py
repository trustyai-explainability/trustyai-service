"""Utility functions and decorators for MariaDB storage operations."""

from collections.abc import Callable, Coroutine
from types import TracebackType
from typing import Any

import mariadb


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


class MariaConnectionManager:
    """Context manager for MariaDB database connections."""

    def __init__(
        self,
        user: str | None,
        password: str | None,
        host: str | None,
        port: int,
        database: str | None,
        ssl_ca: str | None = None,
    ) -> None:
        """Initialize connection manager with database credentials.

        :param user: Database user
        :param password: Database password
        :param host: Database host
        :param port: Database port
        :param database: Database name
        :param ssl_ca: Path to CA certificate for TLS connection
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.ssl_ca = ssl_ca

    def __enter__(self) -> tuple[mariadb.Connection, mariadb.Cursor]:
        """Enter context manager and establish database connection."""
        connect_kwargs: dict[str, str | int | bool | None] = {
            "user": self.user,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "database": self.database,
        }
        if self.ssl_ca:
            connect_kwargs["ssl_ca"] = self.ssl_ca
            connect_kwargs["ssl_verify_cert"] = True
        self.conn = mariadb.connect(**connect_kwargs)
        return self.conn, self.conn.cursor()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit context manager and close database connection."""
        self.conn.close()
