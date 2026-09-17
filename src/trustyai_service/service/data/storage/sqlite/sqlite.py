"""SQLite storage backend for TrustyAI inference data.

Thin :class:`SQLStorage` subclass using the stdlib ``sqlite3`` driver (no extra
dependency). SQLite is single-writer with file-level locking, so this backend is
intended for **local development, tests, and single-instance deployments only**
-- it cannot satisfy multi-pod concurrent writes on shared storage. Use the
PostgreSQL backend for production multi-replica deployments.

For an in-memory database (``:memory:``), a :class:`~sqlalchemy.pool.QueuePool`
limited to one connection keeps every operation pointed at the same database.
Each checkout is exclusive until its transaction and connection are closed,
so threaded model discovery cannot roll back an overlapping writer's work.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from trustyai_service.service.data.storage.sql.base import SQLStorage
from trustyai_service.service.data.storage.sql.engine import sqlite_url

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

IN_MEMORY = ":memory:"


class SQLiteStorage(SQLStorage):
    """SQLite backend built on the shared SQLAlchemy Core base.

    .. warning::
        Not for production multi-replica deployments. See module docstring.
    """

    _backend_name = "SQLite"

    def __init__(self, path: str = IN_MEMORY) -> None:
        """Initialize SQLite storage at ``path`` (``:memory:`` or a file path)."""
        self.path = path
        if path == IN_MEMORY:
            engine = create_engine(
                sqlite_url(path),
                connect_args={"check_same_thread": False},
                poolclass=QueuePool,
                pool_size=1,
                max_overflow=0,
            )
        else:
            engine = create_engine(
                sqlite_url(path),
                connect_args={"check_same_thread": False},
            )
        super().__init__(engine)
