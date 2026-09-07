"""Unit tests for the SQL engine/URL/pool helpers (no running database)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

pytest.importorskip("sqlalchemy")

from sqlalchemy.pool import NullPool, QueuePool

from trustyai_service.service.data.storage.sql import engine as eng

DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_OVERFLOW = 10
DEFAULT_POOL_TIMEOUT = 30
DEFAULT_POOL_RECYCLE = 1800


class TestPoolKwargsFromEnv:
    """pool_kwargs_from_env reads pool config from the environment."""

    def test_defaults_when_unset(self) -> None:
        """Conservative defaults are used when no env vars are set."""
        clean = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("DATABASE_POOL") and k != "DATABASE_MAX_OVERFLOW"
        }
        with patch.dict(os.environ, clean, clear=True):
            kwargs = eng.pool_kwargs_from_env()
        assert kwargs["pool_size"] == DEFAULT_POOL_SIZE
        assert kwargs["max_overflow"] == DEFAULT_MAX_OVERFLOW
        assert kwargs["pool_timeout"] == DEFAULT_POOL_TIMEOUT
        assert kwargs["pool_recycle"] == DEFAULT_POOL_RECYCLE
        assert kwargs["pool_pre_ping"] is True

    def test_env_overrides(self) -> None:
        """Env vars override the pool defaults."""
        env = {
            "DATABASE_POOL_SIZE": "1",
            "DATABASE_MAX_OVERFLOW": "2",
            "DATABASE_POOL_TIMEOUT": "3",
            "DATABASE_POOL_RECYCLE": "4",
        }
        with patch.dict(os.environ, env, clear=False):
            kwargs = eng.pool_kwargs_from_env()
        assert kwargs["pool_size"] == 1
        assert kwargs["max_overflow"] == 2  # noqa: PLR2004 -- exact value under test
        assert kwargs["pool_timeout"] == 3  # noqa: PLR2004 -- exact value under test
        assert kwargs["pool_recycle"] == 4  # noqa: PLR2004 -- exact value under test

    def test_invalid_env_falls_back_to_default(self) -> None:
        """A non-integer env value falls back to the default."""
        with patch.dict(os.environ, {"DATABASE_POOL_SIZE": "not-an-int"}, clear=False):
            kwargs = eng.pool_kwargs_from_env()
        assert kwargs["pool_size"] == DEFAULT_POOL_SIZE


class TestUrlBuilders:
    """URL builders produce the expected driver + components."""

    def test_postgres_url(self) -> None:
        """postgres_url uses the psycopg driver."""
        url = eng.postgres_url("u", "p", "h", 5432, "db")
        assert url.drivername == "postgresql+psycopg"
        assert url.username == "u"
        assert url.host == "h"
        assert url.port == 5432  # noqa: PLR2004 -- exact port under test
        assert url.database == "db"

    def test_mariadb_url(self) -> None:
        """mariadb_url uses the mariadbconnector driver."""
        url = eng.mariadb_url("u", "p", "h", 3306, "db")
        assert url.drivername == "mariadb+mariadbconnector"
        assert url.database == "db"

    def test_sqlite_url_memory(self) -> None:
        """sqlite_url handles the in-memory database."""
        url = eng.sqlite_url(":memory:")
        assert url.drivername == "sqlite+pysqlite"
        assert url.database == ":memory:"

    def test_sqlite_url_file(self) -> None:
        """sqlite_url handles a filesystem path."""
        url = eng.sqlite_url("/tmp/db.sqlite")  # noqa: S108 -- test path only
        assert url.database == "/tmp/db.sqlite"  # noqa: S108 -- test path only


class TestConnectArgs:
    """TLS connect args are produced only when a CA cert is supplied."""

    def test_postgres_connect_args_with_ca(self) -> None:
        """PostgreSQL verify-full args are set when a CA is given."""
        args = eng.postgres_connect_args("/etc/tls/ca.crt")
        assert args == {"sslmode": "verify-full", "sslrootcert": "/etc/tls/ca.crt"}

    def test_postgres_connect_args_without_ca(self) -> None:
        """No TLS args without a CA."""
        assert eng.postgres_connect_args(None) == {}

    def test_mariadb_connect_args_with_ca(self) -> None:
        """MariaDB CA arg is set when a CA is given."""
        assert eng.mariadb_connect_args("/etc/tls/ca.crt") == {
            "ssl_ca": "/etc/tls/ca.crt"
        }

    def test_mariadb_connect_args_without_ca(self) -> None:
        """No TLS args without a CA."""
        assert eng.mariadb_connect_args(None) == {}


class TestBuildEngine:
    """build_engine wires pooling on/off appropriately."""

    def test_pooled_engine_uses_queuepool(self) -> None:
        """A pooled engine uses QueuePool with the configured size."""
        engine = eng.build_engine(eng.postgres_url("u", "p", "h", 5432, "db"))
        assert isinstance(engine.pool, QueuePool)
        engine.dispose()

    def test_unpooled_engine_skips_pool_sizing(self) -> None:
        """use_pool=False builds an engine without queue-pool sizing kwargs."""
        # NullPool has no size; this just asserts no TypeError from passing
        # pool_size to a pool that rejects it.
        engine = eng.build_engine(
            eng.sqlite_url(":memory:"),
            use_pool=False,
        )
        assert not isinstance(engine.pool, QueuePool) or isinstance(
            engine.pool, NullPool
        )
        engine.dispose()
