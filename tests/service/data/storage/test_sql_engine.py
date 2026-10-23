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
            if not k.startswith("DATABASE_POOL")
            and k not in {"DATABASE_MAX_OVERFLOW", "DATABASE_MAX_CONNECTIONS_BUDGET"}
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
        assert kwargs["max_overflow"] == 2
        assert kwargs["pool_timeout"] == 3
        assert kwargs["pool_recycle"] == 4

    def test_invalid_env_is_rejected(self) -> None:
        """A non-integer env value fails fast rather than silently using defaults."""
        with (
            patch.dict(os.environ, {"DATABASE_POOL_SIZE": "not-an-int"}, clear=False),
            pytest.raises(ValueError, match="DATABASE_POOL_SIZE='not-an-int'"),
        ):
            eng.pool_kwargs_from_env()

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_pool_size_below_one_is_rejected(self, value: str) -> None:
        """A nonpositive pool size is rejected instead of being clamped."""
        with (
            patch.dict(os.environ, {"DATABASE_POOL_SIZE": value}, clear=False),
            pytest.raises(ValueError, match=f"DATABASE_POOL_SIZE='{value}'"),
        ):
            eng.pool_kwargs_from_env()

    def test_zero_pool_size_engine_is_rejected(self) -> None:
        """An engine refuses the unsafe unlimited-pool setting."""
        with (
            patch.dict(os.environ, {"DATABASE_POOL_SIZE": "0"}, clear=False),
            pytest.raises(ValueError, match="DATABASE_POOL_SIZE='0'"),
        ):
            eng.build_engine(eng.postgres_url("u", "p", "h", 5432, "db"))

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("DATABASE_MAX_OVERFLOW", "-1", "DATABASE_MAX_OVERFLOW='-1'"),
            ("DATABASE_POOL_TIMEOUT", "0", "DATABASE_POOL_TIMEOUT='0'"),
            ("DATABASE_POOL_RECYCLE", "bad", "DATABASE_POOL_RECYCLE='bad'"),
        ],
    )
    def test_invalid_pool_values_are_rejected(
        self, name: str, value: str, message: str
    ) -> None:
        """Pool settings with invalid values fail with the key and supplied value."""
        with (
            patch.dict(os.environ, {name: value}, clear=False),
            pytest.raises(ValueError, match=message),
        ):
            eng.pool_kwargs_from_env()

    def test_pool_ceiling_must_fit_connection_budget(self) -> None:
        """The total per-pod connection ceiling cannot exceed its configured budget."""
        env = {
            "DATABASE_POOL_SIZE": "5",
            "DATABASE_MAX_OVERFLOW": "10",
            "DATABASE_MAX_CONNECTIONS_BUDGET": "8",
        }
        with (
            patch.dict(os.environ, env, clear=False),
            pytest.raises(
                ValueError,
                match=r"DATABASE_MAX_CONNECTIONS_BUDGET=8.*exceeds the per-pod budget",
            ),
        ):
            eng.pool_kwargs_from_env()


class TestUrlBuilders:
    """URL builders produce the expected driver + components."""

    def test_postgres_url(self) -> None:
        """postgres_url uses the psycopg driver."""
        url = eng.postgres_url("u", "p", "h", 5432, "db")
        assert url.drivername == "postgresql+psycopg"
        assert url.username == "u"
        assert url.host == "h"
        assert url.port == 5432
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
        with patch.dict(os.environ, {}, clear=True):
            args = eng.postgres_connect_args("/etc/tls/ca.crt")
        assert args == {
            "sslmode": "verify-full",
            "sslrootcert": "/etc/tls/ca.crt",
            "options": "-c synchronous_commit=on",
        }

    def test_postgres_connect_args_without_ca(self) -> None:
        """Default session durability is explicit even without TLS args."""
        with patch.dict(os.environ, {}, clear=True):
            assert eng.postgres_connect_args(None) == {
                "options": "-c synchronous_commit=on"
            }

    def test_postgres_tuning_is_passed_to_new_sessions(self) -> None:
        """Application name, timeout, and sync mode become PostgreSQL startup options."""
        env = {
            "DATABASE_APPLICATION_NAME": "storage-test",
            "DATABASE_STATEMENT_TIMEOUT_MS": "45000",
            "DATABASE_SYNCHRONOUS_COMMIT": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            args = eng.postgres_connect_args(None)
        assert args == {
            "application_name": "storage-test",
            "options": "-c statement_timeout=45000 -c synchronous_commit=off",
        }

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            (
                "DATABASE_STATEMENT_TIMEOUT_MS",
                "-5",
                "DATABASE_STATEMENT_TIMEOUT_MS='-5'",
            ),
            (
                "DATABASE_SYNCHRONOUS_COMMIT",
                "maybe",
                "DATABASE_SYNCHRONOUS_COMMIT='maybe'",
            ),
        ],
    )
    def test_invalid_postgres_tuning_is_rejected(
        self, name: str, value: str, message: str
    ) -> None:
        """Invalid PostgreSQL tuning values fail before connecting to the server."""
        with (
            patch.dict(os.environ, {name: value}, clear=True),
            pytest.raises(ValueError, match=message),
        ):
            eng.postgres_connect_args(None)

    def test_mariadb_connect_args_with_ca(self) -> None:
        """MariaDB CA arg and certificate verification are set when a CA is given."""
        assert eng.mariadb_connect_args("/etc/tls/ca.crt") == {
            "ssl_ca": "/etc/tls/ca.crt",
            "ssl_verify_cert": True,
        }

    def test_mariadb_connect_args_without_ca(self) -> None:
        """No TLS args without a CA."""
        assert eng.mariadb_connect_args(None) == {}


class TestBuildEngine:
    """build_engine wires pooling on/off appropriately."""

    def test_pooled_engine_uses_queuepool(self) -> None:
        """A pooled engine uses QueuePool with the configured size."""
        pytest.importorskip("psycopg")  # postgres extra is optional
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
