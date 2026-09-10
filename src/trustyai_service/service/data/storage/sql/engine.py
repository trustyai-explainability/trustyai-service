"""Engine/pool construction and dialect-specific connect args (incl. TLS).

Each SQL backend builds its :class:`sqlalchemy.Engine` here. Connection pooling
(SQLAlchemy's ``QueuePool``) is enabled out of the box, sized via environment
variables so multi-pod deployments do not exhaust the server's connection limit.
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL

# Conservative per-pod pool defaults (env-overridable). See the plan's §6.2.
_DEFAULT_POOL_SIZE = 5
_DEFAULT_MAX_OVERFLOW = 10
_DEFAULT_POOL_TIMEOUT = 30
_DEFAULT_POOL_RECYCLE = (
    1800  # recycle connections after 30 min to dodge server idle-kills
)


def _int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back to ``default`` when unset/invalid."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def pool_kwargs_from_env() -> dict[str, int]:
    """Return ``create_engine`` pool kwargs sourced from the environment.

    Env vars: ``DATABASE_POOL_SIZE``, ``DATABASE_MAX_OVERFLOW``,
    ``DATABASE_POOL_TIMEOUT``, ``DATABASE_POOL_RECYCLE``.
    """
    return {
        "pool_size": _int_env("DATABASE_POOL_SIZE", _DEFAULT_POOL_SIZE),
        "max_overflow": _int_env("DATABASE_MAX_OVERFLOW", _DEFAULT_MAX_OVERFLOW),
        "pool_timeout": _int_env("DATABASE_POOL_TIMEOUT", _DEFAULT_POOL_TIMEOUT),
        "pool_recycle": _int_env("DATABASE_POOL_RECYCLE", _DEFAULT_POOL_RECYCLE),
        "pool_pre_ping": True,
    }


def build_engine(
    url: URL | str,
    *,
    connect_args: dict[str, Any] | None = None,
    use_pool: bool = True,
) -> Engine:
    """Create an :class:`~sqlalchemy.Engine` for ``url``.

    :param connect_args: dialect-specific DBAPI connect args (e.g. TLS).
    :param use_pool: when ``False`` (SQLite), skip queue-pool sizing kwargs that
        SQLite's default pool does not accept.
    """
    kwargs: dict[str, Any] = {"connect_args": connect_args or {}}
    if use_pool:
        kwargs.update(pool_kwargs_from_env())
    return create_engine(url, **kwargs)


def postgres_url(
    user: str | None,
    password: str | None,
    host: str | None,
    port: int,
    database: str | None,
) -> URL:
    """Build a psycopg-3 PostgreSQL URL."""
    return URL.create(
        "postgresql+psycopg",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )


def postgres_connect_args(ssl_ca: str | None) -> dict[str, Any]:
    """Psycopg TLS connect args mirroring the raw-SQL backend (``verify-full``).

    With no CA certificate libpq falls back to ``sslmode=prefer``, which allows an
    unverified or plaintext connection. Configuration read from the environment
    refuses that unless ``DATABASE_ALLOW_INSECURE_TLS`` is set, so a ``None`` here
    means the deployment opted in explicitly (or is a direct, in-test construction).
    """
    if ssl_ca:
        return {"sslmode": "verify-full", "sslrootcert": ssl_ca}
    return {}


def mariadb_url(
    user: str | None,
    password: str | None,
    host: str | None,
    port: int,
    database: str | None,
) -> URL:
    """Build a MariaDB Connector/Python URL (uses the ``mariadb`` PyPI package)."""
    return URL.create(
        "mariadb+mariadbconnector",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )


def mariadb_connect_args(ssl_ca: str | None) -> dict[str, Any]:
    """MariaDB TLS connect args (CA verification).

    ``ssl_verify_cert`` is set explicitly to match ``MariaConnectionManager``. As
    with PostgreSQL, a ``None`` CA leaves the connection unencrypted and is only
    reachable from the environment after a ``DATABASE_ALLOW_INSECURE_TLS`` opt-in.
    """
    if ssl_ca:
        return {"ssl_ca": ssl_ca, "ssl_verify_cert": True}
    return {}


def sqlite_url(path: str) -> URL:
    """Build a SQLite URL. ``path`` may be ``:memory:`` or a filesystem path."""
    if path == ":memory:":
        return URL.create("sqlite+pysqlite", database=":memory:")
    return URL.create("sqlite+pysqlite", database=path)
