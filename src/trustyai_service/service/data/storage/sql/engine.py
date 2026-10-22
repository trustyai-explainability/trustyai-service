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
_DEFAULT_SYNCHRONOUS_COMMIT = "on"


def _int_env(name: str, default: int, *, minimum: int) -> int:
    """Read and validate an integer env var, using ``default`` only when unset."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        msg = f"Invalid {name}={raw!r}: must be an integer"
        raise ValueError(msg) from None
    if value < minimum:
        msg = f"Invalid {name}={raw!r}: must be at least {minimum}"
        raise ValueError(msg)
    return value


def pool_kwargs_from_env() -> dict[str, int]:
    """Return ``create_engine`` pool kwargs sourced from the environment.

    Env vars: ``DATABASE_POOL_SIZE``, ``DATABASE_MAX_OVERFLOW``,
    ``DATABASE_POOL_TIMEOUT``, ``DATABASE_POOL_RECYCLE``.
    """
    errors: list[str] = []

    def pool_int(name: str, default: int, *, minimum: int) -> int:
        try:
            return _int_env(name, default, minimum=minimum)
        except ValueError as error:
            errors.append(str(error))
            return default

    pool_size = pool_int("DATABASE_POOL_SIZE", _DEFAULT_POOL_SIZE, minimum=1)
    max_overflow = pool_int("DATABASE_MAX_OVERFLOW", _DEFAULT_MAX_OVERFLOW, minimum=0)
    pool_timeout = pool_int("DATABASE_POOL_TIMEOUT", _DEFAULT_POOL_TIMEOUT, minimum=1)
    # SQLAlchemy uses -1 to disable connection recycling.
    pool_recycle = pool_int("DATABASE_POOL_RECYCLE", _DEFAULT_POOL_RECYCLE, minimum=-1)
    budget_raw = os.environ.get("DATABASE_MAX_CONNECTIONS_BUDGET")
    if budget_raw is not None:
        budget = pool_int("DATABASE_MAX_CONNECTIONS_BUDGET", 0, minimum=1)
        if budget > 0:
            pool_ceiling = pool_size + max_overflow
            if pool_ceiling > budget:
                errors.append(
                    f"DATABASE_MAX_CONNECTIONS_BUDGET={budget}: configured pool "
                    f"ceiling {pool_ceiling} exceeds the per-pod budget"
                )

    if errors:
        raise ValueError("; ".join(errors))

    return {
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_timeout": pool_timeout,
        "pool_recycle": pool_recycle,
        "pool_pre_ping": True,
    }


def postgres_tuning_from_env() -> dict[str, str | int | None]:
    """Read and validate PostgreSQL session settings from the environment."""
    application_name = os.environ.get("DATABASE_APPLICATION_NAME")
    if application_name is not None and "\x00" in application_name:
        msg = "Invalid DATABASE_APPLICATION_NAME: NUL characters are not supported"
        raise ValueError(msg)

    statement_timeout_ms = None
    if "DATABASE_STATEMENT_TIMEOUT_MS" in os.environ:
        statement_timeout_ms = _int_env("DATABASE_STATEMENT_TIMEOUT_MS", 0, minimum=0)

    raw_synchronous_commit = os.environ.get(
        "DATABASE_SYNCHRONOUS_COMMIT", _DEFAULT_SYNCHRONOUS_COMMIT
    )
    synchronous_commit = {
        "1": "on",
        "true": "on",
        "yes": "on",
        "on": "on",
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
    }.get(raw_synchronous_commit.strip().lower())
    if synchronous_commit is None:
        msg = (
            "Invalid DATABASE_SYNCHRONOUS_COMMIT="
            f"{raw_synchronous_commit!r}: must be a boolean (true or false)"
        )
        raise ValueError(msg)

    return {
        "application_name": application_name,
        "statement_timeout_ms": statement_timeout_ms,
        "synchronous_commit": synchronous_commit,
    }


def sqlite_pragmas_from_env() -> dict[str, str | int]:
    """Read and validate configured SQLite pragmas.

    Only explicitly configured pragmas are returned, preserving SQLite's
    existing defaults when no ConfigMap value is supplied.
    """
    pragmas: dict[str, str | int] = {}

    journal_mode = os.environ.get("DATABASE_SQLITE_JOURNAL_MODE")
    if journal_mode is not None:
        normalized_mode = journal_mode.strip().upper()
        if normalized_mode not in {
            "DELETE",
            "TRUNCATE",
            "PERSIST",
            "MEMORY",
            "WAL",
            "OFF",
        }:
            msg = (
                "Invalid DATABASE_SQLITE_JOURNAL_MODE="
                f"{journal_mode!r}: unsupported SQLite journal mode"
            )
            raise ValueError(msg)
        pragmas["journal_mode"] = normalized_mode

    synchronous = os.environ.get("DATABASE_SQLITE_SYNCHRONOUS")
    if synchronous is not None:
        normalized_synchronous = synchronous.strip().upper()
        synchronous_values = {"OFF": 0, "NORMAL": 1, "FULL": 2, "EXTRA": 3}
        if normalized_synchronous not in synchronous_values:
            msg = (
                "Invalid DATABASE_SQLITE_SYNCHRONOUS="
                f"{synchronous!r}: expected OFF, NORMAL, FULL, or EXTRA"
            )
            raise ValueError(msg)
        pragmas["synchronous"] = synchronous_values[normalized_synchronous]

    busy_timeout_ms = os.environ.get("DATABASE_SQLITE_BUSY_TIMEOUT_MS")
    if busy_timeout_ms is not None:
        pragmas["busy_timeout"] = _int_env(
            "DATABASE_SQLITE_BUSY_TIMEOUT_MS", 0, minimum=0
        )

    cache_size_kb = os.environ.get("DATABASE_SQLITE_CACHE_SIZE_KB")
    if cache_size_kb is not None:
        pragmas["cache_size"] = -_int_env("DATABASE_SQLITE_CACHE_SIZE_KB", 1, minimum=1)

    return pragmas


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

    With no CA certificate libpq uses its default ``sslmode=prefer``, which allows
    an unverified or plaintext connection. This preserves optional database TLS;
    setting ``DATABASE_ALLOW_INSECURE_TLS=false`` requires a usable CA.
    """
    args: dict[str, Any] = {}
    if ssl_ca:
        args.update({"sslmode": "verify-full", "sslrootcert": ssl_ca})

    tuning = postgres_tuning_from_env()
    if tuning["application_name"] is not None:
        args["application_name"] = tuning["application_name"]

    options = []
    if tuning["statement_timeout_ms"] is not None:
        options.append(f"-c statement_timeout={tuning['statement_timeout_ms']}")
    options.append(f"-c synchronous_commit={tuning['synchronous_commit']}")
    args["options"] = " ".join(options)
    return args


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
    with PostgreSQL, a ``None`` CA leaves the connection unencrypted by default;
    setting ``DATABASE_ALLOW_INSECURE_TLS=false`` requires a usable CA.
    """
    if ssl_ca:
        return {"ssl_ca": ssl_ca, "ssl_verify_cert": True}
    return {}


def sqlite_url(path: str) -> URL:
    """Build a SQLite URL. ``path`` may be ``:memory:`` or a filesystem path."""
    if path == ":memory:":
        return URL.create("sqlite+pysqlite", database=":memory:")
    return URL.create("sqlite+pysqlite", database=path)
