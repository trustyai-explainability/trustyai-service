"""Shared SQLAlchemy Core storage base for all SQL backends.

`SQLStorage` (in :mod:`.base`) implements the full :class:`StorageInterface`
control flow once, expressed in SQLAlchemy Core. Each SQL backend
(PostgreSQL, SQLite, MariaDB) is a thin subclass that only supplies its
:class:`sqlalchemy.Engine`, its config parsing, and any dialect override the
Core layer cannot infer. See ``docs/sql-storage-backends-plan.md``.
"""
