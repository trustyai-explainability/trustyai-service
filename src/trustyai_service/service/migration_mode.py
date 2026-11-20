"""Resolve how the startup database migration is configured.

``DATABASE_ATTEMPT_MIGRATION`` historically defaulted to *off*, which left a
database populated by the legacy (Java) service invisible to this service until
someone opted in. The legacy-schema copy is non-destructive (the legacy tables
are only read) and idempotent, so it now runs automatically when the variable is
unset. Setting the variable explicitly keeps working:

* unset or empty     -> ``AUTO``: copy legacy v1 data if legacy tables exist.
* ``1/true/yes/on``  -> ``ENABLED``: the legacy copy plus the PVC-to-DB migration.
* anything else      -> ``DISABLED``: no startup migration (explicit opt-out).

This module only uses the standard library so it can be imported from the health
checks without pulling in the optional database drivers.
"""

from __future__ import annotations

import os
from enum import StrEnum

MIGRATION_ENV_VAR = "DATABASE_ATTEMPT_MIGRATION"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class MigrationMode(StrEnum):
    """How the startup database migration should behave."""

    DISABLED = "disabled"
    AUTO = "auto"
    ENABLED = "enabled"


def get_migration_mode() -> MigrationMode:
    """Return the migration mode configured through ``DATABASE_ATTEMPT_MIGRATION``."""
    raw = os.environ.get(MIGRATION_ENV_VAR)
    if raw is None or not raw.strip():
        return MigrationMode.AUTO
    if raw.strip().lower() in _TRUTHY:
        return MigrationMode.ENABLED
    return MigrationMode.DISABLED
