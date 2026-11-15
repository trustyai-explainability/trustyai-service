"""Process-wide state of the startup data migration, read by the readiness probe."""

from dataclasses import dataclass
from enum import StrEnum


class StartupMigrationState(StrEnum):
    """Progress of the one-time migration that runs before the API starts."""

    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class _Holder:
    state: StartupMigrationState = StartupMigrationState.IDLE


_holder = _Holder()


def get_startup_migration_state() -> StartupMigrationState:
    """Return the current startup migration state."""
    return _holder.state


def set_startup_migration_state(state: StartupMigrationState) -> None:
    """Record the current startup migration state."""
    _holder.state = state
