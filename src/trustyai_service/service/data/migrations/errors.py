"""Exceptions raised by the storage migration framework."""


class MigrationError(Exception):
    """Base class for migration failures."""


class MigrationPreflightError(MigrationError):
    """Raised before writes when the source or configuration is unsafe."""


class MigrationValidationError(MigrationError):
    """Raised when source data cannot be represented safely."""


class MigrationWriteError(MigrationError):
    """Raised when a target write fails after preflight."""
