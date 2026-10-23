"""Exceptions raised by the storage migration framework."""

from sqlalchemy.exc import SQLAlchemyError


def error_detail(error: Exception) -> str:
    """Keep driver SQL, bound inference data and credentials out of results."""
    if isinstance(error, SQLAlchemyError):
        return f"database operation failed ({type(error).__name__})"
    return str(error)


class MigrationError(Exception):
    """Base class for migration failures."""


class MigrationPreflightError(MigrationError):
    """Raised before writes when the source or configuration is unsafe."""


class MigrationValidationError(MigrationError):
    """Raised when source data cannot be represented safely."""


class MigrationWriteError(MigrationError):
    """Raised when a target write fails after preflight."""
