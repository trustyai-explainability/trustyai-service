"""Validation-error sanitization for local explanation requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_SENSITIVE_LOCATION_FIELDS = frozenset({"base_url"})
_SENSITIVE_ERROR_KEYS = frozenset({"input", "ctx"})
_LOCAL_EXPLAINER_PATH = "/explainers/local"


def is_local_explainer_path(path: str) -> bool:
    """Return whether a request path belongs to a local-explainer route."""
    return path == _LOCAL_EXPLAINER_PATH or path.startswith(f"{_LOCAL_EXPLAINER_PATH}/")


def _is_sensitive_location(location: object) -> bool:
    """Return whether a validation location contains a credential-bearing field."""
    if not isinstance(location, (list, tuple)):
        return False
    return any(part in _SENSITIVE_LOCATION_FIELDS for part in location)


def contains_sensitive_validation_error(
    errors: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether validation errors include a local model base URL."""
    return any(_is_sensitive_location(error.get("loc")) for error in errors)


def sanitize_validation_errors(
    errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove raw URL inputs and exception context from sensitive errors."""
    sanitized: list[dict[str, Any]] = []
    for error in errors:
        if _is_sensitive_location(error.get("loc")):
            sanitized.append(
                {
                    key: value
                    for key, value in error.items()
                    if key not in _SENSITIVE_ERROR_KEYS
                }
            )
        else:
            sanitized.append(dict(error))
    return sanitized
