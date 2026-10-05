"""Validation-error sanitization for local explanation requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

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


def _contains_credential_bearing_url(value: object) -> bool:
    """Return whether a JSON-like value contains URL credentials."""
    if isinstance(value, str):
        try:
            parsed = urlsplit(value)
        except ValueError:
            return False
        return (
            parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
        )
    if isinstance(value, Mapping):
        return any(_contains_credential_bearing_url(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_credential_bearing_url(item) for item in value)
    return False


def contains_sensitive_validation_error(
    errors: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether validation errors could expose local model credentials."""
    return any(
        _is_sensitive_location(error.get("loc"))
        or any(
            _contains_credential_bearing_url(error.get(key))
            for key in _SENSITIVE_ERROR_KEYS
        )
        for error in errors
    )


def sanitize_validation_errors(
    errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Remove raw inputs and exception context from a sensitive response."""
    sensitive_response = contains_sensitive_validation_error(errors)
    sanitized: list[dict[str, Any]] = []
    for error in errors:
        if sensitive_response:
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
