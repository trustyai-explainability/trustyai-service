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
    """Return whether a location contains a sensitive field or credential value."""
    if not isinstance(location, (list, tuple)):
        return False
    return any(
        part in _SENSITIVE_LOCATION_FIELDS or _contains_credential_bearing_url(part)
        for part in location
    )


def _contains_credential_bearing_url(value: object) -> bool:
    """Return whether a validation input could contain URL credentials."""
    if isinstance(value, (bytes, bytearray)):
        # Unparsed request bodies cannot be assumed to be credential-free.
        return True
    if isinstance(value, str):
        try:
            parsed = urlsplit(value)
        except ValueError:
            # Malformed URL syntax must not bypass credential redaction.
            return True
        return (
            parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
            or (not parsed.netloc and "@" in parsed.path)
        )
    if isinstance(value, Mapping):
        return any(
            _contains_credential_bearing_url(key)
            or _contains_credential_bearing_url(item)
            for key, item in value.items()
        )
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
    """Remove raw inputs, exception context, and sensitive location components."""
    sensitive_response = contains_sensitive_validation_error(errors)
    sanitized: list[dict[str, Any]] = []
    for error in errors:
        if sensitive_response:
            safe_error = {
                key: value
                for key, value in error.items()
                if key not in _SENSITIVE_ERROR_KEYS
            }
            location = safe_error.get("loc")
            if isinstance(location, (list, tuple)):
                safe_error["loc"] = [
                    "[redacted]" if _contains_credential_bearing_url(part) else part
                    for part in location
                ]
            sanitized.append(safe_error)
        else:
            sanitized.append(dict(error))
    return sanitized
