"""Map local-explainer failures to endpoint-neutral response data.

HTTP adapters serialize :class:`ErrorResponse` as
``{"detail": {"code": "...", "message": "..."}}``.
"""

from dataclasses import dataclass
from enum import StrEnum

from .model_provider import ProviderError


class LocalErrorCode(StrEnum):
    """Stable categories shared by local-explainer endpoints."""

    DATA_INVALID = "data_invalid"
    DATA_MISSING = "data_missing"
    INVALID_REQUEST = "invalid_request"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    CONFIGURATION_INVALID = "configuration_invalid"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"
    UNSUPPORTED_MODEL = "unsupported_model"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    EXECUTION_FAILED = "execution_failed"


class LocalDataError(ValueError):
    """Stored local-explanation data is invalid for the request."""

    code = LocalErrorCode.DATA_INVALID


class LocalDataNotFoundError(LocalDataError):
    """Required stored local-explanation data does not exist."""

    code = LocalErrorCode.DATA_MISSING


class LocalExecutionError(RuntimeError):
    """Unexpected local surrogate or explainer execution failure."""

    code = LocalErrorCode.EXECUTION_FAILED


class LocalWorkerCapacityError(LocalExecutionError):
    """No bounded local-explainer worker slot is currently available."""

    code = LocalErrorCode.RESOURCE_EXHAUSTED

    def __init__(self, message: str = "Local explainer capacity is exhausted") -> None:
        """Create a capacity-admission failure."""
        super().__init__(message)


@dataclass(frozen=True)
class ErrorResponse:
    """HTTP-neutral status, stable code, and safe detail for one failure."""

    status_code: int
    detail: str
    code: str = LocalErrorCode.EXECUTION_FAILED

    def as_http_detail(self) -> dict[str, str]:
        """Serialize the documented ``detail.code/message`` HTTP error shape."""
        return {"code": self.code, "message": self.detail}


_MAPPED_ERRORS: dict[str, tuple[int, str]] = {
    LocalErrorCode.DATA_INVALID: (400, "Local explanation data is invalid"),
    LocalErrorCode.DATA_MISSING: (404, "Local explanation data was not found"),
    LocalErrorCode.INVALID_REQUEST: (400, "Invalid local explanation request"),
    LocalErrorCode.DEPENDENCY_UNAVAILABLE: (
        503,
        "Local explainer dependency is unavailable",
    ),
    LocalErrorCode.CONFIGURATION_INVALID: (
        500,
        "Local explainer configuration is invalid",
    ),
    LocalErrorCode.UNAVAILABLE: (503, "Model provider is unavailable"),
    LocalErrorCode.INVALID_RESPONSE: (
        502,
        "Model provider returned an invalid response",
    ),
    LocalErrorCode.UNSUPPORTED_MODEL: (
        502,
        "Model provider returned an unsupported model",
    ),
    LocalErrorCode.DEADLINE_EXCEEDED: (
        504,
        "Explanation exceeded its deadline",
    ),
    LocalErrorCode.RESOURCE_EXHAUSTED: (
        503,
        "Local explainer capacity is exhausted",
    ),
    LocalErrorCode.EXECUTION_FAILED: (500, "Local explanation failed"),
}


def _code_for_error(error: Exception) -> str:
    """Resolve an explicit error code without inferring data failures."""
    if isinstance(error, ImportError):
        return LocalErrorCode.DEPENDENCY_UNAVAILABLE
    if isinstance(error, TimeoutError):
        return LocalErrorCode.DEADLINE_EXCEEDED
    if not isinstance(error, (LocalDataError, LocalExecutionError, ProviderError)):
        return LocalErrorCode.EXECUTION_FAILED
    code = getattr(error, "code", None)
    if isinstance(code, str) and code in _MAPPED_ERRORS:
        return code
    return LocalErrorCode.EXECUTION_FAILED


def map_error(error: Exception) -> ErrorResponse:
    """Return endpoint-neutral mapping data without constructing a web response."""
    code = _code_for_error(error)
    status_code, generic_detail = _MAPPED_ERRORS[code]
    return ErrorResponse(status_code=status_code, detail=generic_detail, code=code)
