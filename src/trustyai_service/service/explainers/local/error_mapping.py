"""Map local explainer service errors to HTTP-neutral response data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorResponse:
    """HTTP-neutral status and safe detail for an explainer failure."""

    status_code: int
    detail: str


def map_error(error: Exception) -> ErrorResponse:
    """Map an internal provider or execution error to response semantics."""
    if isinstance(error, ImportError):
        return ErrorResponse(503, "Local explainer dependency is unavailable")
    code = getattr(error, "code", None)
    fixed_responses = {
        "unavailable": ErrorResponse(503, "Model provider is unavailable"),
        "dependency_unavailable": ErrorResponse(503, "Model provider is unavailable"),
        "deadline_exceeded": ErrorResponse(504, "Explanation exceeded its deadline"),
        "invalid_response": ErrorResponse(
            502, "Model provider returned an unsupported response"
        ),
        "unsupported_model": ErrorResponse(
            502, "Model provider returned an unsupported response"
        ),
        "configuration_invalid": ErrorResponse(
            500, "Model provider configuration is invalid"
        ),
    }
    if code in fixed_responses:
        return fixed_responses[code]
    if code == "data_missing":
        return ErrorResponse(404, str(error))
    if code in {"invalid_request", "data_invalid"}:
        return ErrorResponse(400, str(error))
    return ErrorResponse(500, "Local explanation failed")
