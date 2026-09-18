"""Map local explainer service errors to HTTP-neutral response data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorResponse:
    status_code: int
    detail: str


def map_error(error: Exception) -> ErrorResponse:
    code = getattr(error, "code", None)
    if code in {"unavailable", "dependency_unavailable"}:
        return ErrorResponse(503, "Model provider is unavailable")
    if code == "deadline_exceeded":
        return ErrorResponse(504, "Explanation exceeded its deadline")
    if code in {"invalid_response", "unsupported_model"}:
        return ErrorResponse(502, "Model provider returned an unsupported response")
    if code == "configuration_invalid":
        return ErrorResponse(500, "Model provider configuration is invalid")
    if code in {"invalid_request", "data_invalid"}:
        return ErrorResponse(400, str(error))
    return ErrorResponse(500, "Local explanation failed")
