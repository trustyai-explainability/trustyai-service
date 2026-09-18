"""Protocol and errors for local model prediction providers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from .types import TaskType


@dataclass(frozen=True)
class KServeModelSpec:
    """Validated identity and tensor selectors for one KServe model."""

    base_url: str
    model_name: str
    model_version: str | None
    input_name: str | None
    output_name: str | None
    task: TaskType


@dataclass(frozen=True)
class PredictionMetadata:
    """Selected numeric input and output tensor metadata."""

    input_name: str
    output_name: str
    input_datatype: str
    output_datatype: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


class PredictionProvider(Protocol):
    """Synchronous callable provider owned by an explanation worker."""

    @property
    def metadata(self) -> PredictionMetadata: ...

    def predict(
        self, inputs: np.ndarray, *, timeout_seconds: float | None = None
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class ProviderError(RuntimeError):
    """Stable, endpoint-neutral provider failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class DependencyUnavailableError(ProviderError):
    def __init__(self, message: str = "Model HTTP dependency is unavailable") -> None:
        super().__init__("dependency_unavailable", message)


class ProviderUnavailableError(ProviderError):
    def __init__(self, message: str = "Model endpoint is unavailable") -> None:
        super().__init__("unavailable", message)


class ProviderDeadlineError(ProviderError):
    def __init__(self, message: str = "Model request deadline exceeded") -> None:
        super().__init__("deadline_exceeded", message)


class ProviderInvalidRequestError(ProviderError):
    def __init__(self, message: str = "Invalid model request") -> None:
        super().__init__("invalid_request", message)


class ProviderUnsupportedModelError(ProviderError):
    def __init__(self, message: str = "Unsupported model contract") -> None:
        super().__init__("unsupported_model", message)


class ProviderInvalidResponseError(ProviderError):
    def __init__(self, message: str = "Invalid model response") -> None:
        super().__init__("invalid_response", message)


class ProviderConfigurationError(ProviderError):
    def __init__(self, message: str = "Invalid model transport configuration") -> None:
        super().__init__("configuration_invalid", message)


class LocalDataError(ValueError):
    """Invalid stored data for a requested local explanation."""

    code = "data_invalid"


class LocalExecutionError(RuntimeError):
    """Unexpected local algorithm or estimator failure."""

    code = "execution_failed"


@dataclass(frozen=True)
class HttpTransportConfig:
    """Deployment-owned transport settings. Request bodies cannot override these."""

    headers: Mapping[str, str]
    verify: bool | str = True
    cert: str | tuple[str, str] | None = None
    max_batch_size: int = 1024
    allowed_hosts: frozenset[str] | None = None
    follow_redirects: bool = False
    trust_env: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.max_batch_size <= 100_000:
            raise ValueError("max_batch_size must be between 1 and 100000")
        if any(
            str(name).lower() in {"host", "content-length"} for name in self.headers
        ):
            raise ValueError("Host and Content-Length are transport-owned headers")
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
