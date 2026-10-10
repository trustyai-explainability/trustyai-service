"""Protocol and errors for local model prediction providers."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from .types import TaskType

_MAX_BATCH_SIZE = 100_000


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
    def metadata(self) -> PredictionMetadata:
        """Return negotiated input and output tensor metadata."""
        ...

    def predict(
        self, inputs: np.ndarray, *, timeout_seconds: float | None = None
    ) -> np.ndarray:
        """Predict for a batch of normalized numeric input rows."""
        ...

    def close(self) -> None:
        """Release the provider's transport resources."""
        ...


class ProviderError(RuntimeError):
    """Stable, endpoint-neutral provider failure."""

    def __init__(self, code: str, message: str) -> None:
        """Create a provider error with a stable mapping code."""
        self.code = code
        super().__init__(message)


class DependencyUnavailableError(ProviderError):
    """The optional HTTP client dependency is not installed."""

    def __init__(self, message: str = "Model HTTP dependency is unavailable") -> None:
        """Create a dependency-unavailable provider error."""
        super().__init__("dependency_unavailable", message)


class ProviderUnavailableError(ProviderError):
    """The model endpoint cannot currently be reached."""

    def __init__(self, message: str = "Model endpoint is unavailable") -> None:
        """Create an unavailable-provider error."""
        super().__init__("unavailable", message)


class ProviderDeadlineError(ProviderError):
    """The model request exceeded the explanation deadline."""

    def __init__(self, message: str = "Model request deadline exceeded") -> None:
        """Create a provider-deadline error."""
        super().__init__("deadline_exceeded", message)


class ProviderInvalidRequestError(ProviderError):
    """The requested model input or selector is invalid."""

    def __init__(self, message: str = "Invalid model request") -> None:
        """Create an invalid-request provider error."""
        super().__init__("invalid_request", message)


class ProviderUnsupportedModelError(ProviderError):
    """The model metadata or contract is unsupported."""

    def __init__(self, message: str = "Unsupported model contract") -> None:
        """Create an unsupported-model provider error."""
        super().__init__("unsupported_model", message)


class ProviderInvalidResponseError(ProviderError):
    """The model returned an invalid or unsafe response."""

    def __init__(self, message: str = "Invalid model response") -> None:
        """Create an invalid-response provider error."""
        super().__init__("invalid_response", message)


class ProviderConfigurationError(ProviderError):
    """Deployment transport configuration is invalid."""

    def __init__(self, message: str = "Invalid model transport configuration") -> None:
        """Create a configuration provider error."""
        super().__init__("configuration_invalid", message)


class LocalDataError(ValueError):
    """Invalid stored data for a requested local explanation."""

    code = "data_invalid"


class LocalDataNotFoundError(LocalDataError):
    """Stored model data required for a local explanation is missing."""

    code = "data_missing"


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
        """Validate deployment-owned transport settings."""
        if not 1 <= self.max_batch_size <= _MAX_BATCH_SIZE:
            msg = "max_batch_size must be between 1 and 100000"
            raise ValueError(msg)
        if any(
            str(name).lower() in {"host", "content-length"} for name in self.headers
        ):
            msg = "Host and Content-Length are transport-owned headers"
            raise ValueError(msg)
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
