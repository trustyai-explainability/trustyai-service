"""Protocol and transport-neutral errors for local model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np


_MAX_BATCH_SIZE = 100_000
_CREDENTIAL_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
    }
)
_TRANSPORT_OWNED_HEADERS = frozenset({"host", "content-length"})
_HEADER_NAME_SYMBOLS = frozenset("!#$%&'*+-.^_`|~")
_CONTROL_CHAR_LIMIT = 32
_DEL_CHAR = 127
_CERTIFICATE_PAIR_LENGTH = 2


def _has_control_character(value: str) -> bool:
    """Return whether a header value contains an HTTP control character."""
    return any(
        ord(character) < _CONTROL_CHAR_LIMIT or ord(character) == _DEL_CHAR
        for character in value
    )


def _validate_transport_basics(
    *,
    verify: object,
    cert: object,
    max_batch_size: object,
    follow_redirects: bool,
    trust_env: bool,
) -> None:
    """Validate scalar transport settings before freezing the configuration."""
    if (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, int)
        or not 1 <= max_batch_size <= _MAX_BATCH_SIZE
    ):
        msg = "Invalid model batch size"
        raise ProviderConfigurationError(msg)
    if follow_redirects or trust_env:
        msg = "Redirects and ambient proxy settings must remain disabled"
        raise ProviderConfigurationError(msg)
    if not isinstance(verify, (bool, str)) or verify == "":
        msg = "Invalid TLS verification configuration"
        raise ProviderConfigurationError(msg)
    if cert is not None and not isinstance(cert, (str, tuple)):
        msg = "Invalid client certificate configuration"
        raise ProviderConfigurationError(msg)
    if isinstance(cert, tuple) and (
        len(cert) != _CERTIFICATE_PAIR_LENGTH
        or not all(isinstance(path, str) and path for path in cert)
    ):
        msg = "Client certificate and key must be configured together"
        raise ProviderConfigurationError(msg)


def _normalize_transport_header_name(name: object) -> str:
    """Validate an HTTP field-name and return its canonical comparison form."""
    if (
        not isinstance(name, str)
        or not name
        or any(
            not (
                "A" <= character <= "Z"
                or "a" <= character <= "z"
                or "0" <= character <= "9"
                or character in _HEADER_NAME_SYMBOLS
            )
            for character in name
        )
    ):
        msg = "Invalid model transport header name"
        raise ProviderConfigurationError(msg)
    return name.lower()


def _normalize_transport_headers(headers: object) -> dict[str, str]:
    """Validate and copy transport headers into a detached mapping."""
    try:
        header_items = tuple(headers.items())  # type: ignore[union-attr]
    except AttributeError as exc:
        msg = "Invalid model transport headers"
        raise ProviderConfigurationError(msg) from exc

    normalized: dict[str, str] = {}
    seen: set[str] = set()
    for name, value in header_items:
        if not isinstance(name, str) or not isinstance(value, str):
            msg = "Invalid model transport headers"
            raise ProviderConfigurationError(msg)
        lowered_name = _normalize_transport_header_name(name)
        if lowered_name in seen:
            msg = "Duplicate model transport header"
            raise ProviderConfigurationError(msg)
        if _has_control_character(value):
            msg = "Invalid model transport header value"
            raise ProviderConfigurationError(msg)
        if lowered_name in _TRANSPORT_OWNED_HEADERS:
            msg = "Host and Content-Length are transport-owned headers"
            raise ProviderConfigurationError(msg)
        seen.add(lowered_name)
        normalized[name] = value
    return normalized


def _normalize_allowed_hosts(allowed_hosts: object) -> frozenset[str]:
    """Detach an optional allowlist while preserving factory normalization."""
    try:
        return frozenset(allowed_hosts or ())
    except TypeError as exc:
        msg = "Invalid outbound host allowlist"
        raise ProviderConfigurationError(msg) from exc


def _reject_credentials_without_allowlist(
    headers: dict[str, str], allowed_hosts: frozenset[str]
) -> None:
    """Prevent credentials from being sent to an unconstrained authority."""
    if (
        any(
            _normalize_transport_header_name(name) in _CREDENTIAL_HEADERS
            for name in headers
        )
        and not allowed_hosts
    ):
        msg = "Credential-bearing transport headers require an outbound host allowlist"
        raise ProviderConfigurationError(msg)


@dataclass(frozen=True)
class PredictionMetadata:
    """Selected input and output tensor metadata for a prediction provider."""

    input_name: str
    output_name: str
    input_datatype: str
    output_datatype: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@runtime_checkable
class PredictionProvider(Protocol):
    """Synchronous provider contract owned by one explanation execution."""

    @property
    def metadata(self) -> PredictionMetadata:
        """Return the negotiated input and output tensor metadata."""
        ...

    def predict(
        self,
        inputs: np.ndarray,
        *,
        timeout_seconds: float | None = None,
    ) -> np.ndarray:
        """Predict outputs for a batch of inputs."""
        ...

    def close(self) -> None:
        """Release provider resources; implementations must be idempotent."""
        ...


class ProviderError(RuntimeError):
    """Stable, endpoint-neutral failure raised by a prediction provider."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        """Create a provider error with a stable mapping code."""
        self.code = code
        super().__init__(message)


class DependencyUnavailableError(ProviderError):
    """The optional provider dependency is unavailable."""

    code = "dependency_unavailable"

    def __init__(self, message: str = "Model HTTP dependency is unavailable") -> None:
        """Create a dependency-unavailable provider error."""
        super().__init__(self.code, message)


class ProviderConfigurationError(ProviderError):
    """Provider configuration is invalid."""

    code = "configuration_invalid"

    def __init__(
        self,
        message: str = "Invalid model transport configuration",
    ) -> None:
        """Create a configuration provider error."""
        super().__init__(self.code, message)


class ProviderInvalidRequestError(ProviderError):
    """The requested model input or selector is invalid."""

    code = "invalid_request"

    def __init__(self, message: str = "Invalid model request") -> None:
        """Create an invalid-request provider error."""
        super().__init__(self.code, message)


class ProviderUnavailableError(ProviderError):
    """The upstream model endpoint is unavailable."""

    code = "unavailable"

    def __init__(self, message: str = "Model endpoint is unavailable") -> None:
        """Create an unavailable-provider error."""
        super().__init__(self.code, message)


class ProviderInvalidResponseError(ProviderError):
    """The upstream model returned an invalid response."""

    code = "invalid_response"

    def __init__(self, message: str = "Invalid model response") -> None:
        """Create an invalid-response provider error."""
        super().__init__(self.code, message)


class ProviderUnsupportedModelError(ProviderError):
    """The requested model contract is unsupported."""

    code = "unsupported_model"

    def __init__(self, message: str = "Unsupported model contract") -> None:
        """Create an unsupported-model provider error."""
        super().__init__(self.code, message)


class ProviderDeadlineError(ProviderError):
    """The model request exceeded its deadline."""

    code = "deadline_exceeded"

    def __init__(self, message: str = "Model request deadline exceeded") -> None:
        """Create a provider-deadline error."""
        super().__init__(self.code, message)


@dataclass(frozen=True)
class HttpTransportConfig:
    """Deployment-owned transport settings for one model execution."""

    headers: Mapping[str, str] = field(repr=False)
    verify: bool | str = True
    cert: str | tuple[str, str] | None = None
    max_batch_size: int = 1024
    allowed_hosts: frozenset[str] | None = None
    allowed_private_hosts: frozenset[str] | None = None
    follow_redirects: bool = False
    trust_env: bool = False

    def __post_init__(self) -> None:
        """Validate and freeze settings that the request body cannot override."""
        _validate_transport_basics(
            verify=self.verify,
            cert=self.cert,
            max_batch_size=self.max_batch_size,
            follow_redirects=self.follow_redirects,
            trust_env=self.trust_env,
        )
        normalized_headers = _normalize_transport_headers(self.headers)
        normalized_hosts = _normalize_allowed_hosts(self.allowed_hosts)
        if normalized_hosts:
            from .transport_config import normalize_host_allowlist  # noqa: PLC0415

            normalized_hosts = normalize_host_allowlist(normalized_hosts)
        normalized_private_hosts = _normalize_allowed_hosts(self.allowed_private_hosts)
        if normalized_private_hosts:
            from .transport_config import normalize_host_allowlist  # noqa: PLC0415

            normalized_private_hosts = normalize_host_allowlist(
                normalized_private_hosts
            )
            if not normalized_private_hosts.issubset(normalized_hosts):
                msg = (
                    "Private outbound hosts must also be in the outbound host allowlist"
                )
                raise ProviderConfigurationError(msg)
        _reject_credentials_without_allowlist(normalized_headers, normalized_hosts)

        object.__setattr__(self, "headers", MappingProxyType(normalized_headers))
        object.__setattr__(self, "allowed_hosts", normalized_hosts)
        object.__setattr__(self, "allowed_private_hosts", normalized_private_hosts)

    def allows_url(self, url: str) -> bool:
        """Return whether the URL authority is covered by the host allowlist."""
        from .transport_config import is_url_allowed  # noqa: PLC0415

        return is_url_allowed(url, self.allowed_hosts)

    def is_url_allowed(self, url: str) -> bool:
        """Alias for URL-authority allowlist matching."""
        return self.allows_url(url)

    def is_host_allowed(self, url: str) -> bool:
        """Alias for URL-authority allowlist matching."""
        return self.allows_url(url)

    def allows_private_url(self, url: str) -> bool:
        """Return whether a URL has explicit private-address deployment consent."""
        from .transport_config import is_url_allowed  # noqa: PLC0415

        return is_url_allowed(url, self.allowed_private_hosts)

    @property
    def has_sensitive_credentials(self) -> bool:
        """Return whether this transport carries credentials that require TLS."""
        return self.cert is not None or any(
            _normalize_transport_header_name(name) in _CREDENTIAL_HEADERS
            for name in self.headers
        )
