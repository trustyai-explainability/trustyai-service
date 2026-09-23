"""Deployment-level transport configuration for local model predictions."""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ResolverTimeoutError
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from urllib.parse import urlsplit

from .model_provider import (
    HttpTransportConfig,
    ProviderConfigurationError,
    ProviderDeadlineError,
    ProviderInvalidRequestError,
    ProviderUnavailableError,
)
from .types import PredictionSource

if TYPE_CHECKING:
    from collections.abc import Iterable

_MAX_PORT = 65_535
_MAX_BATCH_SIZE = 100_000
_CONTROL_CHAR_LIMIT = 32
_DEL_CHAR = 127
_MAX_DNS_RESOLVERS = 4
_SOCKADDR_MIN_LENGTH = 5
_DNS_RESOLVER = ThreadPoolExecutor(
    max_workers=_MAX_DNS_RESOLVERS,
    thread_name_prefix="trustyai-model-dns",
)
_DNS_RESOLVER_SLOTS = threading.BoundedSemaphore(_MAX_DNS_RESOLVERS)


def _configuration_error(message: str, cause: Exception | None = None) -> NoReturn:
    """Raise the canonical provider configuration error."""
    if cause is None:
        raise ProviderConfigurationError(message)
    raise ProviderConfigurationError(message) from cause


def _has_unsafe_allowlist_character(value: str) -> bool:
    """Return whether an entry contains unsafe authority characters."""
    return any(
        ord(character) < _CONTROL_CHAR_LIMIT
        or ord(character) == _DEL_CHAR
        or character.isspace()
        or character in "\\%*?"
        for character in value
    )


def _has_unsafe_url_character(value: str) -> bool:
    """Return whether a URL contains whitespace or control characters."""
    return any(
        ord(character) < _CONTROL_CHAR_LIMIT
        or ord(character) == _DEL_CHAR
        or character.isspace()
        for character in value
    )


def _parse_allowlist_authority(value: str) -> tuple[str, int | None]:
    """Parse a host authority after its raw characters have been checked."""
    try:
        parsed = urlsplit(f"//{value}")
        if parsed.path or parsed.query or parsed.fragment:
            _configuration_error("Invalid outbound host allowlist entry")
        if parsed.username or parsed.password:
            _configuration_error("Invalid outbound host allowlist entry")
        host = parsed.hostname
        if not host:
            _configuration_error("Invalid outbound host allowlist entry")
        port = parsed.port
    except (ValueError, TypeError) as exc:
        _configuration_error("Invalid outbound host allowlist entry", exc)

    if ":" in host and not value.startswith("["):
        _configuration_error("Invalid outbound host allowlist entry")
    if port is not None and not 1 <= port <= _MAX_PORT:
        _configuration_error("Invalid outbound host allowlist port")
    return host, port


def normalize_allowlist_entry(value: str) -> str:
    """Normalize one host authority and reject non-authority syntax."""
    if not isinstance(value, str):
        _configuration_error("Invalid outbound host allowlist entry")
    if not value:
        return ""
    if _has_unsafe_allowlist_character(value):
        _configuration_error("Invalid outbound host allowlist entry")
    raw = value.lower()
    if (
        "://" in raw
        or raw.startswith("//")
        or any(marker in raw for marker in ("?", "#", "@"))
        or raw.endswith(":")
    ):
        _configuration_error("Invalid outbound host allowlist entry")

    host, port = _parse_allowlist_authority(raw)
    host = host.rstrip(".")
    if not host:
        _configuration_error("Invalid outbound host allowlist entry")
    if any(character in host for character in "*?"):
        _configuration_error("Invalid outbound host allowlist entry")

    normalized_host = f"[{host}]" if ":" in host else host
    return f"{normalized_host}:{port}" if port is not None else normalized_host


def normalize_host_allowlist(value: str | Iterable[str]) -> frozenset[str]:
    """Normalize a comma-separated or iterable host allowlist."""
    values = value.split(",") if isinstance(value, str) else value
    entries: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            _configuration_error("Invalid outbound host allowlist entry")
        normalized = normalize_allowlist_entry(item)
        if normalized:
            entries.add(normalized)
    return frozenset(entries)


def _url_authority(url: str) -> tuple[str, int] | None:
    """Extract a normalized HTTP(S) URL authority for policy matching."""
    if not isinstance(url, str) or _has_unsafe_url_character(url):
        return None
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        valid_authority = (
            scheme in {"http", "https"}
            and bool(parsed.netloc)
            and not parsed.username
            and not parsed.password
            and "@" not in parsed.netloc
        )
        if not valid_authority:
            return None
        host = parsed.hostname
        port = parsed.port
    except (ValueError, TypeError):
        return None

    if not host:
        return None
    host = host.lower().rstrip(".")
    if not host or _has_unsafe_allowlist_character(host):
        return None
    normalized_host = f"[{host}]" if ":" in host else host
    effective_port = port if port is not None else (443 if scheme == "https" else 80)
    return normalized_host, effective_port


def is_url_allowed(url: str, allowed_hosts: Iterable[str]) -> bool:
    """Match only the URL authority against normalized host policy entries."""
    authority = _url_authority(url)
    if authority is None:
        return False
    host, port = authority
    try:
        entries = normalize_host_allowlist(allowed_hosts)
    except ProviderConfigurationError:
        return False
    return host in entries or f"{host}:{port}" in entries


def _is_restricted_address(value: str) -> bool:
    """Return whether an address is not globally routable.

    An explicit IP literal may still be used when the operator allowlists it.
    DNS names are stricter: every resolved address must be globally routable so
    a rebinding cannot redirect an allowlisted name to a local or metadata
    service.
    """
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return True
    return not address.is_global


def _is_allowed_private_address(value: str) -> bool:
    """Return whether an address is safe for an explicit private-host opt-in."""
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    return bool(
        address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_unspecified
        and not address.is_multicast
        and not address.is_reserved
    )


def _resolve_hostname(
    host: str,
    port: int,
    timeout_seconds: float | None,
) -> list[object]:
    """Resolve one hostname without allowing DNS to hold an explainer worker."""
    if timeout_seconds is not None and (
        not isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise ProviderDeadlineError

    resolver_slots = _DNS_RESOLVER_SLOTS
    if not resolver_slots.acquire(blocking=False):
        message = "Model DNS resolution capacity is exhausted"
        raise ProviderUnavailableError(message)
    try:
        future = _DNS_RESOLVER.submit(
            socket.getaddrinfo,
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    except Exception:
        resolver_slots.release()
        raise
    future.add_done_callback(lambda _future: resolver_slots.release())
    try:
        return future.result(timeout=timeout_seconds)
    except ResolverTimeoutError as exc:
        future.cancel()
        raise ProviderDeadlineError from exc
    except OSError as exc:
        message = "Model hostname could not be resolved"
        raise ProviderUnavailableError(message) from exc


def _extract_resolved_addresses(results: list[object]) -> list[str]:
    """Extract unique IP addresses from getaddrinfo results."""
    addresses: list[str] = []
    for result in results:
        if not isinstance(result, tuple) or len(result) < _SOCKADDR_MIN_LENGTH:
            continue
        sockaddr = result[4]
        if not isinstance(sockaddr, tuple) or not sockaddr:
            continue
        address = sockaddr[0]
        if isinstance(address, str) and address not in addresses:
            addresses.append(address)
    return addresses


def _validate_address_policy(
    addresses: Iterable[str],
    *,
    allow_private_addresses: bool,
    restricted_message: str,
) -> tuple[str, ...]:
    """Reject restricted addresses unless the exact authority was opted in."""
    address_list = list(addresses)
    restricted_addresses = [
        address for address in address_list if _is_restricted_address(address)
    ]
    if restricted_addresses and (
        not allow_private_addresses
        or any(
            not _is_allowed_private_address(address) for address in restricted_addresses
        )
    ):
        raise ProviderInvalidRequestError(restricted_message) from None
    return tuple(address_list)


def resolve_outbound_addresses(
    url: str,
    *,
    timeout_seconds: float | None = None,
    allow_private_addresses: bool = False,
) -> tuple[str, ...]:
    """Resolve and validate all model-authority addresses for pinned connections."""
    invalid_authority = "Model URL has an invalid authority"
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ProviderInvalidRequestError(invalid_authority) from exc
    if not host:
        raise ProviderInvalidRequestError(invalid_authority)

    try:
        ipaddress.ip_address(host)
    except ValueError:
        results = _resolve_hostname(
            host,
            port if port is not None else (443 if parsed.scheme == "https" else 80),
            timeout_seconds,
        )
        addresses = _extract_resolved_addresses(results)
        if not addresses:
            message = "Model hostname could not be resolved"
            raise ProviderUnavailableError(message) from None
        return _validate_address_policy(
            addresses,
            allow_private_addresses=allow_private_addresses,
            restricted_message="Model hostname resolves to a restricted address",
        )

    return (host,)


def resolve_outbound_address(
    url: str,
    *,
    timeout_seconds: float | None = None,
    allow_private_addresses: bool = False,
) -> str:
    """Resolve one model-authority address for compatibility with older callers."""
    return resolve_outbound_addresses(
        url,
        timeout_seconds=timeout_seconds,
        allow_private_addresses=allow_private_addresses,
    )[0]


def _read_transport_config(*, require_allowlist: bool) -> HttpTransportConfig:
    """Read deployment settings at execution creation."""
    allowlist = normalize_host_allowlist(
        os.getenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "")
    )
    if require_allowlist and not allowlist:
        _configuration_error("Outbound model host allowlist is required")
    private_allowlist = normalize_host_allowlist(
        os.getenv("TRUSTYAI_EXPLAINER_ALLOWED_PRIVATE_HOSTS", "")
    )

    verify: bool | str = os.getenv("TRUSTYAI_EXPLAINER_CA_BUNDLE", "") or True
    if isinstance(verify, str) and not Path(verify).is_file():
        _configuration_error("Configured CA bundle is unreadable")

    cert_path = os.getenv("TRUSTYAI_EXPLAINER_CLIENT_CERT", "")
    key_path = os.getenv("TRUSTYAI_EXPLAINER_CLIENT_KEY", "")
    if bool(cert_path) != bool(key_path):
        _configuration_error("Client certificate and key must be configured together")
    cert = (cert_path, key_path) if cert_path and key_path else None
    if cert and (not Path(cert_path).is_file() or not Path(key_path).is_file()):
        _configuration_error("Configured client certificate is unreadable")

    token = os.getenv("TRUSTYAI_EXPLAINER_AUTH_TOKEN", "")
    if token and (
        token.lower().startswith("bearer ")
        or any(
            ord(character) < _CONTROL_CHAR_LIMIT
            or ord(character) == _DEL_CHAR
            or character.isspace()
            for character in token
        )
    ):
        _configuration_error("Configured model authorization token is invalid")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    raw_batch = os.getenv("TRUSTYAI_EXPLAINER_MAX_BATCH_SIZE", "1024")
    try:
        batch = int(raw_batch)
    except (TypeError, ValueError) as exc:
        _configuration_error("Invalid model batch size", exc)
    if not 1 <= batch <= _MAX_BATCH_SIZE:
        _configuration_error("Invalid model batch size")

    return HttpTransportConfig(
        headers=headers,
        verify=verify,
        cert=cert,
        max_batch_size=batch,
        allowed_hosts=allowlist,
        allowed_private_hosts=private_allowlist,
        follow_redirects=False,
        trust_env=False,
    )


def get_transport_config(
    prediction_source: PredictionSource = PredictionSource.MODEL,
) -> HttpTransportConfig | None:
    """Resolve deployment transport settings only for MODEL executions."""
    if prediction_source == PredictionSource.SURROGATE:
        return None
    if prediction_source != PredictionSource.MODEL:
        _configuration_error("Invalid prediction source")
    return _read_transport_config(require_allowlist=True)


def resolve_transport_config(
    prediction_source: PredictionSource,
) -> HttpTransportConfig | None:
    """Resolve transport settings after selecting the prediction source."""
    return get_transport_config(prediction_source)
