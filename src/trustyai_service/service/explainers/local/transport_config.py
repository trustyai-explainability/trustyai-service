"""Deployment-level HTTP transport configuration for model requests."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from .model_provider import HttpTransportConfig, ProviderConfigurationError

_MAX_PORT = 65_535
_MAX_BATCH_SIZE = 100_000


def _invalid_allowlist() -> None:
    raise ValueError


def _host_entry(value: str) -> str:
    value = value.strip().lower().rstrip(".")
    if not value:
        return ""
    try:
        parsed = urlsplit(f"//{value}")
        if (
            parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            _invalid_allowlist()
        host = parsed.hostname
        if not host:
            _invalid_allowlist()
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
    except ValueError as exc:
        msg = "Invalid outbound host allowlist entry"
        raise ProviderConfigurationError(msg) from exc
    if port is not None and not 1 <= port <= _MAX_PORT:
        msg = "Invalid outbound host allowlist port"
        raise ProviderConfigurationError(msg)
    return f"{host}:{port}" if port is not None else host


def get_transport_config() -> HttpTransportConfig:
    """Read transport settings at execution time, failing closed by default."""
    allowlist = frozenset(
        entry
        for entry in (
            _host_entry(item)
            for item in os.getenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "").split(",")
        )
        if entry
    )
    verify: bool | str = os.getenv("TRUSTYAI_EXPLAINER_CA_BUNDLE", "") or True
    if isinstance(verify, str) and not Path(verify).is_file():
        msg = "Configured CA bundle is unreadable"
        raise ProviderConfigurationError(msg)
    cert_path = os.getenv("TRUSTYAI_EXPLAINER_CLIENT_CERT", "")
    key_path = os.getenv("TRUSTYAI_EXPLAINER_CLIENT_KEY", "")
    if bool(cert_path) != bool(key_path):
        msg = "Client certificate and key must be configured together"
        raise ProviderConfigurationError(msg)
    cert = (cert_path, key_path) if cert_path and key_path else None
    if cert and (not Path(cert_path).is_file() or not Path(key_path).is_file()):
        msg = "Configured client certificate is unreadable"
        raise ProviderConfigurationError(msg)
    token = os.getenv("TRUSTYAI_EXPLAINER_AUTH_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        batch = int(os.getenv("TRUSTYAI_EXPLAINER_MAX_BATCH_SIZE", "1024"))
    except ValueError as exc:
        msg = "Invalid model batch size"
        raise ProviderConfigurationError(msg) from exc
    if not 1 <= batch <= _MAX_BATCH_SIZE:
        msg = "Invalid model batch size"
        raise ProviderConfigurationError(msg)
    return HttpTransportConfig(
        headers=headers,
        verify=verify,
        cert=cert,
        max_batch_size=batch,
        allowed_hosts=allowlist,
    )
