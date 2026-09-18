"""Deployment-level HTTP transport configuration for model requests."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from .model_provider import HttpTransportConfig, ProviderConfigurationError


def _host_entry(value: str) -> str:
    value = value.strip().lower().rstrip(".")
    if not value:
        return ""
    try:
        parsed = urlsplit(f"//{value}")
        if parsed.path not in ("", "/") or parsed.username or parsed.password:
            raise ValueError
        host = parsed.hostname
        if not host:
            raise ValueError
        if ":" in host:
            host = f"[{host}]"
        port = parsed.port
    except ValueError as exc:
        raise ProviderConfigurationError(
            "Invalid outbound host allowlist entry"
        ) from exc
    if port is not None and not 1 <= port <= 65535:
        raise ProviderConfigurationError("Invalid outbound host allowlist port")
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
        raise ProviderConfigurationError("Configured CA bundle is unreadable")
    cert_path = os.getenv("TRUSTYAI_EXPLAINER_CLIENT_CERT", "")
    key_path = os.getenv("TRUSTYAI_EXPLAINER_CLIENT_KEY", "")
    if bool(cert_path) != bool(key_path):
        raise ProviderConfigurationError(
            "Client certificate and key must be configured together"
        )
    cert = (cert_path, key_path) if cert_path and key_path else None
    if cert and (not Path(cert_path).is_file() or not Path(key_path).is_file()):
        raise ProviderConfigurationError("Configured client certificate is unreadable")
    token = os.getenv("TRUSTYAI_EXPLAINER_AUTH_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        batch = int(os.getenv("TRUSTYAI_EXPLAINER_MAX_BATCH_SIZE", "1024"))
    except ValueError as exc:
        raise ProviderConfigurationError("Invalid model batch size") from exc
    if not 1 <= batch <= 100_000:
        raise ProviderConfigurationError("Invalid model batch size")
    return HttpTransportConfig(
        headers=headers,
        verify=verify,
        cert=cert,
        max_batch_size=batch,
        allowed_hosts=allowlist,
    )
