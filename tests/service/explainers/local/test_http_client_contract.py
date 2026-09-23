"""Contract tests for deployment-level local-explainer transport settings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from trustyai_service.service.explainers.local.model_provider import (
    HttpTransportConfig,
    ProviderConfigurationError,
)
from trustyai_service.service.explainers.local.transport_config import (
    get_transport_config,
)
from trustyai_service.service.explainers.local.types import PredictionSource


def _factory(source: PredictionSource) -> HttpTransportConfig | None:
    """Resolve deployment settings for one prediction source."""
    return get_transport_config(source)


def _set_model_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the minimum deployment policy needed by MODEL tests."""
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "model.example")


def test_model_transport_defaults_to_empty_headers_and_normal_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use no auth header, normal certificate verification, and safe client defaults."""
    _set_model_allowlist(monkeypatch)
    for name in (
        "TRUSTYAI_EXPLAINER_AUTH_TOKEN",
        "TRUSTYAI_EXPLAINER_CA_BUNDLE",
        "TRUSTYAI_EXPLAINER_CLIENT_CERT",
        "TRUSTYAI_EXPLAINER_CLIENT_KEY",
        "TRUSTYAI_EXPLAINER_MAX_BATCH_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)

    config = _factory(PredictionSource.MODEL)

    assert dict(config.headers) == {}
    assert config.verify is True
    assert config.cert is None
    assert config.max_batch_size == 1024
    assert config.follow_redirects is False
    assert config.trust_env is False


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_batch_limit_must_be_positive_integer(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Reject invalid deployment batch limits before a client is created."""
    _set_model_allowlist(monkeypatch)
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_MAX_BATCH_SIZE", value)

    with pytest.raises(ProviderConfigurationError, match="batch size"):
        _factory(PredictionSource.MODEL)


def test_batch_limit_is_read_at_execution_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read a changed deployment batch limit when the factory is called."""
    _set_model_allowlist(monkeypatch)
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_MAX_BATCH_SIZE", "7")

    config = _factory(PredictionSource.MODEL)

    assert config.max_batch_size == 7


def test_client_certificate_and_key_are_required_as_a_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject an incomplete mTLS configuration and retain a valid pair."""
    _set_model_allowlist(monkeypatch)
    cert = tmp_path / "client.crt"
    key = tmp_path / "client.key"
    cert.write_text("certificate")
    key.write_text("private key")

    monkeypatch.setenv("TRUSTYAI_EXPLAINER_CLIENT_CERT", str(cert))
    with pytest.raises(ProviderConfigurationError, match="certificate and key"):
        _factory(PredictionSource.MODEL)

    monkeypatch.setenv("TRUSTYAI_EXPLAINER_CLIENT_KEY", str(key))
    config = _factory(PredictionSource.MODEL)
    assert config.cert == (str(cert), str(key))


def test_configured_ca_bundle_is_used_for_tls_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Use a readable deployment CA bundle instead of the normal verifier."""
    _set_model_allowlist(monkeypatch)
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("ca bundle")
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_CA_BUNDLE", str(ca_bundle))

    config = _factory(PredictionSource.MODEL)

    assert config.verify == str(ca_bundle)


def test_model_requires_a_non_empty_host_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed when MODEL has no deployment outbound-host policy."""
    monkeypatch.delenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", raising=False)

    with pytest.raises(ProviderConfigurationError, match="allowlist"):
        _factory(PredictionSource.MODEL)


def test_model_is_the_default_prediction_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve MODEL transport settings when no prediction source is supplied."""
    _set_model_allowlist(monkeypatch)

    config = get_transport_config()

    assert config is not None
    assert config.allowed_hosts == frozenset({"model.example"})


def test_allowlist_matches_hosts_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize host case before comparing a model URL authority."""
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "MODEL.EXAMPLE.")

    config = _factory(PredictionSource.MODEL)

    assert config.allowed_hosts == frozenset({"model.example"})
    assert config.allows_url("https://MoDeL.ExAmPlE/v2/models/m")


def test_direct_transport_config_is_immutable_and_normalized() -> None:
    """Detach headers and normalize authorities before a provider sees them."""
    headers = {"Authorization": "Bearer secret-token"}
    config = HttpTransportConfig(
        headers=headers,
        allowed_hosts={"MODEL.EXAMPLE:443"},
    )
    headers["Authorization"] = "changed"

    assert dict(config.headers) == {"Authorization": "Bearer secret-token"}
    assert config.allowed_hosts == frozenset({"model.example:443"})
    assert "secret-token" not in repr(config)
    with pytest.raises(TypeError):
        config.headers["Other"] = "value"  # type: ignore[index]


def test_allowlist_rejects_an_unlisted_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep URL matching constrained to the configured deployment hosts."""
    _set_model_allowlist(monkeypatch)
    config = _factory(PredictionSource.MODEL)

    assert not config.allows_url("https://other.example/v2/models/m")


def test_allowlist_explicit_port_matches_only_that_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an explicit allowlist port to equal the URL authority port."""
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "model.example:8443")
    config = _factory(PredictionSource.MODEL)

    assert config.allows_url("https://model.example:8443/v2/models/m")
    assert not config.allows_url("https://model.example:443/v2/models/m")
    assert not config.allows_url("https://model.example/v2/models/m")


def test_allowlist_normalizes_trailing_dot_before_explicit_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize a DNS trailing dot on the parsed host before retaining its port."""
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "model.example.:8443")

    config = _factory(PredictionSource.MODEL)

    assert config.allowed_hosts == frozenset({"model.example:8443"})
    assert config.allows_url("https://model.example:8443/v2/models/m")


def test_allowlist_normalizes_default_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat an omitted HTTP or HTTPS port as its scheme default."""
    monkeypatch.setenv(
        "TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "model.example:80,secure.example:443"
    )
    config = _factory(PredictionSource.MODEL)

    assert config.allows_url("http://model.example/v2/models/m")
    assert config.allows_url("http://model.example:80/v2/models/m")
    assert config.allows_url("https://secure.example/v2/models/m")
    assert config.allows_url("https://secure.example:443/v2/models/m")


def test_private_host_allowlist_requires_general_host_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require private-address consent to be narrower than the general policy."""
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "model.example")
    monkeypatch.setenv(
        "TRUSTYAI_EXPLAINER_ALLOWED_PRIVATE_HOSTS", "private.model.example"
    )

    with pytest.raises(ProviderConfigurationError, match=r"Private.*allowlist"):
        _factory(PredictionSource.MODEL)


def test_private_host_allowlist_is_read_at_execution_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permit explicitly configured in-cluster host authorities."""
    monkeypatch.setenv(
        "TRUSTYAI_EXPLAINER_ALLOWED_HOSTS",
        "private.model.example,public.model.example",
    )
    monkeypatch.setenv(
        "TRUSTYAI_EXPLAINER_ALLOWED_PRIVATE_HOSTS", "private.model.example"
    )

    config = _factory(PredictionSource.MODEL)

    assert config.allowed_private_hosts == frozenset({"private.model.example"})
    assert config.allows_private_url("https://private.model.example/v2/models/m")
    assert not config.allows_private_url("https://public.model.example/v2/models/m")


def test_allowlist_normalizes_ipv6_brackets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one bracketed representation for IPv6 host authorities."""
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", "[2001:DB8::1]:8443,[::1]")
    config = _factory(PredictionSource.MODEL)

    assert config.allowed_hosts == frozenset({"[2001:db8::1]:8443", "[::1]"})
    assert config.allows_url("https://[2001:db8::1]:8443/v2/models/m")
    assert config.allows_url("http://[::1]:8080/v2/models/m")


@pytest.mark.parametrize(
    "value",
    [
        "model.example/path",
        "model.example?token=secret",
        "model.example#fragment",
        "user:password@model.example",
        "*.model.example",
        "model.*",
    ],
)
def test_allowlist_rejects_non_authority_and_wildcard_entries(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Do not turn URL components, credentials, or wildcards into host policy."""
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", value)

    with pytest.raises(ProviderConfigurationError, match="allowlist"):
        _factory(PredictionSource.MODEL)


@pytest.mark.parametrize(
    "value",
    [" model.example", "model.example ", "\nmodel.example", "model.example\n"],
)
def test_allowlist_rejects_boundary_whitespace_and_control_characters(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Reject unsafe entry boundaries instead of silently trimming them."""
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", value)

    with pytest.raises(ProviderConfigurationError, match="allowlist"):
        _factory(PredictionSource.MODEL)


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@model.example/v2/models/m",  # pragma: allowlist secret
        "https://@model.example/v2/models/m",
        "ftp://model.example/v2/models/m",
        "//model.example/v2/models/m",
    ],
)
def test_allowlist_matching_rejects_unsafe_url_authorities(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Do not treat credentials or unsupported URL schemes as model authorities."""
    _set_model_allowlist(monkeypatch)
    config = _factory(PredictionSource.MODEL)

    assert not config.allows_url(url)


@pytest.mark.parametrize(
    "url",
    [
        " https://model.example/v2/models/m",
        "https://model.example/v2/models/m ",
        "https://model.example\n/v2/models/m",
        "\thttps://model.example/v2/models/m",
    ],
)
def test_allowlist_matching_rejects_boundary_whitespace_and_control_characters(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    """Reject unsafe URL characters before urlsplit can normalize them away."""
    _set_model_allowlist(monkeypatch)
    config = _factory(PredictionSource.MODEL)

    assert not config.allows_url(url)


def test_bearer_token_is_a_service_level_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct the only supported auth header from deployment configuration."""
    _set_model_allowlist(monkeypatch)
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_AUTH_TOKEN", "secret-token")

    config = _factory(PredictionSource.MODEL)

    assert dict(config.headers) == {"Authorization": "Bearer secret-token"}


def test_credential_bearing_header_requires_an_allowlist() -> None:
    """Never permit credentials on a transport with arbitrary outbound hosts."""
    with pytest.raises(ProviderConfigurationError, match="allowlist"):
        HttpTransportConfig(
            headers={"x-api-key": "secret"},
            allowed_hosts=frozenset(),
        )


def test_malformed_credential_header_name_cannot_bypass_allowlist_requirement() -> None:
    """Reject an invalid credential header name before credential classification."""
    with pytest.raises(ProviderConfigurationError, match="header name"):
        HttpTransportConfig(
            headers={"Authorization ": "secret"},
            allowed_hosts=frozenset(),
        )


def test_bearer_token_without_an_allowlist_fails_without_leaking_the_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject credential-bearing deployment settings without exposing the token."""
    monkeypatch.delenv("TRUSTYAI_EXPLAINER_ALLOWED_HOSTS", raising=False)
    credential_value = "do-not-log-this-token"
    monkeypatch.setenv("TRUSTYAI_EXPLAINER_AUTH_TOKEN", credential_value)

    with pytest.raises(ProviderConfigurationError) as raised:
        _factory(PredictionSource.MODEL)

    assert credential_value not in str(raised.value)


def test_surrogate_does_not_resolve_transport_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip all deployment transport reads for an explicit provider-free source."""

    def fail_getenv(*_args: object, **_kwargs: object) -> str:
        pytest.fail("SURROGATE resolved deployment transport settings")

    monkeypatch.setattr(
        "trustyai_service.service.explainers.local.transport_config.os.getenv",
        fail_getenv,
    )

    assert get_transport_config(PredictionSource.SURROGATE) is None


def test_transport_config_import_does_not_import_optional_http_integrations() -> None:
    """Keep transport configuration import-safe without HTTP or endpoint packages."""
    repository_root = Path(__file__).parents[4]
    script = """
import builtins
import sys

sys.path.insert(0, "src")
forbidden = (
    "httpx2",
    "fastapi",
    "lime",
    "shap",
)
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if any(name == module or name.startswith(module + ".") for module in forbidden):
        raise AssertionError(f"forbidden import: {name}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import trustyai_service.service.explainers.local.transport_config  # noqa: E402
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
