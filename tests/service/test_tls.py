"""Tests for TLS crypto policy compliance."""

import ssl
import subprocess
from pathlib import Path

import pytest

from trustyai_service.service.tls import (
    PolicyAwareConfig,
    TLSSettings,
    serving_certificates,
)


def _generate_self_signed_cert(tmp_path: Path) -> tuple[str, str]:
    """Generate a self-signed cert and key for testing."""
    cert_file = tmp_path / "tls.crt"
    key_file = tmp_path / "tls.key"
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
        ],
        check=True,
        capture_output=True,
    )
    return str(cert_file), str(key_file)


def _make_tls_config(cert_file: str, key_file: str) -> PolicyAwareConfig:
    """Create a PolicyAwareConfig with cert and key set."""
    config = PolicyAwareConfig()
    config.certfile = cert_file
    config.keyfile = key_file
    return config


class TestPolicyAwareConfig:
    """Tests for PolicyAwareConfig SSL context creation."""

    def test_returns_none_when_ssl_disabled(self) -> None:
        """No SSL context when TLS is not configured."""
        config = PolicyAwareConfig()
        assert config.create_ssl_context() is None

    def test_creates_context_with_certs(self, tmp_path: Path) -> None:
        """SSL context is created when cert and key are provided."""
        cert_file, key_file = _generate_self_signed_cert(tmp_path)
        config = _make_tls_config(cert_file, key_file)

        ctx = config.create_ssl_context()

        assert ctx is not None
        assert isinstance(ctx, ssl.SSLContext)

    def test_does_not_override_ciphers(self, tmp_path: Path) -> None:
        """Ciphers are not hardcoded — system policy decides."""
        cert_file, key_file = _generate_self_signed_cert(tmp_path)
        config = _make_tls_config(cert_file, key_file)

        ctx = config.create_ssl_context()
        system_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        assert ctx.get_ciphers() == system_ctx.get_ciphers()

    def test_does_not_override_minimum_version(self, tmp_path: Path) -> None:
        """Minimum TLS version is not hardcoded — system policy decides."""
        cert_file, key_file = _generate_self_signed_cert(tmp_path)
        config = _make_tls_config(cert_file, key_file)

        ctx = config.create_ssl_context()
        system_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

        assert ctx.minimum_version == system_ctx.minimum_version

    def test_compression_disabled(self, tmp_path: Path) -> None:
        """TLS compression is disabled (RFC 7540 Section 9.2.1)."""
        cert_file, key_file = _generate_self_signed_cert(tmp_path)
        config = _make_tls_config(cert_file, key_file)

        ctx = config.create_ssl_context()

        assert ctx.options & ssl.OP_NO_COMPRESSION


@pytest.mark.parametrize(
    "settings",
    [
        TLSSettings(policy="typo"),
        TLSSettings(minimum_version="TLSv1.3"),
        TLSSettings(policy="configured"),
        TLSSettings(policy="configured", minimum_version="TLSv1.0"),
        TLSSettings(policy="configured", minimum_version="TLSv1.2"),
        TLSSettings(
            policy="configured",
            minimum_version="TLSv1.3",
            ciphers="TLS_AES_128_GCM_SHA256",
        ),
        TLSSettings(
            policy="configured", minimum_version="TLSv1.3", curves="X25519:prime256v1"
        ),
    ],
)
def test_invalid_settings(settings: TLSSettings) -> None:
    """Incomplete or unrepresentable policy must fail, even without certs."""
    config = PolicyAwareConfig()
    config.tls_settings = settings
    with pytest.raises(ValueError, match=r"TLS|Python|Explicit"):
        config.create_ssl_context()


@pytest.mark.parametrize(
    "ciphers",
    [
        "ALL",
        "ECDHE-RSA-AES128-GCM-SHA256:UNKNOWN",
        "ECDHE-RSA-AES128-GCM-SHA256:",
        "ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256",
    ],
)
def test_cipher_list_is_exact(ciphers: str, tmp_path: Path) -> None:
    """OpenSSL's permissive filtering/selectors must not weaken requested policy."""
    config = _make_tls_config(*_generate_self_signed_cert(tmp_path))
    config.tls_settings = TLSSettings(
        policy="configured", minimum_version="TLSv1.2", ciphers=ciphers
    )
    with pytest.raises(ValueError, match="exact supported"):
        config.create_ssl_context()


def test_unknown_curve_fails(tmp_path: Path) -> None:
    """Unsupported group names cannot silently revert to default groups."""
    config = _make_tls_config(*_generate_self_signed_cert(tmp_path))
    config.tls_settings = TLSSettings(
        policy="configured", minimum_version="TLSv1.3", curves="unknown-group"
    )
    with pytest.raises(ValueError, match="curve"):
        config.create_ssl_context()


def test_required_certificate_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing required certs fail; explicit local-development HTTP stays available."""
    cert = tmp_path / "tls.crt"
    key = tmp_path / "tls.key"
    monkeypatch.setenv("TLS_CERT_FILE", str(cert))
    monkeypatch.setenv("TLS_KEY_FILE", str(key))
    assert serving_certificates(TLSSettings()) is None
    for settings in [
        TLSSettings(required=True),
        TLSSettings(policy="configured", minimum_version="TLSv1.3"),
    ]:
        with pytest.raises(ValueError, match="required"):
            serving_certificates(settings)
    cert.write_text("not a certificate")
    with pytest.raises(ValueError, match="required"):
        serving_certificates(TLSSettings())
    key.write_text("not a key")
    config = _make_tls_config(str(cert), str(key))
    with pytest.raises(ssl.SSLError):
        config.create_ssl_context()


def test_environment_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse a platform-independent configuration and reject boolean typos."""
    for name in [
        "TLS_POLICY",
        "TLS_REQUIRED",
        "TLS_MIN_VERSION",
        "TLS_CIPHERS",
        "TLS_CURVES",
    ]:
        monkeypatch.delenv(name, raising=False)
    assert TLSSettings.from_env() == TLSSettings()
    monkeypatch.setenv("TLS_POLICY", "configured")
    monkeypatch.setenv("TLS_REQUIRED", "true")
    monkeypatch.setenv("TLS_MIN_VERSION", "TLSv1.2")
    monkeypatch.setenv("TLS_CIPHERS", "ECDHE-RSA-AES128-GCM-SHA256")
    monkeypatch.setenv("TLS_CURVES", "prime256v1")
    assert TLSSettings.from_env() == TLSSettings(
        policy="configured",
        required=True,
        minimum_version="TLSv1.2",
        ciphers="ECDHE-RSA-AES128-GCM-SHA256",
        curves="prime256v1",
    )
    monkeypatch.setenv("TLS_REQUIRED", "ture")
    with pytest.raises(ValueError, match="true or false"):
        TLSSettings.from_env()


def _handshake(server: ssl.SSLContext, client: ssl.SSLContext) -> ssl.SSLObject:
    """Negotiate real OpenSSL TLS over memory BIOs, without network races."""
    server_in, server_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    client_in, client_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    server_tls = server.wrap_bio(server_in, server_out, server_side=True)
    client_tls = client.wrap_bio(client_in, client_out, server_hostname="localhost")
    done = set()
    for _ in range(100):
        for connection in [client_tls, server_tls]:
            try:
                connection.do_handshake()
                done.add(connection)
            except ssl.SSLWantReadError:
                pass
        server_in.write(client_out.read())
        client_in.write(server_out.read())
        if len(done) == 2:
            return server_tls
    pytest.fail("TLS handshake did not complete")


def test_negotiated_version_cipher_curve_and_alpn(tmp_path: Path) -> None:
    """Allow the configured TLS 1.2 cipher/group; reject peers outside the policy."""
    cert, key = _generate_self_signed_cert(tmp_path)
    config = _make_tls_config(cert, key)
    config.tls_settings = TLSSettings(
        policy="configured",
        required=True,
        minimum_version="TLSv1.2",
        ciphers="ECDHE-RSA-AES128-GCM-SHA256",
        curves="prime256v1",
    )
    server = config.create_ssl_context()
    assert server is not None
    client = ssl.create_default_context(cafile=cert)
    client.maximum_version = ssl.TLSVersion.TLSv1_2
    client.set_ciphers("ECDHE-RSA-AES128-GCM-SHA256")
    client.set_ecdh_curve("prime256v1")
    client.set_alpn_protocols(["h2", "http/1.1"])
    negotiated = _handshake(server, client)
    assert negotiated.version() == "TLSv1.2"
    assert negotiated.cipher()[0] == "ECDHE-RSA-AES128-GCM-SHA256"
    assert negotiated.selected_alpn_protocol() == "h2"
    client.set_ciphers("ECDHE-RSA-AES256-GCM-SHA384")
    with pytest.raises(ssl.SSLError):
        _handshake(server, client)
    client.set_ciphers("ECDHE-RSA-AES128-GCM-SHA256")
    client.set_ecdh_curve("X25519")
    with pytest.raises(ssl.SSLError):
        _handshake(server, client)
    config.tls_settings = TLSSettings(
        policy="configured", required=True, minimum_version="TLSv1.3"
    )
    modern = config.create_ssl_context()
    assert modern is not None
    with pytest.raises(ssl.SSLError):
        _handshake(modern, client)
    client.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
    assert _handshake(modern, client).version() == "TLSv1.3"
    config.tls_settings = TLSSettings(
        policy="configured", minimum_version="TLSv1.3", curves="prime256v1"
    )
    modern_curve = config.create_ssl_context()
    assert modern_curve is not None
    with pytest.raises(ssl.SSLError):
        _handshake(modern_curve, client)
    client.set_ecdh_curve("prime256v1")
    assert _handshake(modern_curve, client).version() == "TLSv1.3"
