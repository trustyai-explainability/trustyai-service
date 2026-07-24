"""Tests for TLS crypto policy compliance."""

import ssl
import subprocess
from pathlib import Path

from trustyai_service.service.tls import PolicyAwareConfig


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
