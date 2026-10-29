"""System-policy TLS or explicit, validated deployment-provided TLS settings.

System crypto policy is not the same as a cluster TLS profile. This module
contains no platform API dependencies; a deploying controller supplies settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from ssl import (
    OP_NO_COMPRESSION,
    Purpose,
    SSLContext,
    TLSVersion,
    create_default_context,
)

from hypercorn.config import Config


@dataclass(frozen=True)
class TLSSettings:
    """Generic server TLS controls, independent of any platform adherence API."""

    policy: str = "system"
    required: bool = False
    minimum_version: str = ""
    ciphers: str = ""
    curves: str = ""

    @classmethod
    def from_env(cls) -> "TLSSettings":
        """Parse controls strictly so typos cannot silently disable enforcement."""
        required = os.getenv("TLS_REQUIRED", "false").lower()
        if required not in {"true", "false"}:
            msg = "TLS_REQUIRED must be true or false"
            raise ValueError(msg)
        settings = cls(
            policy=os.getenv("TLS_POLICY", "system"),
            required=required == "true",
            minimum_version=os.getenv("TLS_MIN_VERSION", ""),
            ciphers=os.getenv("TLS_CIPHERS", ""),
            curves=os.getenv("TLS_CURVES", ""),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Reject incomplete settings and unsupported policy modes."""
        if self.policy not in {"system", "configured"}:
            msg = "TLS_POLICY must be system or configured"
            raise ValueError(msg)
        if self.policy == "system":
            if self.minimum_version or self.ciphers or self.curves:
                msg = "Explicit TLS settings require TLS_POLICY=configured"
                raise ValueError(msg)
            return
        if self.minimum_version not in {"TLSv1.2", "TLSv1.3"}:
            msg = "Configured TLS_MIN_VERSION must be TLSv1.2 or TLSv1.3"
            raise ValueError(msg)
        if self.minimum_version == "TLSv1.2" and not self.ciphers:
            msg = "TLS_CIPHERS is required for configured TLSv1.2"
            raise ValueError(msg)
        if self.minimum_version == "TLSv1.3" and self.ciphers:
            msg = "Python cannot configure TLS 1.3 cipher suites; omit TLS_CIPHERS"
            raise ValueError(msg)
        if self.curves and len(self.curves.split(":")) != 1:
            msg = "Python supports only a single TLS_CURVES entry; use a capable TLS terminator for group lists"
            raise ValueError(msg)

    def apply(self, context: SSLContext) -> None:
        """Apply representable controls without silently filtering requested values."""
        self.validate()
        if self.policy == "system":
            return
        context.minimum_version = (
            TLSVersion.TLSv1_3
            if self.minimum_version == "TLSv1.3"
            else TLSVersion.TLSv1_2
        )
        if self.ciphers:
            requested = self.ciphers.split(":")
            context.set_ciphers(self.ciphers)
            actual = {
                cipher["name"]
                for cipher in context.get_ciphers()
                if cipher["protocol"] != "TLSv1.3"
            }
            if actual != set(requested) or len(requested) != len(set(requested)):
                msg = "TLS_CIPHERS must contain exact supported OpenSSL cipher names (no selectors, duplicates or unknown suites)"
                raise ValueError(msg)
        if self.curves:
            context.set_ecdh_curve(self.curves)


def serving_certificates(settings: TLSSettings) -> dict[str, str] | None:
    """Resolve serving files; never downgrade required/configured TLS to HTTP."""
    settings.validate()
    cert = Path(os.getenv("TLS_CERT_FILE", "/etc/tls/internal/tls.crt"))
    key = Path(os.getenv("TLS_KEY_FILE", "/etc/tls/internal/tls.key"))
    if cert.is_file() and key.is_file():
        return {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    if (
        settings.required
        or settings.policy == "configured"
        or cert.is_file()
        or key.is_file()
    ):
        msg = f"TLS serving certificate and key are required: {cert}, {key}"
        raise ValueError(msg)
    return None


class PolicyAwareConfig(Config):
    """Hypercorn Config with system or explicit deployment-provided TLS policy."""

    tls_settings: TLSSettings = TLSSettings()

    def create_ssl_context(self) -> SSLContext | None:
        """Build and validate a context before exposing a TLS listener."""
        self.tls_settings.validate()
        if not self.ssl_enabled:
            if self.tls_settings.required or self.tls_settings.policy == "configured":
                msg = "TLS is required but serving certificate/key are not configured"
                raise ValueError(msg)
            return None

        context = create_default_context(Purpose.CLIENT_AUTH)
        context.options |= OP_NO_COMPRESSION
        self.tls_settings.apply(context)
        context.set_alpn_protocols(self.alpn_protocols)

        if self.certfile is not None and self.keyfile is not None:
            context.load_cert_chain(
                certfile=self.certfile,
                keyfile=self.keyfile,
                password=self.keyfile_password,
            )

        if self.ca_certs is not None:
            context.load_verify_locations(self.ca_certs)
        if self.verify_mode is not None:
            context.verify_mode = self.verify_mode
        if self.verify_flags is not None:
            context.verify_flags = self.verify_flags

        return context
