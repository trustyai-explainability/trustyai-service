"""TLS configuration that honors the system crypto policy.

Hypercorn hardcodes its cipher list and minimum TLS version, overriding
the system crypto policy set by ``update-crypto-policies``.  This module
provides a Config subclass that delegates cipher and version selection to
the operating system, so the service automatically honors the cluster TLS
profile (Old, Intermediate, Modern, Custom) and FIPS crypto policy.
"""

from ssl import OP_NO_COMPRESSION, Purpose, SSLContext, create_default_context

from hypercorn.config import Config


class PolicyAwareConfig(Config):
    """Hypercorn Config that respects the system crypto policy for TLS."""

    def create_ssl_context(self) -> SSLContext | None:
        """Build an SSL context without overriding ciphers or min version."""
        if not self.ssl_enabled:
            return None

        context = create_default_context(Purpose.CLIENT_AUTH)
        context.options |= OP_NO_COMPRESSION
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
