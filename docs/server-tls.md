# Server TLS configuration

The service supports two TLS policies with generic environment variables. It does
not read cluster/platform APIs. A deploying controller is responsible for resolving
its platform's policy, selecting a representable configuration and restarting the
service when that policy changes. Startup-only settings do not provide live reload.

## Environment interface

| Variable | Default | Meaning |
|---|---|---|
| `TLS_POLICY` | `system` | `system` preserves OS/OpenSSL crypto policy; `configured` applies explicit controls. Unknown values fail startup. |
| `TLS_REQUIRED` | `false` | Only `true`/`false` (case insensitive). When true, missing serving files are fatal. `configured` always requires TLS, regardless of this flag. |
| `TLS_MIN_VERSION` | unset | Required in configured mode: `TLSv1.2` or `TLSv1.3`. Older/unknown versions fail rather than being approximated. |
| `TLS_CIPHERS` | unset | In configured TLS 1.2 mode, required colon-separated **exact OpenSSL cipher names**, e.g. `ECDHE-RSA-AES128-GCM-SHA256`. No IANA names, selectors (`DEFAULT`, `ALL`), unknown/duplicate entries or partial filtering. |
| `TLS_CURVES` | unset | Optional **single** OpenSSL curve name, e.g. `prime256v1` or `X25519`. Multiple colon-separated groups or unsupported names fail. Unset preserves the runtime defaults. |
| `TLS_CERT_FILE` | `/etc/tls/internal/tls.crt` | PEM serving certificate/chain. |
| `TLS_KEY_FILE` | `/etc/tls/internal/tls.key` | Matching PEM private key. |

Example of a representable configured policy:

```sh
TLS_POLICY=configured
TLS_REQUIRED=true
TLS_MIN_VERSION=TLSv1.2
TLS_CIPHERS=ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384
TLS_CURVES=prime256v1
TLS_CERT_FILE=/etc/tls/internal/tls.crt
TLS_KEY_FILE=/etc/tls/internal/tls.key
```

This example is **not** an assertion that it matches a named cluster profile.
Cipher controls apply to TLS <=1.2; TLS 1.3 may still negotiate when the minimum is
1.2, using OpenSSL's TLS 1.3 defaults. For a TLS 1.3 minimum, omit `TLS_CIPHERS`:
Python cannot configure TLS 1.3 suites. Supplying them fails rather than pretending
to enforce them. There is no maximum-version override or security-level lowering.
OS/FIPS restrictions may make the effective policy stricter than supplied settings.

## Legacy/system behaviour and required serving material

`TLS_POLICY=system` leaves minimum version, cipher suites and groups to the system
crypto policy, retaining the historical implementation. **System crypto policy is
not automatically a cluster TLS profile.** Explicit version/cipher/curve inputs in
system mode are rejected to avoid silently ignoring requested restrictions. An
operator mapping Legacy/no-opinion to system mode must omit those explicit inputs.

When neither file exists, optional system mode permits local development over HTTP
on `127.0.0.1:$HTTP_PORT` (default 8081). A partial certificate/key pair fails, even
in optional mode. Missing files in required/configured mode, invalid certificates,
mismatched keys, invalid settings or unsupported controls fail startup before either
application or health server is started. `TLS_REQUIRED=true` can also be used with
system mode to preserve Legacy crypto selection without permitting HTTP fallback.

The dual-server deployment still has a deliberate loopback HTTP backend for a
terminating proxy, and a separate health/consumer HTTP listener. Requiring TLS here
requires the direct HTTPS listener; it does not convert these other endpoints to
TLS. Network exposure and plaintext exceptions must be evaluated by the deployer.

ALPN remains a component choice, independent of the supplied TLS profile. The
Hypercorn configuration continues to advertise `h2` and `http/1.1`. This change does
not introduce an HTTP/2 toggle or disable HTTP/2.

## Capability boundary: when to use a TLS terminator

Python's standard SSL API cannot enforce an arbitrary multi-group curve preference
list, including profiles containing hybrid groups that the runtime does not support.
It also cannot configure TLS 1.3 ciphers. The implementation must **not** truncate a
profile's group list to one curve, substitute defaults for unsupported groups, or
claim to configure TLS 1.3 suites. If exact controls are required but unrepresentable,
deploy a capable TLS terminator (for example an appropriately built kube-rbac-proxy
or Envoy) and remove the direct Python HTTPS surface from the deployed topology,
rewiring all consumers and HTTPS monitors through the terminator. Merely adding a
proxy while retaining a non-adherent direct TLS listener does not solve the gap.

For RHOAI this is the preferred fallback for profiles requiring multiple groups.
The operator's follow-up deployment PR must choose the terminator topology where
needed, protect the private HTTP backend, propagate policy changes via rollouts and
configure trusted HTTPS scraping. This application PR does not add OpenShift APIs,
wire operator inputs, establish monitor trust or claim complete Strict adherence.

## Verification

```sh
pytest tests/service/test_tls.py tests/test_app_integration.py
```

Tests preserve default system crypto selection, reject incomplete/unrepresentable
settings and verify real OpenSSL handshakes for TLS 1.2/1.3, allowed versus excluded
ciphers, matching/mismatching single groups in both versions, and ALPN. Application
startup tests assert no `serve` call on missing required or invalid serving material.
Runtime-specific crypto/FIPS policies and the final operator/proxy topology still
need target-image and cluster acceptance tests.
