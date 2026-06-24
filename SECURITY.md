# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### Red Hat Product Security

For vulnerabilities affecting Red Hat OpenShift AI (RHOAI) deployments:

- **Email:** secalert@redhat.com
- **Web:** https://access.redhat.com/security/team/contact/

### Upstream Project

For vulnerabilities in the upstream open source project:

- **GitHub:** Use [GitHub Security Advisories](https://github.com/trustyai-explainability/trustyai-service/security/advisories/new) to report privately

### Response Timeline

- **Acknowledgement:** Within 3 business days
- **Initial assessment:** Within 7 business days
- **Fix timeline:** Depends on severity; critical issues are prioritized

## Supported Versions

| Version | Supported |
|---------|-----------|
| main branch | Yes |
| Released tags | Yes |

## Security Practices

This project employs:

- **Static analysis:** Bandit (Python security linter) and Ruff
- **Dependency scanning:** Dependabot, Trivy container scanning
- **Type checking:** Pyrefly
- **Secret detection:** detect-secrets, gitleaks
- **Container hardening:** UBI minimal base image, non-root user, FIPS-compatible crypto policy
