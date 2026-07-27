# Multi-stage build: UBI10 Python 3.14 minimal for builder and runtime.
# FIPS crypto policy support included.

ARG EXTRAS=""
ARG VERSION="0.0.0.dev0"
ARG BUILD_DATE
ARG VCS_REF
ARG ENABLE_FIPS_POLICY="true"

# =============================================================================
# Stage 1: Builder — UBI10 Python 3.14 minimal with dev tools
# =============================================================================
FROM registry.access.redhat.com/ubi10/python-314-minimal:latest AS builder

ARG EXTRAS

USER root

# Install development tools and MariaDB libraries for C extension compilation
# UBI10 ships MariaDB Connector/C 3.4.4, compatible with Python mariadb >= 1.1
# Note: UBI10 uses python3.14-devel (version-specific package name)
RUN if echo "$EXTRAS" | grep -q "mariadb"; then \
        microdnf install -y \
            gcc \
            python3.14-devel \
            make \
            mariadb-connector-c-devel && \
        microdnf clean all; \
    else \
        echo "MariaDB extra not requested, installing minimal dev tools for other C extensions" && \
        microdnf install -y gcc python3.14-devel make && \
        microdnf clean all && \
        touch /usr/lib64/libmariadb.so.stub; \
    fi

USER 1001

COPY pyproject.toml README.md ./

ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0"

RUN mkdir -p src/trustyai_service && \
    pip install --no-cache-dir --upgrade pip==26.1.1 uv==0.11.22 && \
    uv pip install --no-cache ".[$EXTRAS]" && \
    pip uninstall -y uv && \
    rm -f /opt/app-root/bin/uv /opt/app-root/bin/uvx && \
    rm -rf /root/.cache /tmp/*

# =============================================================================
# Stage 2: Runtime — UBI10 Python 3.14 minimal (no compilers, no dev headers)
# =============================================================================
FROM registry.access.redhat.com/ubi10/python-314-minimal:latest

ARG EXTRAS
ARG VERSION
ARG BUILD_DATE
ARG VCS_REF
ARG ENABLE_FIPS_POLICY

USER root

# ---------------------------------------------------------------------------
# FIPS COMPLIANCE CONFIGURATION
# ---------------------------------------------------------------------------
# Sets the system crypto policy to FIPS for maximum compatibility with FIPS
# environments.
#
# IMPORTANT: This enables FIPS crypto policy but NOT full FIPS mode
# (requires kernel fips=1).
#   - On NON-FIPS hosts: Uses FIPS-approved algorithms where possible
#   - On FIPS-enabled hosts: Container inherits full FIPS mode from kernel
#
# To enable full FIPS mode, deploy on FIPS-enabled infrastructure:
#   https://docs.openshift.com/container-platform/latest/installing/installing-fips.html
# ---------------------------------------------------------------------------
RUN if [ "$ENABLE_FIPS_POLICY" = "true" ]; then \
        microdnf install -y crypto-policies-scripts && \
        update-crypto-policies --set FIPS && \
        microdnf clean all && \
        rm -rf /var/cache/yum && \
        echo "FIPS crypto policy enabled (full FIPS mode requires FIPS-enabled host)" && \
        echo "Crypto Policy: $(update-crypto-policies --show)"; \
    else \
        echo "FIPS crypto policy not enabled (set ENABLE_FIPS_POLICY=true to enable)"; \
    fi

# Copy MariaDB shared libraries from builder if needed
# Note: We copy instead of installing to avoid needing yum in the minimal image
COPY --from=builder /usr/lib64/libmariadb.so* /usr/lib64/
RUN rm -f /usr/lib64/libmariadb.so.stub

# Upgrade system pip to eliminate base image CVEs
RUN pip install --no-cache-dir --upgrade pip==26.1.1 && rm -rf /root/.cache

WORKDIR /opt/app-root

# Copy installed packages from builder
COPY --from=builder /opt/app-root/lib/python3.14/site-packages /opt/app-root/lib/python3.14/site-packages
COPY --from=builder /opt/app-root/lib64/python3.14/site-packages /opt/app-root/lib64/python3.14/site-packages
COPY --from=builder /opt/app-root/bin /opt/app-root/bin

COPY src/trustyai_service trustyai_service
COPY pyproject.toml README.md ./

RUN printf '__version__ = version = "%s"\n' "${VERSION}" > trustyai_service/_version.py && \
    chown 1001:0 trustyai_service/_version.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random

ENV FIPS_POLICY_ENABLED=${ENABLE_FIPS_POLICY}

USER 1001
EXPOSE 8080 4443

LABEL org.opencontainers.image.title="TrustyAI Service" \
      org.opencontainers.image.description="Python implementation of TrustyAI Service for AI explainability and fairness" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/trustyai-explainability/trustyai-service" \
      org.opencontainers.image.vendor="TrustyAI" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.trustyai.fips.policy="${ENABLE_FIPS_POLICY}" \
      io.trustyai.fips.mode="host-dependent" \
      io.trustyai.fips.compatible="true" \
      org.opencontainers.image.base.name="registry.redhat.io/ubi10/python-314-minimal"

CMD ["python", "-m", "trustyai_service.main"]
