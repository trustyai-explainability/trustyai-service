# UBI9 Python 3.12 Minimal — reduced Trivy surface with FIPS support.
# Uses the full UBI9 Python image as builder, minimal as runtime.

ARG EXTRAS=""
ARG VERSION="1.0.0rc0"
ARG BUILD_DATE
ARG VCS_REF
ARG ENABLE_FIPS_POLICY="true"

# =============================================================================
# Stage 1: Builder — full UBI9 image with compilers and dev headers
# =============================================================================
FROM registry.access.redhat.com/ubi9/python-312:latest AS builder

ARG EXTRAS

# Install MariaDB dev libraries if needed
RUN if [[ "$EXTRAS" == *"mariadb"* ]]; then \
        curl --fail -LsSO https://r.mariadb.com/downloads/mariadb_repo_setup && \
        grep -q "mariadb_repo_setup" mariadb_repo_setup || { echo "ERROR: Downloaded script appears invalid"; exit 1; } && \
        [ -s mariadb_repo_setup ] || { echo "ERROR: Downloaded script is empty"; exit 1; } && \
        chmod +x mariadb_repo_setup && \
        ./mariadb_repo_setup --mariadb-server-version="mariadb-11.4" --skip-check-installed && \
        dnf install -y --disablerepo=rhel-9-for-aarch64-appstream-rpms \
                       --disablerepo=rhel-9-for-aarch64-baseos-rpms \
                       --disablerepo=rhel-9-for-x86_64-appstream-rpms \
                       --disablerepo=rhel-9-for-x86_64-baseos-rpms \
                       MariaDB-shared-11.4* MariaDB-devel-11.4* && \
        dnf clean all && \
        rm -rf /var/cache/dnf /var/cache/yum /root/.cache mariadb_repo_setup; \
    fi

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --upgrade pip==26.1 uv==0.11.1 && \
    uv pip install --no-cache ".[$EXTRAS]" && \
    pip uninstall -y uv && \
    rm -rf /root/.cache /tmp/*

# =============================================================================
# Stage 2: Runtime — minimal UBI9 image (no compilers, no dev headers)
# =============================================================================
FROM registry.access.redhat.com/ubi9/python-312-minimal:latest

ARG EXTRAS
ARG VERSION
ARG BUILD_DATE
ARG VCS_REF
ARG ENABLE_FIPS_POLICY

USER root

# FIPS crypto policy
RUN if [ "$ENABLE_FIPS_POLICY" = "true" ]; then \
        microdnf --disablerepo='rhel-*' install -y crypto-policies-scripts && \
        update-crypto-policies --set FIPS && \
        microdnf clean all && \
        rm -rf /var/cache/yum && \
        echo "FIPS crypto policy enabled (full FIPS mode requires FIPS-enabled host)"; \
    fi

# Install MariaDB shared libraries if needed
RUN if echo "$EXTRAS" | grep -q "mariadb"; then \
        curl --fail -LsSO https://r.mariadb.com/downloads/mariadb_repo_setup && \
        chmod +x mariadb_repo_setup && \
        ./mariadb_repo_setup --mariadb-server-version="mariadb-11.4" --skip-check-installed && \
        microdnf install -y MariaDB-shared-11.4* && \
        microdnf clean all && \
        rm -rf /var/cache/yum mariadb_repo_setup; \
    fi

# Upgrade system pip to eliminate base image CVEs
RUN pip3 install --no-cache-dir --upgrade pip==26.1 && rm -rf /root/.cache

WORKDIR /opt/app-root

# Copy installed packages from builder
COPY --from=builder /opt/app-root/lib/python3.12/site-packages /opt/app-root/lib/python3.12/site-packages
COPY --from=builder /opt/app-root/lib64/python3.12/site-packages /opt/app-root/lib64/python3.12/site-packages
COPY --from=builder /opt/app-root/bin /opt/app-root/bin

COPY src src
COPY pyproject.toml README.md ./

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
      io.trustyai.fips.documentation="https://github.com/trustyai-explainability/trustyai-service/blob/main/docs/FIPS_compliance.md" \
      org.opencontainers.image.base.name="registry.access.redhat.com/ubi9/python-312-minimal"

CMD ["python", "-m", "src.main"]
