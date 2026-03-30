FROM registry.access.redhat.com/ubi9/python-312:latest

ARG EXTRAS=""
ARG VERSION="1.0.0rc0"
ARG BUILD_DATE
ARG VCS_REF
ARG ENABLE_FIPS_POLICY="true"

# Use UBI standard directory instead of /app
WORKDIR /opt/app-root

USER root

# ===========================================================================================
# FIPS COMPLIANCE CONFIGURATION
# ===========================================================================================
# Set system crypto policy to FIPS for maximum compatibility with FIPS environments.
#
# IMPORTANT: This enables FIPS crypto policy but NOT full FIPS mode (requires kernel fips=1).
# - On NON-FIPS hosts: Uses FIPS-approved algorithms where possible
# - On FIPS-enabled hosts: Container automatically inherits full FIPS mode from kernel
#
# Application code is FIPS-compatible (verified in Phase 3 security review).
# For details, see: docs/FIPS_compliance.md
# ===========================================================================================
RUN if [ "$ENABLE_FIPS_POLICY" = "true" ]; then \
        echo "Configuring FIPS crypto policy..." && \
        update-crypto-policies --set FIPS && \
        echo "FIPS crypto policy enabled (full FIPS mode requires FIPS-enabled host)"; \
    else \
        echo "FIPS crypto policy not enabled (set ENABLE_FIPS_POLICY=true to enable)"; \
    fi

# Verify and display FIPS configuration
RUN echo "========================================" && \
    echo "FIPS Configuration Status:" && \
    echo "========================================" && \
    echo "Crypto Policy: $(update-crypto-policies --show 2>/dev/null || echo 'Unknown')" && \
    echo "FIPS Modules: $(ls -1 /usr/lib64/ossl-modules/fips.so 2>/dev/null && echo 'Installed' || echo 'Not found')" && \
    echo "OpenSSL Version: $(openssl version 2>/dev/null || echo 'Unknown')" && \
    fips-mode-setup --check 2>&1 | grep -E "(enabled|disabled|Inconsistent)" || echo "FIPS status check unavailable" && \
    echo "========================================" && \
    echo "Note: Full FIPS mode activation requires:" && \
    echo "  1. FIPS-enabled host (kernel fips=1)" && \
    echo "  2. Container inherits FIPS mode from host" && \
    echo "  3. See docs/FIPS_compliance.md for details" && \
    echo "========================================"

# Install MariaDB connector if needed (requires version >= 3.3.1, UBI has 3.2.6)
# Pin to 11.4.x for stability (allows patch updates)
RUN if [[ "$EXTRAS" == *"mariadb"* ]]; then  \
		curl -LsSO https://r.mariadb.com/downloads/mariadb_repo_setup && \
		# Verify script integrity (basic sanity check - script should contain mariadb_repo_setup signature)
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
    	rm -rf /var/cache/dnf /var/cache/yum /root/.cache mariadb_repo_setup;  \
    fi

# Copy only dependency files first for better layer caching
COPY pyproject.toml README.md ./

# Install dependencies with pinned uv version and cleanup
RUN pip install --no-cache-dir --upgrade pip==26.0.1 && \
    pip install --no-cache-dir uv==0.11.1 && \
    uv pip install --no-cache ".[$EXTRAS]" && \
    rm -rf /root/.cache /tmp/*

# Copy source code last to maximize cache hits
COPY src src

# Python optimizations and security settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random

# Set FIPS env var to match build-time configuration
ENV FIPS_POLICY_ENABLED=${ENABLE_FIPS_POLICY}

# Base image /opt/app-root is already owned by 1001:0 with group write permissions
# No need to chown - just ensure our copied files inherit the correct ownership
USER 1001
EXPOSE 8080 4443

# OCI image metadata
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
      io.trustyai.fips.documentation="https://github.com/trustyai-explainability/trustyai-service/blob/main/docs/FIPS_compliance.md"

CMD ["python", "-m", "src.main"]
