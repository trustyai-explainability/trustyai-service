ARG BASE_IMAGE=quay.io/opendatahub/odh-midstream-python-base-3-12:1.20260731.1

FROM ${BASE_IMAGE}

USER root

ARG VERSION="0.0.0.dev0"
ARG BUILD_DATE
ARG VCS_REF

# mariadb (Python) 1.1.x requires mariadb-connector-c >= 3.3.1;
# the ODH base image ships 3.2.6, so install the connector from the
# official MariaDB repo and remove the repo config afterwards.
RUN rpm --import https://supplychain.mariadb.com/MariaDB-Server-GPG-KEY && \
    printf '[mariadb]\nname=MariaDB Server\nbaseurl=https://dlm.mariadb.com/repo/mariadb-server/11.4/yum/rhel/9/$basearch\ngpgcheck=1\ngpgkey=https://supplychain.mariadb.com/MariaDB-Server-GPG-KEY\nenabled=1\n' > /etc/yum.repos.d/mariadb.repo && \
    dnf install -y MariaDB-devel && \
    rm -f /etc/yum.repos.d/mariadb.repo && \
    dnf clean all && \
    CONNECTOR_V="$(mariadb_config --cc_version)" && \
    { printf '3.4.9\n%s\n' "$CONNECTOR_V" | sort -V -C || \
      { echo "FATAL: Connector/C $CONNECTOR_V < 3.4.9 (CVE-2026-44172)"; exit 1; }; }

WORKDIR /opt/app-root

ENV SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}"

COPY pyproject.toml README.md ./
COPY requirements.txt requirements-build.txt ./

RUN pip install --no-cache-dir --no-deps --require-hashes \
        -r requirements-build.txt && \
    pip install --no-cache-dir --no-deps --no-build-isolation --require-hashes \
        -r requirements.txt

COPY src src

RUN pip install --no-cache-dir --no-deps --no-build-isolation . && \
    pip uninstall -y hatchling hatch-vcs setuptools setuptools-scm \
        vcs-versioning trove-classifiers pathspec pluggy

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random

USER 1001
EXPOSE 8081 4443 8080

CMD ["python", "-m", "trustyai_service.main"]

LABEL org.opencontainers.image.title="TrustyAI Service" \
      org.opencontainers.image.description="Python implementation of TrustyAI Service for AI explainability and fairness" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/trustyai-explainability/trustyai-service" \
      org.opencontainers.image.vendor="TrustyAI" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="quay.io/opendatahub/odh-midstream-python-base-3-12"
