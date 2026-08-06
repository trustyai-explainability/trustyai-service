ARG BASE_IMAGE=quay.io/opendatahub/odh-midstream-python-base-3-12:1.20260731.1

FROM ${BASE_IMAGE} AS builder

USER root

RUN printf '[mariadb]\nname=MariaDB Server\nbaseurl=https://dlm.mariadb.com/repo/mariadb-server/11.4/yum/rhel/9/$basearch\ngpgcheck=0\nenabled=1\n' > /etc/yum.repos.d/mariadb.repo && \
    dnf install -y MariaDB-devel && \
    rm -f /etc/yum.repos.d/mariadb.repo && \
    dnf clean all

COPY pyproject.toml README.md ./
COPY requirements.txt requirements-build.txt ./

ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0"

RUN mkdir -p src/trustyai_service && \
    pip install --no-cache-dir -r requirements-build.txt && \
    pip install --no-cache-dir -r requirements.txt

FROM ${BASE_IMAGE}

USER root

COPY --from=builder /opt/app-root/lib/python3.12/site-packages /opt/app-root/lib/python3.12/site-packages
COPY --from=builder /opt/app-root/lib64/python3.12/site-packages /opt/app-root/lib64/python3.12/site-packages
COPY --from=builder /usr/lib64/libmariadb.so* /usr/lib64/

WORKDIR /opt/app-root

COPY src/trustyai_service trustyai_service
COPY pyproject.toml README.md ./

ARG VERSION="0.0.0.dev0"
RUN printf '__version__ = version = "%s"\n' "${VERSION}" > trustyai_service/_version.py && \
    chown 1001:0 trustyai_service/_version.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random

USER 1001
EXPOSE 8080 4443

CMD ["python", "-m", "trustyai_service.main"]

LABEL org.opencontainers.image.title="TrustyAI Service" \
      org.opencontainers.image.description="Python implementation of TrustyAI Service for AI explainability and fairness" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/trustyai-explainability/trustyai-service" \
      org.opencontainers.image.vendor="TrustyAI" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.base.name="quay.io/opendatahub/odh-midstream-python-base-3-12"
