# Dockerfile

FROM registry.access.redhat.com/ubi8/python-311:latest

ARG EXTRAS=""


WORKDIR /app

COPY pyproject.toml poetry.lock* README.md ./



USER root
RUN pip install uv==0.6.16 && \
    uv pip install .[$EXTRAS]
COPY . .
USER 1001
EXPOSE 4443


CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
#CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4443", "--ssl-keyfile", "/etc/tls/internal/tls.key", "--ssl-certfile", "/etc/tls/internal/tls.crt"]