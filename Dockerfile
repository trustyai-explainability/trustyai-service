# Dockerfile

FROM registry.access.redhat.com/ubi8/python-311:latest

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

USER root
RUN pip install poetry==1.6.1 && \
    poetry export -f requirements.txt --without dev > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt
COPY . .
USER 1001
EXPOSE 4443

#CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4443", "--ssl-keyfile", "/etc/tls/internal/tls.key", "--ssl-certfile", "/etc/tls/internal/tls.crt", "--log-level", "trace"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4443", "--ssl-keyfile", "/etc/tls/internal/tls.key", "--ssl-certfile", "/etc/tls/internal/tls.crt"]