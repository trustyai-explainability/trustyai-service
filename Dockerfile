# Dockerfile

FROM registry.access.redhat.com/ubi8/python-311:latest

ARG EXTRAS=""


WORKDIR /app

COPY pyproject.toml poetry.lock* README.md ./


USER root

# install mariadb connector from MariaDB Community Service package repository
RUN if [[ "$EXTRAS" == *"mariadb"* ]]; then  \
		curl -LsSO https://r.mariadb.com/downloads/mariadb_repo_setup && \
    	echo "c4a0f3dade02c51a6a28ca3609a13d7a0f8910cccbb90935a2f218454d3a914a  mariadb_repo_setup" | sha256sum -c - && \
    	chmod +x mariadb_repo_setup && \
    	./mariadb_repo_setup --mariadb-server-version="mariadb-10.6" && \
    	dnf install -y MariaDB-shared MariaDB-devel;  \
    fi
RUN pip install uv==0.6.16 && \
    uv pip install ".[$EXTRAS]"
COPY . .
USER 1001
EXPOSE 4443


CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
#CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "4443", "--ssl-keyfile", "/etc/tls/internal/tls.key", "--ssl-certfile", "/etc/tls/internal/tls.crt"]