# Dockerfile

FROM registry.access.redhat.com/ubi8/python-311:latest

ARG EXTRAS=""


WORKDIR /app

COPY pyproject.toml poetry.lock* README.md ./


USER root

# install mariadb connector from MariaDB Community Service package repository
RUN if [[ "$EXTRAS" == *"mariadb"* ]]; then  \
		curl -LsSO https://r.mariadb.com/downloads/mariadb_repo_setup && \
    	chmod +x mariadb_repo_setup && \
    	./mariadb_repo_setup --mariadb-server-version="mariadb-11.4" && \
    	dnf install -y MariaDB-shared MariaDB-devel;  \
    fi
RUN pip install uv==0.6.16 && \
    uv pip install ".[$EXTRAS]"
COPY . .
USER 1001
EXPOSE 8080 4443


CMD ["python", "-m", "src.main"]
