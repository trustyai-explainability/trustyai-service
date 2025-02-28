# Dockerfile

FROM registry.access.redhat.com/ubi8/python-311:latest

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN pip install poetry==1.6.1

RUN poetry export -f requirements.txt --without dev > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
