# Minimal MLflow tracking server (works on both amd64 and arm64).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow==2.17.2

RUN mkdir -p /mlflow/artifacts
WORKDIR /mlflow

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=10 \
    CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", \
     "--host", "0.0.0.0", "--port", "5000"]
