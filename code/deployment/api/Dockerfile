# ---------------------------------------------------------------------------
# Model API image (FastAPI + uvicorn).
# Build context: the repository root  ->  docker build -f code/deployment/api/Dockerfile .
# The trained model is baked into the image, which makes the container fully
# self-contained and reproducible for a given pipeline run.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_METADATA_PATH=/app/models/model_metadata.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Dependencies (cached across pipeline runs).
COPY code/deployment/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2. Application code + the feature-engineering module that the pickled
#    pipeline refers to (it must be importable as `features`).
COPY code/deployment/api/main.py code/deployment/api/schemas.py /app/
COPY code/models/features.py /app/features.py

# 3. The model artifact produced by stage 2 (last layer: changes every run).
COPY models/model.joblib models/model_metadata.json models/metrics.json /app/models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
