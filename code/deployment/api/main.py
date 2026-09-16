"""Stage 3a - model API.

A FastAPI service that serves the packaged scikit-learn pipeline produced by
stage 2.  The artifact is a *complete* pipeline (feature engineering ->
preprocessing -> classifier), so the API only validates the raw request payload
and forwards it to the model - training and serving can never drift apart.

Run locally:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ``features`` lives next to this file inside the image (copied from
# ``code/models/features.py``) so the pickled pipeline can be unpickled.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import RAW_FEATURES  # noqa: E402
from schemas import (  # noqa: E402
    BatchPredictionResponse,
    BatchRequest,
    FeatureSchemaResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    WineFeatures,
)

API_VERSION = "1.0.0"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/model.joblib"))
METADATA_PATH = Path(os.environ.get("MODEL_METADATA_PATH", "/app/models/model_metadata.json"))
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.5"))

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("wine-api")

STATE: Dict[str, Any] = {"model": None, "metadata": {}, "started_at": time.time(), "load_error": None}


def load_model() -> None:
    """Load the packaged model and its metadata into the module state."""
    try:
        STATE["model"] = joblib.load(MODEL_PATH)
        LOGGER.info("Loaded model from %s", MODEL_PATH)
        STATE["load_error"] = None
    except Exception as exc:  # noqa: BLE001 - the API stays up and reports "degraded"
        STATE["model"] = None
        STATE["load_error"] = str(exc)
        LOGGER.error("Could not load the model from %s: %s", MODEL_PATH, exc)

    if METADATA_PATH.exists():
        try:
            STATE["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            trained_version = STATE["metadata"].get("sklearn_version")
            if trained_version and trained_version != sklearn.__version__:
                LOGGER.warning(
                    "scikit-learn version mismatch: model trained with %s, serving with %s",
                    trained_version,
                    sklearn.__version__,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not read model metadata: %s", exc)
            STATE["metadata"] = {}
    else:
        LOGGER.warning("Model metadata not found at %s", METADATA_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield
    LOGGER.info("Shutting down the wine-quality API.")


app = FastAPI(
    title="Wine Quality Prediction API",
    description=(
        "Predicts whether a wine is of **good quality** (expert score >= 6) from its "
        "physicochemical properties. Part of the PMLDL Assignment 1 MLOps pipeline."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model():
    model = STATE.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not available: {STATE.get('load_error') or 'not loaded'}",
        )
    return model


def _to_frame(items) -> pd.DataFrame:
    return pd.DataFrame([item.model_dump() for item in items])[RAW_FEATURES]


def _build_response(prediction: int, probability: float, elapsed_ms: float) -> PredictionResponse:
    metadata = STATE.get("metadata", {})
    return PredictionResponse(
        prediction=prediction,
        label="Good quality wine" if prediction == 1 else "Standard quality wine",
        probability_good=round(probability, 6),
        confidence=round(probability if prediction == 1 else 1.0 - probability, 6),
        threshold=DECISION_THRESHOLD,
        model_name=metadata.get("model_name"),
        model_version=metadata.get("model_version"),
        inference_ms=round(elapsed_ms, 3),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
    LOGGER.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    """Entry point with links to the interactive documentation."""
    return {
        "service": "Wine Quality Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/feature-schema", "/predict", "/predict/batch"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness/readiness probe - also used by the Docker health check."""
    metadata = STATE.get("metadata", {})
    loaded = STATE.get("model") is not None
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_name=metadata.get("model_name"),
        model_version=metadata.get("model_version"),
        trained_at=metadata.get("trained_at"),
        api_version=API_VERSION,
        uptime_seconds=round(time.time() - STATE["started_at"], 1),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    """Champion model description, test metrics and candidate leaderboard."""
    metadata = STATE.get("metadata", {})
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata is not available")
    return ModelInfoResponse(
        model_name=metadata.get("model_name"),
        model_version=metadata.get("model_version"),
        trained_at=metadata.get("trained_at"),
        git_revision=metadata.get("git_revision"),
        target=metadata.get("target"),
        positive_class_meaning=metadata.get("positive_class_meaning"),
        metrics={k: float(v) for k, v in (metadata.get("metrics") or {}).items()},
        leaderboard=metadata.get("leaderboard") or {},
        confusion_matrix=metadata.get("confusion_matrix") or [],
        train_rows=metadata.get("train_rows"),
        test_rows=metadata.get("test_rows"),
        sklearn_version=metadata.get("sklearn_version"),
        runtime_sklearn_version=sklearn.__version__,
    )


@app.get("/feature-schema", response_model=FeatureSchemaResponse, tags=["meta"])
def feature_schema() -> FeatureSchemaResponse:
    """Feature names and training-set ranges, used by the web app to build its inputs."""
    metadata = STATE.get("metadata", {})
    if not metadata.get("feature_ranges"):
        raise HTTPException(status_code=503, detail="Feature schema is not available")
    return FeatureSchemaResponse(
        numeric_features=metadata.get("raw_numeric_features", []),
        categorical_features=metadata.get("raw_categorical_features", []),
        wine_types=metadata.get("wine_types", ["red", "white"]),
        feature_ranges=metadata["feature_ranges"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: WineFeatures) -> PredictionResponse:
    """Predict the quality class of a single wine sample."""
    model = _require_model()
    started = time.perf_counter()
    probability = float(model.predict_proba(_to_frame([payload]))[0, 1])
    elapsed_ms = (time.perf_counter() - started) * 1000
    prediction = int(probability >= DECISION_THRESHOLD)
    LOGGER.info("prediction=%s probability=%.4f in %.2f ms", prediction, probability, elapsed_ms)
    return _build_response(prediction, probability, elapsed_ms)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(payload: BatchRequest) -> BatchPredictionResponse:
    """Predict the quality class for a batch of wine samples."""
    model = _require_model()
    started = time.perf_counter()
    probabilities = model.predict_proba(_to_frame(payload.items))[:, 1]
    elapsed_ms = (time.perf_counter() - started) * 1000
    predictions = [
        _build_response(int(p >= DECISION_THRESHOLD), float(p), elapsed_ms / len(probabilities))
        for p in probabilities
    ]
    LOGGER.info("Batch prediction of %d items in %.2f ms", len(predictions), elapsed_ms)
    return BatchPredictionResponse(
        predictions=predictions, count=len(predictions), inference_ms=round(elapsed_ms, 3)
    )


@app.post("/reload", tags=["meta"])
def reload_model() -> Dict[str, Any]:
    """Reload the model artifact from disk (useful when the model file is mounted)."""
    load_model()
    return {"model_loaded": STATE["model"] is not None, "error": STATE.get("load_error")}
