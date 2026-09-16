"""Stage 3 tests: the API contract (schema validation + inference endpoints).

The tests build a tiny model on the fly, point the API at it and exercise the
endpoints through FastAPI's TestClient - no Docker required.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "deployment" / "api"))

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from features import RAW_FEATURES, RAW_NUMERIC_FEATURES  # noqa: E402
from train import build_pipeline  # noqa: E402

VALID_PAYLOAD = {
    "fixed_acidity": 7.0,
    "volatile_acidity": 0.27,
    "citric_acid": 0.36,
    "residual_sugar": 6.1,
    "chlorides": 0.045,
    "free_sulfur_dioxide": 45.0,
    "total_sulfur_dioxide": 170.0,
    "density": 0.9938,
    "ph": 3.0,
    "sulphates": 0.45,
    "alcohol": 11.8,
    "wine_type": "white",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A TestClient wired to a throw-away model artifact."""
    model_dir = tmp_path_factory.mktemp("models")
    rng = np.random.default_rng(3)
    size = 150
    frame = pd.DataFrame({name: rng.normal(5, 1, size).clip(0.1) for name in RAW_NUMERIC_FEATURES})
    frame["density"] = rng.normal(0.995, 0.002, size)
    frame["wine_type"] = rng.choice(["red", "white"], size)
    frame = frame[RAW_FEATURES]
    target = (frame["alcohol"] > frame["alcohol"].median()).astype(int)

    pipeline = build_pipeline("logistic_regression", random_state=0).fit(frame, target)
    joblib.dump(pipeline, model_dir / "model.joblib")
    (model_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "logistic_regression",
                "model_version": "test",
                "trained_at": "2024-01-01T00:00:00+00:00",
                "metrics": {"f1": 0.9, "roc_auc": 0.95},
                "raw_numeric_features": RAW_NUMERIC_FEATURES,
                "raw_categorical_features": ["wine_type"],
                "wine_types": ["red", "white"],
                "feature_ranges": {
                    name: {"min": 0.1, "max": 10.0, "median": 5.0, "mean": 5.0, "p01": 1.0, "p99": 9.0}
                    for name in RAW_NUMERIC_FEATURES
                },
            }
        ),
        encoding="utf-8",
    )

    import os

    os.environ["MODEL_PATH"] = str(model_dir / "model.joblib")
    os.environ["MODEL_METADATA_PATH"] = str(model_dir / "model_metadata.json")

    main = importlib.import_module("main")
    importlib.reload(main)
    with fastapi_testclient.TestClient(main.app) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_model_info_exposes_metrics(client) -> None:
    body = client.get("/model-info").json()
    assert body["model_name"] == "logistic_regression"
    assert body["metrics"]["f1"] == pytest.approx(0.9)


def test_feature_schema_lists_every_feature(client) -> None:
    body = client.get("/feature-schema").json()
    assert set(body["numeric_features"]) == set(RAW_NUMERIC_FEATURES)
    assert body["wine_types"] == ["red", "white"]


def test_predict_returns_a_valid_prediction(client) -> None:
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_good"] <= 1.0
    assert body["label"]


def test_batch_prediction_matches_the_number_of_items(client) -> None:
    response = client.post("/predict/batch", json={"items": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


@pytest.mark.parametrize(
    "field,value",
    [("ph", 99.0), ("alcohol", -1.0), ("wine_type", "rose"), ("density", 5.0)],
)
def test_invalid_payloads_are_rejected(client, field: str, value) -> None:
    payload = {**VALID_PAYLOAD, field: value}
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_field_is_rejected(client) -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "alcohol"}
    assert client.post("/predict", json=payload).status_code == 422
