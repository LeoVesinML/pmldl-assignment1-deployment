"""Stage 2b - model validation gate.

Runs between model engineering and deployment: a model is only promoted to the
API if (a) the packaged artifact can actually be loaded and used for inference
and (b) its test metrics clear the thresholds declared in ``params.yaml``.

Exits with a non-zero status when the gate fails, which stops the deployment
tasks of the Airflow DAG and keeps the previously deployed model serving.

Usage
-----
    python code/models/validate_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_logger, load_params, resolve  # noqa: E402
from features import RAW_FEATURES  # noqa: E402

LOGGER = get_logger("stage2.validate")


def main() -> int:
    params = load_params()
    model_cfg, thresholds = params["model"], params["validation"]
    model_dir = resolve(model_cfg["model_dir"])

    model_path = model_dir / model_cfg["model_file"]
    metrics_path = model_dir / model_cfg["metrics_file"]
    metadata_path = model_dir / model_cfg["metadata_file"]

    for path in (model_path, metrics_path, metadata_path):
        if not path.exists():
            LOGGER.error("Required artifact is missing: %s", path)
            return 1

    # --- 1. the artifact must load and predict -----------------------------
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sample = {feature: metadata["feature_ranges"][feature]["median"] for feature in metadata["raw_numeric_features"]}
    sample["wine_type"] = metadata["wine_types"][0]
    frame = pd.DataFrame([sample])[RAW_FEATURES]

    proba = float(model.predict_proba(frame)[0, 1])
    prediction = int(model.predict(frame)[0])
    if not 0.0 <= proba <= 1.0 or prediction not in (0, 1):
        LOGGER.error("Smoke inference produced an invalid result: pred=%s proba=%s", prediction, proba)
        return 1
    LOGGER.info("Smoke inference OK: prediction=%s probability=%.4f", prediction, proba)

    # --- 2. metrics must clear the thresholds ------------------------------
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["test_metrics"]
    failures = []
    for key, minimum in thresholds.items():
        metric_name = key.replace("min_", "")
        value = metrics.get(metric_name)
        if value is None:
            failures.append(f"{metric_name}: missing from metrics.json")
        elif value < float(minimum):
            failures.append(f"{metric_name}={value:.4f} < required {float(minimum):.4f}")
        else:
            LOGGER.info("Gate passed: %s=%.4f >= %.4f", metric_name, value, float(minimum))

    if failures:
        LOGGER.error("Model validation FAILED: %s", "; ".join(failures))
        return 1

    LOGGER.info(
        "Model validation PASSED - %s (version %s) is cleared for deployment.",
        metadata.get("model_name"),
        metadata.get("model_version"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
