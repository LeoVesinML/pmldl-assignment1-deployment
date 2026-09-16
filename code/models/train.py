"""Stage 2 - model engineering: feature engineering, training, evaluation, packaging.

Input  artifacts : ``data/processed/train.csv``, ``data/processed/test.csv``
Output artifacts : ``models/model.joblib``      - the packaged end-to-end pipeline
                   ``models/metrics.json``      - test metrics of the champion model
                   ``models/model_metadata.json`` - everything the API needs to
                                                    describe / validate itself

Several candidate models are cross-validated on the training split, the best one
(by ``model.selection_metric``) is refitted and evaluated on the untouched test
split, and every run - parameters, metrics, artifacts and the model itself - is
logged to MLflow.  If the MLflow server is unreachable the run is transparently
logged to a local file store so the pipeline never breaks because of tracking.

Usage
-----
    python code/models/train.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_logger, load_params, resolve  # noqa: E402
from features import (  # noqa: E402
    RAW_CATEGORICAL_FEATURES,
    RAW_FEATURES,
    RAW_NUMERIC_FEATURES,
    TARGET,
    add_engineered_features,
    build_preprocessor,
)

LOGGER = get_logger("stage2.train")


# --------------------------------------------------------------------------- #
# Candidate models
# --------------------------------------------------------------------------- #
def build_candidate(name: str, random_state: int):
    """Return an untrained estimator for the requested candidate name."""
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model candidate: {name!r}")


def build_pipeline(model_name: str, random_state: int) -> Pipeline:
    """Raw input -> engineered features -> preprocessing -> classifier."""
    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(add_engineered_features, validate=False),
            ),
            ("preprocessor", build_preprocessor()),
            ("classifier", build_candidate(model_name, random_state)),
        ]
    )


# --------------------------------------------------------------------------- #
# Data / metrics helpers
# --------------------------------------------------------------------------- #
def load_processed() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    processed_dir = resolve(load_params()["data"]["processed_dir"])
    train_path, test_path = processed_dir / "train.csv", processed_dir / "test.csv"
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing - run code/datasets/prepare_data.py (stage 1) first."
            )
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    LOGGER.info("Loaded train=%s test=%s", train.shape, test.shape)

    missing = [c for c in RAW_FEATURES if c not in train.columns]
    if missing:
        raise ValueError(f"Processed data is missing expected columns: {missing}")

    return (
        train[RAW_FEATURES],
        train[TARGET],
        test[RAW_FEATURES],
        test[TARGET],
    )


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
    }


def feature_ranges(x_train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per-feature statistics used by the web app to build sensible inputs."""
    ranges: Dict[str, Dict[str, float]] = {}
    for column in RAW_NUMERIC_FEATURES:
        series = pd.to_numeric(x_train[column], errors="coerce").dropna()
        ranges[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p01": float(series.quantile(0.01)),
            "p99": float(series.quantile(0.99)),
        }
    return ranges


# --------------------------------------------------------------------------- #
# MLflow (with a graceful local fallback)
# --------------------------------------------------------------------------- #
def setup_mlflow():
    """Return the ``mlflow`` module configured against the tracking server.

    Falls back to a local ``mlruns`` file store (and finally to ``None``) so a
    missing tracking server can never fail the pipeline.
    """
    cfg = load_params()["mlflow"]
    # Fail fast when the tracking server is down instead of retrying for a minute.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    try:
        import mlflow  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError:  # pragma: no cover
        LOGGER.warning("mlflow is not installed - skipping experiment tracking.")
        return None

    uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["tracking_uri"])
    for candidate, label in ((uri, "tracking server"), (f"file:{resolve('mlruns')}", "local store")):
        try:
            mlflow.set_tracking_uri(candidate)
            mlflow.set_experiment(cfg["experiment_name"])
            LOGGER.info("MLflow %s ready at %s", label, candidate)
            return mlflow
        except Exception as exc:  # noqa: BLE001 - tracking must never break training
            LOGGER.warning("MLflow %s at %s is unavailable: %s", label, candidate, exc)
    return None


def _git_revision() -> str:
    import subprocess  # noqa: PLC0415

    try:
        return subprocess.run(
            # `-c safe.directory=*` keeps this working inside the Airflow
            # container, where the repository is owned by another UID.
            ["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"],
            cwd=str(resolve(".")),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main() -> int:
    params = load_params()
    model_cfg, mlflow_cfg = params["model"], params["mlflow"]
    random_state = int(model_cfg["random_state"])
    selection_metric = str(model_cfg["selection_metric"])

    x_train, y_train, x_test, y_test = load_processed()
    mlflow = setup_mlflow()

    cv = StratifiedKFold(n_splits=int(model_cfg["cv_folds"]), shuffle=True, random_state=random_state)
    leaderboard: Dict[str, Dict[str, Any]] = {}
    best_name, best_score, best_pipeline = None, -np.inf, None

    run_name = f"pipeline-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    parent_run = mlflow.start_run(run_name=run_name) if mlflow else None

    try:
        # ---------------- candidate selection (cross-validation) -------------
        for name in model_cfg["candidates"]:
            pipeline = build_pipeline(name, random_state)
            started = time.perf_counter()
            scores = cross_val_score(
                pipeline, x_train, y_train, cv=cv, scoring=selection_metric, n_jobs=1
            )
            elapsed = time.perf_counter() - started
            leaderboard[name] = {
                f"cv_{selection_metric}_mean": float(scores.mean()),
                f"cv_{selection_metric}_std": float(scores.std()),
                "cv_seconds": round(elapsed, 2),
            }
            LOGGER.info(
                "Candidate %-24s cv_%s = %.4f (+/- %.4f) in %.1fs",
                name,
                selection_metric,
                scores.mean(),
                scores.std(),
                elapsed,
            )
            if mlflow:
                with mlflow.start_run(run_name=f"candidate-{name}", nested=True):
                    mlflow.log_param("model_type", name)
                    mlflow.log_params(
                        {f"clf__{k}": v for k, v in build_candidate(name, random_state).get_params().items()}
                    )
                    mlflow.log_metrics(leaderboard[name])
            if scores.mean() > best_score:
                best_name, best_score, best_pipeline = name, float(scores.mean()), pipeline

        assert best_pipeline is not None and best_name is not None
        LOGGER.info("Champion model: %s (cv_%s=%.4f)", best_name, selection_metric, best_score)

        # ---------------- refit + evaluation on the held-out test set --------
        started = time.perf_counter()
        best_pipeline.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started

        y_pred = best_pipeline.predict(x_test)
        y_proba = best_pipeline.predict_proba(x_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics[f"cv_{selection_metric}_mean"] = best_score
        metrics["train_seconds"] = round(train_seconds, 2)

        LOGGER.info("Test metrics: %s", json.dumps(metrics, indent=2))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        LOGGER.info("Confusion matrix [[TN, FP], [FN, TP]]: %s", cm)

        # ---------------- packaging -----------------------------------------
        model_dir = resolve(model_cfg["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / model_cfg["model_file"]
        joblib.dump(best_pipeline, model_path)
        LOGGER.info("Saved model to %s (%.1f KiB)", model_path, model_path.stat().st_size / 1024)

        (model_dir / model_cfg["metrics_file"]).write_text(
            json.dumps({"test_metrics": metrics, "leaderboard": leaderboard}, indent=2),
            encoding="utf-8",
        )

        metadata = {
            "model_name": best_name,
            "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_revision": _git_revision(),
            "selection_metric": selection_metric,
            "metrics": metrics,
            "leaderboard": leaderboard,
            "confusion_matrix": cm,
            "classification_report": report,
            "target": TARGET,
            "positive_class_meaning": (
                f"expert quality score >= {params['data']['good_quality_threshold']}"
            ),
            "raw_numeric_features": RAW_NUMERIC_FEATURES,
            "raw_categorical_features": RAW_CATEGORICAL_FEATURES,
            "wine_types": sorted(x_train["wine_type"].astype(str).unique().tolist()),
            "feature_ranges": feature_ranges(x_train),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        }

        # ---------------- MLflow logging -------------------------------------
        if mlflow:
            try:
                mlflow.log_params(
                    {
                        "champion_model": best_name,
                        "candidates": ",".join(model_cfg["candidates"]),
                        "cv_folds": model_cfg["cv_folds"],
                        "selection_metric": selection_metric,
                        "random_state": random_state,
                        "train_rows": len(x_train),
                        "test_rows": len(x_test),
                        "good_quality_threshold": params["data"]["good_quality_threshold"],
                    }
                )
                mlflow.log_metrics(metrics)
                mlflow.log_dict(metadata, "model_metadata.json")
                mlflow.log_dict({"confusion_matrix": cm, "classification_report": report}, "evaluation.json")
                mlflow.sklearn.log_model(
                    sk_model=best_pipeline,
                    artifact_path="model",
                    registered_model_name=mlflow_cfg["registered_model_name"],
                    # Cast to float so MLflow does not warn about integer columns
                    # that cannot represent missing values at inference time.
                    input_example=x_train.head(3).astype(
                        {c: "float64" for c in RAW_NUMERIC_FEATURES}
                    ),
                )
                run = mlflow.active_run()
                metadata["mlflow_run_id"] = run.info.run_id if run else None
                LOGGER.info("Logged run %s to MLflow.", metadata.get("mlflow_run_id"))
            except Exception as exc:  # noqa: BLE001 - tracking must not fail the stage
                LOGGER.warning("MLflow logging failed (continuing): %s", exc)

        (model_dir / model_cfg["metadata_file"]).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        LOGGER.info("Stage 2 finished successfully.")
    finally:
        if parent_run is not None and mlflow is not None:
            try:
                mlflow.end_run()
            except Exception:  # noqa: BLE001, S110
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
