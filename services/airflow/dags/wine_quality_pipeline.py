"""End-to-end MLOps pipeline DAG (PMLDL Assignment 1).

Runs **every 5 minutes** and chains the three required stages:

┌──────────────────────┐  ┌───────────────────────┐  ┌────────────────────────┐
│ 1. data engineering  │→ │ 2. model engineering  │→ │ 3. deployment          │
│  download → clean →  │  │  features → train →   │  │  build images → run    │
│  split               │  │  evaluate → package   │  │  API + app → smoke test│
└──────────────────────┘  └───────────────────────┘  └────────────────────────┘

The deployment stage talks to the host Docker daemon through the mounted
socket, so the API and the web app run in their own containers, completely
separate from Airflow.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/opt/project"))
COMPOSE_FILE = PROJECT_ROOT / "code" / "deployment" / "docker-compose.yml"
COMPOSE_PROJECT = "pmldl-deployment"
API_PORT = os.environ.get("HOST_API_PORT", "8000")
APP_PORT = os.environ.get("HOST_APP_PORT", "8501")

DEFAULT_ARGS = {
    "owner": "pmldl",
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
    "execution_timeout": timedelta(minutes=8),
}

# Environment shared by every task: makes `code/` importable and points the
# training script at the MLflow tracking server.
TASK_ENV = {
    "PROJECT_ROOT": str(PROJECT_ROOT),
    "PYTHONPATH": f"{PROJECT_ROOT}/code",
    "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "PYTHONUNBUFFERED": "1",
}


def _log_pipeline_summary(**_) -> str:
    """Print the artifacts of the run so the whole pipeline is auditable from the UI."""
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    metadata_path = PROJECT_ROOT / "models" / "model_metadata.json"
    report_path = PROJECT_ROOT / "data" / "processed" / "data_report.json"

    lines = ["", "=" * 72, "PMLDL pipeline run summary", "=" * 72]

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        lines += [
            "[stage 1] data engineering",
            f"  rows loaded          : {report.get('rows_loaded')}",
            f"  duplicates removed   : {report.get('duplicates_removed')}",
            f"  outliers removed     : {report.get('outliers_removed')}",
            f"  train / test rows    : {report.get('train_rows')} / {report.get('test_rows')}",
            f"  class balance        : {report.get('class_balance')}",
        ]

    if metadata_path.exists() and metrics_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["test_metrics"]
        lines += [
            "[stage 2] model engineering",
            f"  champion model       : {metadata.get('model_name')}",
            f"  model version        : {metadata.get('model_version')}",
            "  test metrics         : "
            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)),
        ]

    lines += [
        "[stage 3] deployment",
        f"  model API            : http://localhost:{API_PORT}  (docs at /docs)",
        f"  web application      : http://localhost:{APP_PORT}",
        "=" * 72,
        "",
    ]
    summary = "\n".join(lines)
    print(summary)
    return summary


with DAG(
    dag_id="wine_quality_pipeline",
    description="Data engineering → model engineering → deployment, every 5 minutes",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
    # The pipeline must run on its schedule as soon as the stack is started.
    is_paused_upon_creation=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=20),
    tags=["pmldl", "mlops", "wine-quality"],
    doc_md=__doc__,
) as dag:

    # ----------------------------------------------------------------- #
    # Stage 1 - data engineering
    # ----------------------------------------------------------------- #
    with TaskGroup(group_id="data_engineering", tooltip="Load, clean and split the raw data") as stage_data:
        download_data = BashOperator(
            task_id="download_raw_data",
            bash_command="python code/datasets/download_data.py",
            cwd=str(PROJECT_ROOT),
            env=TASK_ENV,
            append_env=True,
            doc_md="Downloads the UCI Wine Quality dataset into `data/raw` (idempotent).",
        )

        prepare_data = BashOperator(
            task_id="clean_and_split_data",
            bash_command="python code/datasets/prepare_data.py",
            cwd=str(PROJECT_ROOT),
            env=TASK_ENV,
            append_env=True,
            doc_md="Cleans the data (duplicates, missing values, IQR outliers) and writes "
                   "`data/processed/train.csv` and `data/processed/test.csv`.",
        )

        download_data >> prepare_data

    # ----------------------------------------------------------------- #
    # Stage 2 - model engineering
    # ----------------------------------------------------------------- #
    with TaskGroup(group_id="model_engineering", tooltip="Train, evaluate and package the model") as stage_model:
        train_model = BashOperator(
            task_id="train_and_evaluate",
            bash_command="python code/models/train.py",
            cwd=str(PROJECT_ROOT),
            env=TASK_ENV,
            append_env=True,
            doc_md="Cross-validates the candidate models, refits the champion, evaluates it on "
                   "the test split, logs everything to MLflow and packages `models/model.joblib`.",
        )

        validate_model = BashOperator(
            task_id="validate_model",
            bash_command="python code/models/validate_model.py",
            cwd=str(PROJECT_ROOT),
            env=TASK_ENV,
            append_env=True,
            doc_md="Quality gate: blocks the deployment when the packaged model cannot be loaded "
                   "or does not reach the thresholds declared in `params.yaml`.",
        )

        train_model >> validate_model

    # ----------------------------------------------------------------- #
    # Stage 3 - deployment
    # ----------------------------------------------------------------- #
    with TaskGroup(group_id="deployment", tooltip="Build and run the API and the web app") as stage_deploy:
        build_and_deploy = BashOperator(
            task_id="build_and_deploy_containers",
            bash_command=(
                "docker compose -p {project} -f {compose_file} up -d --build "
                "--remove-orphans --wait --wait-timeout 300"
            ).format(project=COMPOSE_PROJECT, compose_file=COMPOSE_FILE),
            cwd=str(PROJECT_ROOT),
            env={**TASK_ENV, "API_PORT": API_PORT, "APP_PORT": APP_PORT, "DOCKER_BUILDKIT": "1"},
            append_env=True,
            execution_timeout=timedelta(minutes=15),
            doc_md="Rebuilds both images with the freshly trained model baked in and (re)starts "
                   "the `pmldl-api` and `pmldl-app` containers, waiting for their health checks.",
        )

        smoke_test = BashOperator(
            task_id="smoke_test_deployment",
            # NB: the trailing space stops Airflow from treating a command that
            # ends in ".sh" as a path to a Jinja template.
            bash_command="bash code/deployment/smoke_test.sh ",
            cwd=str(PROJECT_ROOT),
            env=TASK_ENV,
            append_env=True,
            doc_md="Verifies `/health`, `/model-info` and a real `/predict` call, and checks that "
                   "the app container can reach the API over the compose network.",
        )

        build_and_deploy >> smoke_test

    pipeline_summary = PythonOperator(
        task_id="pipeline_summary",
        python_callable=_log_pipeline_summary,
        doc_md="Prints the artifacts and endpoints produced by this pipeline run.",
    )

    stage_data >> stage_model >> stage_deploy >> pipeline_summary
