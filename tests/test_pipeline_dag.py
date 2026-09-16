"""Pipeline tests: the Airflow DAG is importable and scheduled every 5 minutes.

Skipped automatically when Airflow is not installed in the current environment
(the DAG always runs inside the Airflow container).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

DAGS_FOLDER = Path(__file__).resolve().parents[1] / "services" / "airflow" / "dags"
DAG_ID = "wine_quality_pipeline"


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(DAGS_FOLDER), include_examples=False)


def test_dag_imports_without_errors(dagbag: DagBag) -> None:
    assert dagbag.import_errors == {}, dagbag.import_errors
    assert DAG_ID in dagbag.dags


def test_dag_runs_every_five_minutes(dagbag: DagBag) -> None:
    dag = dagbag.dags[DAG_ID]
    assert dag.schedule_interval == "*/5 * * * *"
    assert dag.catchup is False
    assert dag.max_active_runs == 1


def test_dag_contains_the_three_required_stages(dagbag: DagBag) -> None:
    task_ids = set(dagbag.dags[DAG_ID].task_ids)
    expected = {
        "data_engineering.download_raw_data",
        "data_engineering.clean_and_split_data",
        "model_engineering.train_and_evaluate",
        "model_engineering.validate_model",
        "deployment.build_and_deploy_containers",
        "deployment.smoke_test_deployment",
    }
    assert expected <= task_ids


def test_stages_run_in_order(dagbag: DagBag) -> None:
    dag = dagbag.dags[DAG_ID]
    downstream = dag.get_task("data_engineering.clean_and_split_data").get_flat_relative_ids(upstream=False)
    assert "model_engineering.train_and_evaluate" in downstream
    assert "deployment.build_and_deploy_containers" in downstream
