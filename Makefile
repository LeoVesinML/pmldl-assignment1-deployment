# ---------------------------------------------------------------------------
# Convenience targets for the PMLDL Assignment 1 pipeline.
# ---------------------------------------------------------------------------
COMPOSE_INFRA := docker compose -f docker-compose.yml
COMPOSE_DEPLOY := docker compose -p pmldl-deployment -f code/deployment/docker-compose.yml
PYTHON ?= python3

.DEFAULT_GOAL := help
.PHONY: help up down restart logs status trigger pipeline-local data train validate deploy \
        deploy-down smoke test clean clean-all urls unpause

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Build and start Airflow + MLflow (the pipeline then runs every 5 minutes)
	$(COMPOSE_INFRA) up -d --build
	@$(MAKE) --no-print-directory urls

down: ## Stop Airflow + MLflow
	$(COMPOSE_INFRA) down

restart: down up ## Restart the infrastructure

logs: ## Follow the Airflow scheduler logs
	$(COMPOSE_INFRA) logs -f airflow-scheduler

status: ## Show the state of every container of the project
	@docker ps --filter "name=pmldl-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

trigger: ## Trigger one pipeline run immediately (without waiting for the schedule)
	$(COMPOSE_INFRA) exec airflow-scheduler airflow dags trigger wine_quality_pipeline

unpause: ## Make sure the DAG is not paused
	$(COMPOSE_INFRA) exec airflow-scheduler airflow dags unpause wine_quality_pipeline

pipeline-local: data train validate deploy smoke ## Run the whole pipeline locally (without Airflow)

data: ## Stage 1 - download, clean and split the data
	$(PYTHON) code/datasets/download_data.py
	$(PYTHON) code/datasets/prepare_data.py

train: ## Stage 2 - train, evaluate and package the model
	$(PYTHON) code/models/train.py

validate: ## Stage 2b - model quality gate
	$(PYTHON) code/models/validate_model.py

deploy: ## Stage 3 - build and start the API and the web app containers
	$(COMPOSE_DEPLOY) up -d --build --wait --wait-timeout 300
	@$(MAKE) --no-print-directory urls

deploy-down: ## Stop the API and the web app containers
	$(COMPOSE_DEPLOY) down

smoke: ## Verify the deployment end to end
	bash code/deployment/smoke_test.sh

test: ## Run the unit tests
	$(PYTHON) -m pytest tests -v

urls: ## Print the URLs of every service
	@echo ""
	@echo "  Web application : http://localhost:$${APP_PORT:-8501}"
	@echo "  Model API       : http://localhost:$${API_PORT:-8000}       (docs: /docs)"
	@echo "  Airflow UI      : http://localhost:$${AIRFLOW_PORT:-8080}   (admin / admin)"
	@echo "  MLflow UI       : http://localhost:$${MLFLOW_PORT:-5555}"
	@echo ""

clean: ## Remove generated data and model artifacts
	rm -f data/processed/*.csv data/processed/*.json
	rm -f models/*.joblib models/*.json

clean-all: down deploy-down clean ## Stop everything and remove artifacts and volumes
	$(COMPOSE_INFRA) down -v
