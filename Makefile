.PHONY: help up down logs ps db-init seed data verify test lint clean reset mlflow-up migrate-m2 extract test-m2 baseline train hpo evaluate promote writeback verify-m2 m2

.DEFAULT_GOAL := help

COMPOSE ?= docker compose -p masteryloop

# Automatically load environment variables from .env if present
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

DB_USER ?= mastery
DB_NAME ?= masteryloop
POSTGRES_USER ?= $(DB_USER)
POSTGRES_DB ?= $(DB_NAME)

PY ?= PYTHONPATH=ml/src python
PKG := masteryloop
PSQL := $(COMPOSE) exec -T postgres psql -U $(DB_USER)

help: ## show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

up: ## start the full stack
	$(COMPOSE) up -d --build
	@echo "stack up — run 'make ps' to check health"

down: ## stop containers, keep volumes
	$(COMPOSE) down

reset: ## stop and wipe all volumes
	$(COMPOSE) down -v

ps: ## container health
	$(COMPOSE) ps

logs: ## tail all logs
	$(COMPOSE) logs -f --tail=100

db-init: ## re-apply schema (idempotent)
	$(COMPOSE) exec -T postgres psql -U $(DB_USER) -d $(DB_NAME) -f /docker-entrypoint-initdb.d/01_init.sql

seed: ## load reference + demo data
	python scripts/seed_db.py

data: ## download raw datasets
	python scripts/download_datasets.py

verify: ## M1 acceptance checks
	python scripts/verify_m1.py

test: ## unit tests for the domain library
	pytest tests/ -v

lint: ## format and lint
	black . && flake8 .

clean: ## remove python cache artifacts
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +

# ---- M2 targets ----
mlflow-up: ## stand MLflow up with postgres backend and minio artifacts
	@$(PSQL) -tc "SELECT 1 FROM pg_database WHERE datname='mlflow'" | grep -q 1 \
	|| $(PSQL) -c "CREATE DATABASE mlflow"
	@$(COMPOSE) exec -T minio sh -c \
	"mc alias set l http://localhost:9000 $$MINIO_ROOT_USER $$MINIO_ROOT_PASSWORD \
	&& mc mb -p l/mlflow-artifacts" || true
	$(COMPOSE) -f docker-compose.yml -f docker-compose.mlflow.yml up -d mlflow

migrate-m2: ## apply M2 database migration
	$(PSQL) -d $(DB_NAME) < db/migrations/002_m2_predictions.sql

extract: ## extract first-attempt responses and create learner split
	$(PY) -m $(PKG).split

test-m2: ## run leakage and protocol tests
	PYTHONPATH=ml/src pytest -q ml/tests/test_m2_leakage.py

baseline: ## evaluate classical test theory baselines
	$(PY) -m $(PKG).baseline

train: ## train 2PL IRT model with MLflow tracking
	$(PY) -m $(PKG).train

hpo: ## hyper-parameter search with Optuna
	$(PY) -m $(PKG).hpo --trials 30

evaluate: ## evaluate model on test learners and synthetic recovery harness
	$(PY) -m $(PKG).evaluate

promote: ## promote model to champion if all gates pass
	$(PY) -m $(PKG).promote

writeback: ## write champion calibrations back into items table
	$(PY) -m $(PKG).writeback

verify-m2: ## run M2 acceptance checks
	$(PY) scripts/verify_m2.py

m2: extract test-m2 train evaluate promote writeback verify-m2 ## run complete M2 pipeline

# ---- M3 targets: append below the M2 block. .RECIPEPREFIX applies only to rules
# written after it, so M1 and M2's tab-indented rules keep working and every M3
# rule survives being copied out of this PDF. ----
.RECIPEPREFIX = >
PY ?= python
M3_PATH := PYTHONPATH=ml/src:libs/psychocore/src:services/kt/src:.

prepare-kt:
> dvc repro prepare_kt

fit-bkt:
> dvc repro fit_bkt

train-sakt:
> dvc repro train_sakt

eval-kt:
> $(M3_PATH) $(PY) -m masteryloop_ml.kt.run_evaluate

migrate-m3:
> $(M3_PATH) alembic upgrade head

kt-service:
> $(M3_PATH) uvicorn kt_svc.main:app --port 8030

kt-consumer:
> $(M3_PATH) $(PY) -m kt_svc.run_consumer

test-m3:
> $(M3_PATH) $(PY) -m pytest -q tests/m3/unit

test-m3-int:
> $(M3_PATH) $(PY) -m pytest -q tests/m3/integration

verify-m3:
> $(M3_PATH) $(PY) scripts/verify_m3.py

m3: test-m3 prepare-kt fit-bkt train-sakt eval-kt test-m3-int verify-m3

.PHONY: prepare-kt fit-bkt train-sakt eval-kt migrate-m3 kt-service kt-consumer \
 test-m3 test-m3-int verify-m3 m3

