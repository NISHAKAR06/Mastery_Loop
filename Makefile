.PHONY: help up down logs ps db-init seed data verify test lint clean reset

.DEFAULT_GOAL := help

COMPOSE ?= docker compose -p masteryloop

# Load environment variables from .env if present
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

DB_USER ?= mastery
DB_NAME ?= masteryloop

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

