.DEFAULT_GOAL := help

ROOT_DIR := ..
PYTHON := $(ROOT_DIR)/.venv/bin/python
UV := uv

-include $(ROOT_DIR)/.env
POSTGRES_USER ?= al_medlit
POSTGRES_PASSWORD ?= al_medlit
POSTGRES_HOST ?= localhost
POSTGRES_PORT ?= 5432
POSTGRES_DB ?= al_medlit
AL_MEDLIT_JWT_SECRET ?=
AL_MEDLIT_STORAGE_SECURE ?= true
AL_MEDLIT_STORAGE_ENDPOINT ?= localhost:9000
AL_MEDLIT_STORAGE_BUCKET ?= al-medlit
AL_MEDLIT_STORAGE_ACCESS_KEY ?= al-medlit-app
AL_MEDLIT_STORAGE_SECRET_KEY ?=
AL_MEDLIT_AUTH_COOKIE_SECURE ?= false
AL_MEDLIT_DEPLOYMENT_PROFILE ?= laptop
AL_MEDLIT_STORAGE_CA_CERT_PATH ?= $(abspath $(ROOT_DIR)/infra/certs/minio/CAs/ca.crt)
export AL_MEDLIT_DATABASE_URL ?= postgresql+psycopg2://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)
export AL_MEDLIT_JWT_SECRET
export AL_MEDLIT_STORAGE_SECURE
export AL_MEDLIT_STORAGE_CA_CERT_PATH
export AL_MEDLIT_STORAGE_ENDPOINT AL_MEDLIT_STORAGE_BUCKET AL_MEDLIT_STORAGE_ACCESS_KEY
export AL_MEDLIT_STORAGE_SECRET_KEY AL_MEDLIT_AUTH_COOKIE_SECURE AL_MEDLIT_DEPLOYMENT_PROFILE

.PHONY: help sync install install-dev dev migrate test lint format seed clean

help:
	@echo "AL-MedLit Backend — available commands:"
	@echo ""
	@echo "  sync          Sync the root Python .venv with uv"
	@echo "  install       Sync production dependencies"
	@echo "  install-dev   Sync all dependencies including dev tools"
	@echo "  dev           Run the development server with auto-reload"
	@echo "  migrate       Apply Alembic migrations to the configured database"
	@echo "  test          Run the test suite"
	@echo "  lint          Lint the source code with ruff"
	@echo "  format        Auto-format the source code with ruff"
	@echo "  seed          Seed the database with demo data"
	@echo "  clean         Remove build artifacts and caches"

sync:
	cd $(ROOT_DIR) && $(UV) sync

install:
	cd $(ROOT_DIR) && $(UV) sync --no-dev

install-dev:
	cd $(ROOT_DIR) && $(UV) sync

dev: migrate
	$(PYTHON) -m uvicorn al_medlit.main:app --reload

migrate:
	$(PYTHON) -m alembic upgrade head

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check al_medlit/ tests/ scripts/

format:
	$(PYTHON) -m ruff format al_medlit/ tests/ scripts/
	$(PYTHON) -m ruff check --fix al_medlit/ tests/ scripts/

seed: migrate
	$(PYTHON) -m scripts.seed_demo

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
