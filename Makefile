.DEFAULT_GOAL := help

COMPOSE_FILE := infra/docker-compose.yml
COMPOSE_ENV_FILE := .env
COMPOSE := docker compose --env-file $(COMPOSE_ENV_FILE) -f $(COMPOSE_FILE)
LAB_ENV := AL_MEDLIT_CELERY_BROKER_URL=redis://redis:6379/0 \
           AL_MEDLIT_CELERY_TASK_ALWAYS_EAGER=false \
           AL_MEDLIT_DEPLOYMENT_PROFILE=lab
UV := uv
# Keep in sync with frontend.Dockerfile so lockfile updates use the same npm as `npm ci` in Docker.
FRONTEND_NODE_IMAGE := node:20-alpine

.PHONY: help sync backend backend-deps-up frontend frontend-lock laptop-up laptop-down laptop-logs lab-up lab-down lab-logs runtime-up runtime-down db-current migrate seed test lint clean

help:
	@echo "AL-MedLit — top-level commands:"
	@echo ""
	@echo "  sync           Sync the root Python .venv with uv"
	@echo "  backend        Run the backend dev server on the host"
	@echo "  backend-deps-up Start PostgreSQL and MinIO for host backend development"
	@echo "  frontend       Run the frontend dev server"
	@echo "  frontend-lock  Update frontend package-lock.json with the Docker-matched npm (set ARGS=\"pkg\" to add deps)"
	@echo "  laptop-up      Start the default laptop Docker stack"
	@echo "  laptop-down    Stop the default laptop Docker stack"
	@echo "  laptop-logs    Tail logs from the default laptop stack"
	@echo "  lab-up         Start the lab Docker stack with lightweight control workers"
	@echo "  lab-down       Stop the lab Docker stack"
	@echo "  lab-logs       Tail logs from the lab stack"
	@echo "  runtime-up     Add one optional worker (RUNTIME=classical-cpu by default)"
	@echo "  runtime-down   Stop one optional worker profile"
	@echo "  db-current     Show the Docker database Alembic revision"
	@echo "  migrate        Run Alembic migrations against the configured database"
	@echo "  seed           Seed the configured database with demo data"
	@echo "  test           Run the backend test suite"
	@echo "  lint           Lint backend (ruff) and frontend (eslint)"
	@echo "  clean          Remove build artifacts and caches"

sync:
	$(UV) sync

backend:
	$(MAKE) -C backend dev

backend-deps-up:
	$(COMPOSE) up -d db minio

frontend:
	cd frontend && npm run dev

# Regenerate package-lock.json using the same npm as the Docker build (node:20-alpine),
# so `npm ci` inside the image stays in sync. The host npm (11.x) dedupes nested esbuild
# differently and corrupts the lockfile. Pass packages to add via ARGS, e.g.
#   make frontend-lock ARGS="lodash @types/lodash"
frontend-lock:
	docker run --rm -v "$(CURDIR)/frontend":/app -w /app $(FRONTEND_NODE_IMAGE) \
		npm install --package-lock-only --no-audit --no-fund $(ARGS)

laptop-up:
	$(COMPOSE) up --build -d

laptop-down:
	$(COMPOSE) down

laptop-logs:
	$(COMPOSE) logs -f

lab-up:
	$(LAB_ENV) $(COMPOSE) --profile lab up --build -d

lab-down:
	$(COMPOSE) --profile lab down

lab-logs:
	$(COMPOSE) --profile lab logs -f

RUNTIME ?= classical-cpu

runtime-up:
	$(LAB_ENV) $(COMPOSE) --profile lab --profile $(RUNTIME) up --build -d

runtime-down:
	$(COMPOSE) --profile lab --profile $(RUNTIME) down

db-current:
	$(COMPOSE) exec -T backend alembic current

migrate:
	$(MAKE) -C backend migrate

seed:
	$(MAKE) -C backend seed

test:
	$(MAKE) -C backend test

lint:
	$(MAKE) -C backend lint
	cd frontend && npm run lint

clean:
	$(MAKE) -C backend clean
	cd frontend && rm -rf dist node_modules/.vite
