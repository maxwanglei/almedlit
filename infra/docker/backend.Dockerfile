FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PATH="/app/.venv/bin:$PATH"

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
COPY backend/pyproject.toml backend/README.md backend/
COPY backend/al_medlit/ backend/al_medlit/
RUN uv sync --frozen --no-dev --package al-medlit-backend --no-editable
COPY backend/alembic.ini backend/
COPY backend/migrations/ backend/migrations/
COPY backend/scripts/ backend/scripts/

RUN groupadd --system almedlit \
	&& useradd --system --gid almedlit --create-home --home-dir /home/almedlit almedlit \
	&& mkdir -p /var/lib/al-medlit/attempts \
	&& chown -R almedlit:almedlit /app /var/lib/al-medlit

USER almedlit
WORKDIR /app/backend

CMD ["python", "-m", "scripts.start_api"]
