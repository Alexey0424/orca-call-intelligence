# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
#  ORCA_UI_SER_CONTROL — local control center for the SER pipeline
# ─────────────────────────────────────────────────────────────────────────────
#  Pure-Python FastAPI + Jinja2 + HTMX. No JS toolchain, no Node, no
#  matplotlib. Talks to ser-api (HTTP), Postgres (psycopg2), Ollama
#  (HTTP), and the Docker socket (Unix domain socket via httpx) to
#  observe and steer the running stack.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libpq for psycopg2; curl for the container's own /healthz check.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py            ./app.py
COPY config.py         ./config.py
COPY logging_setup.py  ./logging_setup.py
COPY clients/          ./clients/
COPY templates/        ./templates/

EXPOSE 8040

# Single uvicorn worker. The UI is read-mostly and the routes that
# trigger ser-api restarts are serialised by Docker anyway.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8040", "--workers", "1", "--log-level", "info"]
