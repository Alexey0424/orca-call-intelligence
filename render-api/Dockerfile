# syntax=docker/dockerfile:1.7

# ─────────────────────────────────────────────────────────────────────────────
#  ser-api-render Dockerfile — optimized for smaller final image and faster
#  rebuilds while keeping WeasyPrint runtime dependencies intact.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/venv/bin:$PATH"

WORKDIR /app

# Build deps + runtime libs needed during wheel install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libfreetype6-dev \
    libglib2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz-subset0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv

COPY requirements-render.txt ./requirements-render.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements-render.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/venv/bin:$PATH"

WORKDIR /app

# Runtime-only libs for WeasyPrint and related native bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libglib2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz-subset0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi8 \
    shared-mime-info \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
COPY app-render.py    ./app-render.py
COPY config.py        ./config.py
COPY logging_setup.py ./logging_setup.py
COPY utils.py         ./utils.py
COPY payload.py       ./payload.py
COPY exec_view.py     ./exec_view.py
COPY insights.py      ./insights.py
COPY rendering.py     ./rendering.py
COPY dashboard.py     ./dashboard.py
COPY schemas.py       ./schemas.py
COPY charts/          ./charts/
COPY templates/       ./templates/
COPY tests/           ./tests/

EXPOSE 8010

CMD ["python", "/app/app-render.py"]
