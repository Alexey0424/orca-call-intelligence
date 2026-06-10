# syntax=docker/dockerfile:1.7
# rag-api — CPU-only read/serve path (search + dashboard + on-demand report).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY rag-api/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY rag-api/app.py            ./app.py
COPY rag-api/logging_setup.py  ./logging_setup.py
COPY rag-api/retrieval.py      ./retrieval.py
COPY rag-api/dashboard_data.py ./dashboard_data.py
COPY rag-api/dashboards.py     ./dashboards.py
COPY rag-api/report.py         ./report.py
COPY rag-api/tests/            ./tests/
COPY packages/ragcore/         ./ragcore/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120"]
