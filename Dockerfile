# syntax=docker/dockerfile:1
#
# AML-Graph images (multi-stage):
#
#   serve (default) — slim image for the demo surfaces (Streamlit + FastAPI).
#                     No torch/lightgbm: the demos read the pre-computed
#                     scored-edge artifact, so the image stays small and fast.
#   train           — full stack (torch, torch-geometric, lightgbm, mlflow)
#                     for reproducing the pipeline inside Docker:
#                     docker build --target train -t aml-graph:train .
#                     docker run --rm -v ./data:/app/data -v ./outputs:/app/outputs \
#                       aml-graph:train python -c "from src import pipeline; pipeline.run_mvp()"
#
# Targets clean linux; none of the local Intel/Rosetta workarounds are baked in.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY requirements-serve.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements-serve.txt

COPY src ./src
COPY app ./app
COPY configs ./configs
COPY pyproject.toml ./
# Pre-computed artifacts so the image also works without volume mounts.
COPY outputs ./outputs

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app

# --- full training stack (opt-in target) -----------------------------------
FROM base AS train
USER root
# libgomp1 provides the OpenMP runtime LightGBM links against at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install -r requirements.txt
# CUDA option: swap the torch pin for the cu121 wheels if a GPU is available.
# RUN pip install --index-url https://download.pytorch.org/whl/cu121 "torch>=2.2"
USER appuser

# --- default: slim serving image --------------------------------------------
FROM base AS serve
USER appuser

EXPOSE 8501 8000

# Default: the Streamlit demo. docker-compose overrides this for the API.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
