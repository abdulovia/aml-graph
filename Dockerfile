# syntax=docker/dockerfile:1
#
# Reproducible CPU image for AML-Graph.
#
# Targets a clean linux/amd64 environment — the standard pins in
# requirements.txt (torch>=2.2, polars>=1.0) install cleanly here, so none of
# the local Intel/Rosetta workarounds (torch==2.2.2, polars[rtcompat]) are
# baked in. The LightGBM subprocess isolation in src/lgbm_worker.py is harmless
# and remains.
FROM python:3.11-slim AS base

# libgomp1 provides the OpenMP runtime LightGBM links against at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# CUDA option (opt-in): to build a GPU image, comment out the CPU torch pin in
# requirements.txt and uncomment the line below to pull the cu121 wheels.
# RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
#     torch>=2.2 torch-geometric>=2.5
# ---------------------------------------------------------------------------

# Project code and pre-computed artifacts (figures + metrics).
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY pyproject.toml ./
COPY outputs ./outputs

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Default: launch the Streamlit demo. Override with docker-compose for the API.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
