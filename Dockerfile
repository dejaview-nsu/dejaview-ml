# syntax=docker/dockerfile:1

ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

# base: CUDA, Python 3.12, system/service dependencies and application code.
# Ubuntu 22.04's default Python 3.10 does not satisfy requires-python >=3.11.
FROM ${BASE_IMAGE} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    ML_CACHE_DIR=/models/huggingface \
    ML_DEVICE=auto

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app

# Select matching CUDA wheels explicitly. Local CPU builds override the index.
ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
RUN python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"

# pyproject.toml is the single source of dependencies. This layer stays cached
# when only application code changes.
COPY pyproject.toml ./
RUN python -c 'import pathlib, tomllib; p = tomllib.loads(pathlib.Path("pyproject.toml").read_text()); pathlib.Path("/tmp/requirements.txt").write_text("\n".join(p["project"]["dependencies"]))' \
    && python -m pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY src ./src
RUN python -m pip install --no-deps .

# test: CI-only dependencies, tests and the original contract fixture.
FROM base AS test

RUN python -c 'import pathlib, tomllib; p = tomllib.loads(pathlib.Path("pyproject.toml").read_text()); pathlib.Path("/tmp/requirements-dev.txt").write_text("\n".join(p["project"]["optional-dependencies"]["dev"]))' \
    && python -m pip install -r /tmp/requirements-dev.txt \
    && rm /tmp/requirements-dev.txt

COPY tests ./tests
COPY ml-service.openapi.yaml ./
CMD ["sh", "-c", "ruff check src tests && python -m pytest -q"]

# runtime: last stage, built by default; inherits base, not the test stage.
FROM base AS runtime

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "dejaview_ml.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
