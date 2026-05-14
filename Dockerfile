# AURORA-XCon — CUDA runtime image with Python 3.11 venv
# Multi-stage build: base → builder (deps) → runtime (copy venv)

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
ENV OPENBLAS_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    wget \
    git \
    software-properties-common \
    ffmpeg \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV="/venv"

RUN python -m ensurepip && pip install --upgrade pip setuptools wheel

FROM base AS builder

# Some deps (e.g. pytinyrenderer) build from source and need g++; arm64 may still compile sdists.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .

RUN pip install -r requirements.txt

# JAX CUDA wheels may import-probe NVCC; PyPI publishes nvcc-cu12 from 12.3.x upward (no 12.2.x wheel).
RUN pip install "nvidia-cuda-nvcc-cu12>=12.3,<12.4"

FROM base AS runtime

COPY --from=builder /venv /venv

WORKDIR /workspace/src
ENV PYTHONPATH="/workspace/src"

CMD ["/bin/bash"]
