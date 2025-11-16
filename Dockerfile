# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    MPLBACKEND=Agg \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl ca-certificates git \
    g++ \
    libgomp1 \
    gdal-bin libgdal-dev \
    libgeos-dev \
    proj-bin libproj-dev \
    libspatialindex-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv uv sync --no-dev --frozen

FROM base AS api

RUN mkdir -p /app/data/processed
COPY ["data/raw/Bicycle Theft Data.csv", "/app/data/raw/Bicycle Theft Data.csv"]

COPY data/raw/population.csv /app/data/raw/population.csv
COPY data/processed/preproc.json /app/data/processed/preproc.json
COPY data/processed/bike_thefts_lgbm.pkl /app/data/processed/bike_thefts_lgbm.pkl

EXPOSE 8080

ENV UV_NO_SYNC=1
CMD ["sh", "-c", "python -m bicycle_theft.api --host 0.0.0.0 --port ${PORT:-8080}"]

FROM base AS ml

EXPOSE 8888