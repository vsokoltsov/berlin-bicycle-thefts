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

RUN uv sync --no-dev

FROM base AS api

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "bicycle_theft.api:api", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base AS ml

EXPOSE 8888

# Ставим Jupyter без изменения зависимостей проекта
# RUN /opt/venv/bin/pip install --no-cache-dir jupyterlab ipykernel