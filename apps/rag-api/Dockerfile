# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Install system deps needed by FAISS and Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg62-turbo-dev \
        libpng-dev \
        libwebp-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (cached layer)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install --progress-bar off --default-timeout=120 --retries 20 \
      --resume-retries 20 \
      --prefer-binary -r requirements.txt

# Install only the importable service package. The local .env is excluded by
# .dockerignore and runtime configuration is provided by Compose/environment.
COPY pyproject.toml .
COPY src ./src
RUN python -m pip install --no-cache-dir --no-deps .

EXPOSE 8081

CMD ["uvicorn", "rag_api.main:app", "--host", "0.0.0.0", "--port", "8081"]
