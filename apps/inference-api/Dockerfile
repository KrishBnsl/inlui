# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY apps/inference-api/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install --prefer-binary --timeout 120 --retries 20 --resume-retries 20 \
      -r /tmp/requirements.txt

COPY packages/josaa-core /tmp/josaa-core
COPY apps/inference-api/pyproject.toml /tmp/inference-api/pyproject.toml
COPY apps/inference-api/src /tmp/inference-api/src
RUN python -m pip install --no-cache-dir --no-deps /tmp/josaa-core /tmp/inference-api

EXPOSE 8082

CMD ["uvicorn", "inference_api.main:app", "--host", "0.0.0.0", "--port", "8082", "--timeout-keep-alive", "10"]
