FROM python:3.13-slim

WORKDIR /app

# System deps for numpy/pandas/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        python3-numpy \
        python3-scipy \
        python3-sklearn \
        python3-pandas \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/usr/lib/python3/dist-packages:/usr/lib/python3.13/dist-packages:${PYTHONPATH}"

COPY inference_service/requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --progress-bar off --default-timeout=120 --retries 10 --prefer-binary \
        fastapi==0.115.12 \
        "uvicorn[standard]==0.34.3" \
        python-multipart==0.0.20 \
        pydantic==2.11.4 \
        pydantic-settings==2.9.1 \
        python-dotenv==1.1.0 \
        httpx==0.28.1 \
        pytest==8.3.5 \
        pytest-asyncio==0.25.3
COPY inference_service/ .
COPY shared/ ./shared/

EXPOSE 8082

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
