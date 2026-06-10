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
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8081

# main.py imports the factory → `main:app` works with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
