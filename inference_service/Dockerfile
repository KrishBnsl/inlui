FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/pandas/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8082

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8082"]
