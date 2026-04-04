FROM python:3.11-slim

WORKDIR /app

# Install system deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache.
COPY requirements.txt* ./
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    httpx \
    pandas \
    numpy \
    scipy \
    pydantic \
    python-dotenv \
    websockets \
    || true

# Copy application source.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
