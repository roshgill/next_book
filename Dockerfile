# Stage 1: build Next.js static export
FROM node:20-slim AS web-builder
WORKDIR /build
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ ./
RUN npm run build
# Produces /build/out

# Stage 2: Python runtime
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch first (avoids bundling CUDA, saves ~2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY api/ ./api/
COPY data/processed/catalog_clean.parquet ./data/processed/
COPY models/ ./models/
COPY --from=web-builder /build/out ./web/out

EXPOSE 8000
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
