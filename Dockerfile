# ============================================================
# Dockerfile for titan-auth-api
# Compatible with: Render.com (Docker) AND HuggingFace Spaces
# Python 3.11-slim — TensorFlow/DeepFace compatible
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Install system libraries for OpenCV + TensorFlow on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# HuggingFace Spaces requires port 7860
# Render.com injects $PORT
# Default fallback: 7860 (HF) — change to 8000 for local
EXPOSE 7860

CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}
