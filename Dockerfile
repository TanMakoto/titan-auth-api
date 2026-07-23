# ============================================================
# Dockerfile for titan-auth-api (Face AI API)
# Uses Python 3.11-slim for TensorFlow/DeepFace compatibility
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Install system libraries needed by OpenCV and TensorFlow on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .

# Expose port (Render injects $PORT at runtime)
EXPOSE 8000

# Start command — uses $PORT from Render
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
