# ── Base image ──────────────────────────────────────────────────────────────
# python:3.11-slim is a minimal Debian image with Python 3.11.
# "slim" removes unnecessary files (docs, tests) to reduce image size.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# ── Working directory ─────────────────────────────────────────────────────
# All commands from here on run inside /app inside the container.
WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────
# These are needed by some Python packages (faiss, sentence-transformers).
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────
# WHY copy requirements.txt BEFORE the rest of the code?
# Docker builds in layers. If only your code changes (not requirements),
# Docker reuses the cached pip install layer — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────
COPY . .

# ── Create directories for runtime data ───────────────────────────────────
# In Cloud Run, ephemeral storage is fine for demo purposes.
# For production, mount a persistent volume or use cloud storage here.
RUN mkdir -p uploads index

# ── Port ──────────────────────────────────────────────────────────────────
# Cloud Run expects the app to listen on PORT env variable (defaults to 8080)
EXPOSE 8080

# ── Start command ─────────────────────────────────────────────────────────
# $PORT is set automatically by Cloud Run. 0.0.0.0 makes it accessible
# from outside the container (required for Docker networking).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
