# ────────────────────────────────────────────────────────────────────────────
# RAMP Bot — Cloud Run image
# Base: python:3.11-slim  (3.11 is the latest LTS; project uses 3.10+ syntax)
# Entry: src/entrypoint.py  (health probe + bot polling in one event loop)
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System packages
#   gcc / libffi-dev / libssl-dev  — needed to compile cryptography wheel
#   ca-certificates                — yfinance / requests need valid TLS roots
#   remaining packages             — Chromium system dependencies for Cloud Run
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
        ca-certificates \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libdbus-1-3 \
        libxkbcommon0 \
        libx11-6 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libcairo2 \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer — only rebuilds on requirements change) ─
# apt-get update is re-run here so playwright --with-deps can resolve any
# remaining Chromium dependencies after the cache was cleared above.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get update \
    && playwright install chromium --with-deps \
    && rm -rf /var/lib/apt/lists/*

# ── Application source ────────────────────────────────────────────────────────
COPY src/ ./src/

# ── Runtime data directories ──────────────────────────────────────────────────
# NOTE: Cloud Run containers are ephemeral — SQLite data is lost on restart.
# For production, migrate to Cloud SQL (PostgreSQL) and Cloud Storage for PDFs.
RUN mkdir -p /app/data/user_reports

# ── Environment ───────────────────────────────────────────────────────────────
# All secrets (RAMP_BOT_TOKEN, FINTECH_MASTER_KEY, ANTHROPIC_API_KEY) are
# injected at runtime via Cloud Run Secret Manager bindings — never bake them
# into the image.
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Cloud Run injects $PORT; default 8080 used by the health server in entrypoint.py
EXPOSE 8080

CMD ["python", "/app/src/entrypoint.py"]
