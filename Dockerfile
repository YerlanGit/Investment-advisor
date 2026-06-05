# ────────────────────────────────────────────────────────────────────────────
# RAMP Bot — Cloud Run image
# Base: python:3.11-slim  (3.11 is the latest LTS; project uses 3.10+ syntax)
# Entry: src/entrypoint.py  (health probe + bot polling in one event loop)
# ────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System packages
#   gcc / libffi-dev / libssl-dev  — needed to compile cryptography wheel
#   ca-certificates                — requests need valid TLS roots
# Chromium system deps removed 2026-05-17: PDF generation was retired,
# reports now ship as interactive HTML URLs via Cloud Storage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        libssl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer — only rebuilds on requirements change) ─
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY src/ ./src/
# System prompt — MUST be present at /app/SYSTEM_PROMPT.md, where
# ai_narrative._build_system_prompt looks (src/../SYSTEM_PROMPT.md).
# Previously never copied → the bot ran on a 2-line fallback prompt in
# production, silently dropping the analyst persona + injection guardrail.
COPY SYSTEM_PROMPT.md ./
# Test suite — copied so the CI test-gate (`docker run … python -m unittest
# discover -s /app/tests`) finds them inside the just-built image.  Files
# are tiny (~500 KB), Python-only, and the runtime entrypoint never imports
# from /app/tests, so there is zero behavioural impact in production.
COPY tests/ ./tests/

# ── Runtime data directories ──────────────────────────────────────────────────
# Tokenomics SQLite lives on a gcsfuse-mounted persistent volume in
# production (TOKENOMICS_DB_PATH=/mnt/state/tokenomics.db, set in
# cloudbuild.yaml).  The /app/data fallback below is only used in local dev.
# Reports are rendered to /tmp/user_reports (writable, ephemeral) and then
# uploaded to GCS via services/report_storage.py — see REPORT_BUCKET_NAME.
# /app/data/chroma_db is pre-warmed on boot from gs://ramp-bot-chroma-db/
# (see entrypoint.py:_download_chroma_db) so RAG sees the freshest snapshot.
RUN mkdir -p /app/data/chroma_db

# Pre-bake the Chroma ONNX embedding model (79 MB) so it's in the image layer.
# Without this every first ChromaDB query downloads it from S3, adding ~2s latency.
RUN python -c \
  "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; \
   DefaultEmbeddingFunction()([])" 2>/dev/null || true

# ── Environment ───────────────────────────────────────────────────────────────
# All secrets (RAMP_BOT_TOKEN, FINTECH_MASTER_KEY, ANTHROPIC_API_KEY) are
# injected at runtime via Cloud Run Secret Manager bindings — never bake them
# into the image.
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Cloud Run injects $PORT; default 8080 used by the health server in entrypoint.py
EXPOSE 8080

CMD ["python", "/app/src/entrypoint.py"]
