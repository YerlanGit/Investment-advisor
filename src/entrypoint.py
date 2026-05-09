"""
Cloud Run entrypoint — runs the Telegram bot alongside a minimal HTTP health
server so Cloud Run's startup probe succeeds.

Cloud Run requires a container to bind on $PORT (default 8080). The bot uses
long-polling, not webhooks, so this thin stdlib-only server handles the probe.

On startup, the entrypoint also pre-warms the on-disk ChromaDB by syncing
the latest snapshot from gs://ramp-bot-chroma-db/chroma_db/. The bucket is
populated by the Cloud Function `cloud_function/main.py` whenever a new
bank-analytic PDF is uploaded — so each container restart picks up the
freshest knowledge base without a redeploy.
"""

import asyncio
import logging
import os
import sys

# yfinance был удалён 2026-04. Цены теперь берутся через Tradernet
# (freedom_portfolio.history); кеш истории живёт в /tmp/freedom_history_cache.

logger = logging.getLogger("ramp.entrypoint")

# ── ChromaDB sync from GCS ──────────────────────────────────────────────────
CHROMA_LOCAL_PATH  = os.environ.get("CHROMA_LOCAL_PATH",  "/app/data/chroma_db")
CHROMA_GCS_BUCKET  = os.environ.get("CHROMA_BUCKET",      "ramp-bot-chroma-db")
CHROMA_GCS_PREFIX  = os.environ.get("CHROMA_GCS_PREFIX",  "chroma_db/")


def _download_chroma_db() -> None:
    """
    Mirror gs://<CHROMA_BUCKET>/<CHROMA_GCS_PREFIX>* into CHROMA_LOCAL_PATH.

    Behaviour:
      • Empty bucket / no prefix         → log + continue (RAG works empty)
      • google-cloud-storage missing     → log + continue (degrade gracefully)
      • Auth / network failure           → log + continue (don't block start)
      • Successful sync                  → log how many blobs were pulled

    The function NEVER raises; bot start must not depend on RAG availability.
    """
    try:
        from google.cloud import storage   # noqa: PLC0415  — optional dep
    except Exception as exc:
        logger.info("ChromaDB sync skipped (google-cloud-storage missing): %s", exc)
        return

    try:
        client = storage.Client()
        bucket = client.bucket(CHROMA_GCS_BUCKET)
        os.makedirs(CHROMA_LOCAL_PATH, exist_ok=True)

        blobs = list(bucket.list_blobs(prefix=CHROMA_GCS_PREFIX))
        if not blobs:
            logger.info(
                "ChromaDB bucket gs://%s/%s is empty — bot starts with empty RAG.",
                CHROMA_GCS_BUCKET, CHROMA_GCS_PREFIX,
            )
            return

        downloaded = 0
        for blob in blobs:
            if blob.name.endswith("/"):
                continue   # skip directory placeholders
            relative = blob.name[len(CHROMA_GCS_PREFIX):]
            if not relative:
                continue
            dest = os.path.join(CHROMA_LOCAL_PATH, relative)
            os.makedirs(os.path.dirname(dest) or CHROMA_LOCAL_PATH, exist_ok=True)
            blob.download_to_filename(dest)
            downloaded += 1

        logger.info(
            "ChromaDB synced from gs://%s/%s → %s (%d blobs).",
            CHROMA_GCS_BUCKET, CHROMA_GCS_PREFIX, CHROMA_LOCAL_PATH, downloaded,
        )
    except Exception as exc:
        # Auth, permission, network — log everything, never block bot start.
        logger.warning("ChromaDB sync failed (continuing anyway): %s", exc)


async def _health_server() -> None:
    """Minimal HTTP/1.0 server: any GET → 200 OK."""
    port = int(os.environ.get("PORT", 8080))

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.wait_for(reader.read(4096), timeout=5)
        except asyncio.TimeoutError:
            pass
        writer.write(
            b"HTTP/1.0 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 2\r\n"
            b"\r\n"
            b"OK"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(_handle, "0.0.0.0", port)
    logger.info("Health server listening on :%s", port)
    async with server:
        await server.serve_forever()


async def _main() -> None:
    # Import here so the module-level bot setup only runs after the event loop
    # is already running (aiogram 3.x requires this).
    from tg_bot import main as bot_main  # noqa: PLC0415

    await asyncio.gather(
        _health_server(),
        bot_main(),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    )
    # Pre-warm ChromaDB before the event loop starts so the first user query
    # already sees the fresh knowledge base.  Synchronous I/O here is fine —
    # this runs once at boot.
    _download_chroma_db()

    # Playwright spawns child processes via asyncio subprocess.
    # On Linux/Cloud Run (Python 3.10+), the default child watcher raises
    # NotImplementedError in non-main threads. ThreadedChildWatcher is safe
    # in all environments including multi-threaded asyncio.gather() contexts.
    if sys.platform != "win32":
        asyncio.set_child_watcher(asyncio.ThreadedChildWatcher())
    asyncio.run(_main())

