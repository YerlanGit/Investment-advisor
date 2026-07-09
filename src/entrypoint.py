"""
Cloud Run entrypoint — runs the Telegram bot alongside a minimal HTTP health
server so Cloud Run's startup probe succeeds.

Cloud Run requires a container to bind on $PORT (default 8080). The bot uses
long-polling, not webhooks, so this thin stdlib-only server handles the probe.

On startup, the entrypoint also pre-warms the on-disk ChromaDB by syncing
the latest snapshot from gs://ramp-bot-chroma-db-investadv/chroma_db/. The bucket is
populated by the Cloud Function `cloud_function/main.py` whenever a new
bank-analytic PDF is uploaded — so each container restart picks up the
freshest knowledge base without a redeploy.
"""

import asyncio
import logging
import os
import sys
import threading

# yfinance был удалён 2026-04. Цены теперь берутся через Tradernet
# (freedom_portfolio.history); кеш истории живёт в /tmp/freedom_history_cache.

logger = logging.getLogger("ramp.entrypoint")

# ── ChromaDB sync from GCS ──────────────────────────────────────────────────
CHROMA_LOCAL_PATH  = os.environ.get("CHROMA_LOCAL_PATH",  "/app/data/chroma_db")
CHROMA_GCS_BUCKET  = os.environ.get("CHROMA_BUCKET",      "ramp-bot-chroma-db-investadv")
CHROMA_GCS_PREFIX  = os.environ.get("CHROMA_GCS_PREFIX",  "chroma_db/")

# ── RAG boot-ingest fallback (2026-07-04) ────────────────────────────────────
# The Cloud-Function → Eventarc → ChromaDB path is fragile in practice: the
# trigger's region must match the bucket's region (europe-west3), the function
# has to download the embedding model at runtime (the BOT image PRE-BAKES it),
# and a fail-soft cloudbuild step can silently skip the deploy.  When any of
# that breaks, the STORE bucket stays empty and RAG never lights up even though
# PDFs are sitting in the INBOX.  This fallback makes the bot self-sufficient:
# if the synced ChromaDB is empty but the INBOX holds PDFs, ingest them
# IN-CONTAINER (deps + pre-baked model + pinned chromadb already present) and
# publish the built store back to STORE.  Off switch: RAG_BOOT_INGEST=0.
RAG_INBOX_BUCKET = os.environ.get("RAG_INBOX_BUCKET", "ramp-bot-chroma-db-inbox-investadv")


def _rag_boot_ingest_enabled() -> bool:
    return str(os.getenv("RAG_BOOT_INGEST", "on")).strip().lower() not in (
        "0", "false", "no", "off", "")


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


def _upload_chroma_db_to_store() -> int:
    """Mirror the local ChromaDB dir → gs://CHROMA_BUCKET/CHROMA_GCS_PREFIX so a
    store built in-container by the boot-ingest fallback persists (future boots
    just sync it, and any sibling revision sees it).  Best-effort; returns the
    number of blobs uploaded (0 on any failure)."""
    try:
        from google.cloud import storage  # noqa: PLC0415
        client = storage.Client()
        bucket = client.bucket(CHROMA_GCS_BUCKET)
        uploaded = 0
        for root, _, files in os.walk(CHROMA_LOCAL_PATH):
            for fname in files:
                local_file = os.path.join(root, fname)
                rel = os.path.relpath(local_file, CHROMA_LOCAL_PATH).replace("\\", "/")
                bucket.blob(f"{CHROMA_GCS_PREFIX}{rel}").upload_from_filename(local_file)
                uploaded += 1
        logger.info("RAG boot-ingest: published %d ChromaDB blob(s) → gs://%s/%s.",
                    uploaded, CHROMA_GCS_BUCKET, CHROMA_GCS_PREFIX)
        return uploaded
    except Exception as exc:
        logger.warning("RAG boot-ingest: publish to STORE skipped (%s).", exc)
        return 0


def _boot_ingest_from_inbox() -> None:
    """Fallback: ingest INBOX PDFs that are NOT yet in the ChromaDB store, then
    publish the store back to STORE — so RAG works even if the Cloud Function /
    Eventarc path is misconfigured (wrong region, missing trigger, embedding-
    model download failure).  Runs on a daemon thread so it never blocks the
    health probe or the bot event loop; every failure is swallowed.

    Self-heals BOTH cases (замечание #4 — «база не обновилась после 8 новых
    отчётов»):
      • empty store        → ingest every INBOX PDF (original behaviour);
      • non-empty store     → ingest only the INBOX PDFs whose filename is not
        already an ingested `source`, so incrementally-added reports are picked
        up on the next boot even when the Cloud Function silently failed.
    """
    if not _rag_boot_ingest_enabled():
        return
    try:
        from agent.rag_engine import FinancialRAG  # noqa: PLC0415
        rag = FinancialRAG(db_path=CHROMA_LOCAL_PATH)
        already = int(rag.collection.count())

        from google.cloud import storage  # noqa: PLC0415
        client = storage.Client()
        pdfs = [b for b in client.bucket(RAG_INBOX_BUCKET).list_blobs()
                if b.name.lower().endswith(".pdf")]
        if not pdfs:
            logger.info("RAG boot-ingest: INBOX gs://%s has no PDFs — RAG stays %s.",
                        RAG_INBOX_BUCKET, "empty" if already == 0 else "as-is")
            return

        # Which INBOX PDFs are missing from the store?  Compare by normalised
        # basename against the already-ingested `source` filenames.
        def _norm(name) -> str:
            return os.path.basename(str(name or "")).strip().lower()
        ingested: set[str] = set()
        if already > 0:
            try:
                ingested = {_norm(d.get("source")) for d in rag.list_documents()}
            except Exception:
                ingested = set()
        missing = [b for b in pdfs if _norm(os.path.basename(b.name)) not in ingested]
        if not missing:
            logger.info("RAG boot-ingest: store has %d chunks · all %d INBOX PDF(s) "
                        "already ingested — skip.", already, len(pdfs))
            return
        logger.info("RAG boot-ingest: store has %d chunks · %d of %d INBOX PDF(s) "
                    "missing — ingesting in-container.", already, len(missing), len(pdfs))

        import tempfile  # noqa: PLC0415
        total = 0
        for blob in missing:
            fname = os.path.basename(blob.name)
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp_path = tmp.name
                    blob.download_to_filename(tmp_path)
                n = rag.ingest_pdf(tmp_path, doc_metadata={"filename": fname})
                total += n
                logger.info("RAG boot-ingest: %s → %d chunks.", fname, n)
            except Exception as exc:
                logger.warning("RAG boot-ingest: %s failed (%s) — skipped.", fname, exc)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        logger.info("RAG boot-ingest: ingested %d chunks from %d new PDF(s).",
                    total, len(missing))
        if total > 0:
            _upload_chroma_db_to_store()
    except Exception as exc:
        logger.warning("RAG boot-ingest skipped (%s).", exc)


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

    # Reliable RAG fallback: if the synced store is empty but PDFs sit in the
    # INBOX, ingest them in-container on a DAEMON thread (never blocks the
    # startup probe or the bot).  Makes RAG work without a healthy Cloud
    # Function / Eventarc trigger.  Off switch: RAG_BOOT_INGEST=0.
    threading.Thread(target=_boot_ingest_from_inbox,
                     name="rag-boot-ingest", daemon=True).start()

    # Playwright spawns child processes via asyncio subprocess.
    # On Linux/Cloud Run (Python 3.10+), the default child watcher raises
    # NotImplementedError in non-main threads. ThreadedChildWatcher is safe
    # in all environments including multi-threaded asyncio.gather() contexts.
    if sys.platform != "win32":
        asyncio.set_child_watcher(asyncio.ThreadedChildWatcher())
    asyncio.run(_main())

