"""
Cloud Run entrypoint — runs the Telegram bot alongside a minimal HTTP health
server so Cloud Run's startup probe succeeds.

Cloud Run requires a container to bind on $PORT (default 8080). The bot uses
long-polling, not webhooks, so this thin stdlib-only server handles the probe.
"""

import asyncio
import logging
import os
import sys

# Fix yfinance cache location for Cloud Run (read-only /root/.cache)
os.makedirs("/tmp/yf_cache", exist_ok=True)
try:
    import yfinance as yf
    yf.set_tz_cache_location("/tmp/yf_cache")
except Exception:
    pass  # yfinance not yet installed or import order issue — handled later

logger = logging.getLogger("ramp.entrypoint")


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
    # Playwright spawns child processes via asyncio subprocess.
    # On Linux/Cloud Run (Python 3.10+), the default child watcher raises
    # NotImplementedError in non-main threads. ThreadedChildWatcher is safe
    # in all environments including multi-threaded asyncio.gather() contexts.
    if sys.platform != "win32":
        asyncio.set_child_watcher(asyncio.ThreadedChildWatcher())
    asyncio.run(_main())
