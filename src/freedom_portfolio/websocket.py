"""
Tradernet live WebSocket subscription.

This module is OPTIONAL — it only loads when ``python-socketio[client]`` is
installed.  The Telegram bot does not import it; it exists for the standalone
CLI (``python -m freedom_portfolio --live``) and for downstream consumers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from freedom_portfolio.auth import build_signature
from freedom_portfolio.models import Portfolio

logger = logging.getLogger(__name__)

WS_URL_LIVE = "wss://ws.tradernet.com"
WS_URL_DEMO = "wss://wsbeta.tradernet.com"

PortfolioCallback = Callable[[Portfolio], Awaitable[None] | None]
ErrorCallback     = Callable[[Exception], Awaitable[None] | None]


class TradernetWebSocket:
    """
    Async Socket.IO subscription to Tradernet live portfolio updates.

    Auth flow (different from REST):
        sio.emit("auth", auth_data, sig, callback=on_auth)
    where ``auth_data`` carries apiKey/cmd/nonce and ``sig`` is the md5
    signature.  After successful auth, emit ``notifyPortfolio`` and listen
    for ``portfolio`` events.
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        *,
        demo: bool = False,
        max_reconnects: int = 5,
    ) -> None:
        self.public_key     = public_key.strip()
        self.secret_key     = secret_key.strip()
        self.url            = WS_URL_DEMO if demo else WS_URL_LIVE
        self.max_reconnects = max_reconnects

    async def connect_and_subscribe(
        self,
        on_portfolio: PortfolioCallback,
        on_error: ErrorCallback | None = None,
    ) -> None:
        """
        Connect, authenticate, and stream portfolio events to *on_portfolio*.

        Reconnects with exponential backoff (1s, 2s, 4s, 8s, 16s) up to
        ``max_reconnects`` times before giving up.
        """
        try:
            import socketio  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — optional dep
            raise RuntimeError(
                "python-socketio[client] is required for TradernetWebSocket. "
                "Install with: pip install 'python-socketio[client]>=5.10'"
            ) from exc

        attempt = 0
        while attempt <= self.max_reconnects:
            sio = socketio.AsyncClient(reconnection=False, logger=False, engineio_logger=False)
            self._wire_handlers(sio, on_portfolio, on_error)
            try:
                await sio.connect(self.url, transports=["websocket"])
                await self._authenticate(sio)
                await sio.emit("notifyPortfolio")
                await sio.wait()
                return
            except Exception as exc:
                logger.error("WebSocket session failed (attempt %d/%d): %s",
                             attempt + 1, self.max_reconnects + 1, exc)
                if on_error is not None:
                    await _maybe_await(on_error(exc))
                attempt += 1
                if attempt > self.max_reconnects:
                    raise
                backoff = min(2 ** attempt, 30)
                logger.info("Reconnecting in %ds…", backoff)
                await asyncio.sleep(backoff)
            finally:
                if sio.connected:
                    await sio.disconnect()

    # ── Internals ────────────────────────────────────────────────────────────

    def _wire_handlers(
        self,
        sio,
        on_portfolio: PortfolioCallback,
        on_error: ErrorCallback | None,
    ) -> None:
        @sio.on("portfolio")
        async def _on_portfolio(data):  # noqa: ANN001 — socketio handler
            try:
                payload = data[0] if isinstance(data, list) and data else data
                ps = (payload or {}).get("ps", payload)
                await _maybe_await(on_portfolio(Portfolio(**ps)))
            except Exception as exc:
                logger.exception("Failed to parse portfolio event")
                if on_error is not None:
                    await _maybe_await(on_error(exc))

        @sio.on("disconnect")
        async def _on_disconnect():
            logger.info("Tradernet WebSocket disconnected")

    async def _authenticate(self, sio) -> None:
        nonce = str(int(time.time() * 1000))
        auth_data = {
            "apiKey": self.public_key,
            "cmd":    "getAuthInfo",
            "nonce":  nonce,
        }
        sig = build_signature(auth_data, self.secret_key)
        result_future: asyncio.Future = asyncio.get_event_loop().create_future()

        def _ack(*payload):
            if not result_future.done():
                result_future.set_result(payload)

        await sio.emit("auth", auth_data, sig, callback=_ack)
        try:
            payload = await asyncio.wait_for(result_future, timeout=10)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Tradernet WebSocket auth timeout (no callback)") from exc
        logger.info("Tradernet WebSocket authenticated: %r", payload)


async def _maybe_await(value):
    """Await *value* if it is a coroutine, otherwise return it as-is."""
    if asyncio.iscoroutine(value):
        return await value
    return value
