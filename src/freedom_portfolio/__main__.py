"""
CLI entry-point for the freedom_portfolio package.

Usage:
    python -m freedom_portfolio              # one-shot REST pull, rich table
    python -m freedom_portfolio --json       # raw JSON output
    python -m freedom_portfolio --live       # WebSocket live feed
    python -m freedom_portfolio --demo       # use demo / beta server (WS only)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from freedom_portfolio.client import (
    AuthenticationError,
    BrokerAPIError,
    InvalidSignatureError,
    TradernetClient,
)
from freedom_portfolio.models import Portfolio


def _load_env() -> tuple[str, str]:
    """Load credentials from .env (if python-dotenv is present) and validate."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
        load_dotenv()
    except ImportError:
        pass

    public_key = (os.getenv("TRADERNET_PUBLIC_KEY") or os.getenv("FREEDOM_API_KEY") or "").strip()
    secret_key = (os.getenv("TRADERNET_SECRET_KEY") or os.getenv("FREEDOM_API_SECRET") or "").strip()

    if not public_key:
        sys.stderr.write(
            "ERROR: TRADERNET_PUBLIC_KEY (or FREEDOM_API_KEY) is not set.\n"
            "Copy config/.env.example to .env and fill in your keys.\n"
        )
        sys.exit(2)

    return public_key, secret_key


def _output_json(portfolio: Portfolio) -> None:
    print(portfolio.model_dump_json(indent=2))


def _output_table(portfolio: Portfolio) -> None:
    try:
        from freedom_portfolio.display import print_portfolio
    except ImportError:
        sys.stderr.write("rich is not installed — falling back to JSON output.\n")
        _output_json(portfolio)
        return
    print_portfolio(portfolio)


async def _run_live(public_key: str, secret_key: str, *, demo: bool, json_output: bool) -> None:
    from freedom_portfolio.websocket import TradernetWebSocket
    ws = TradernetWebSocket(public_key, secret_key, demo=demo)

    async def on_portfolio(p: Portfolio) -> None:
        if json_output:
            _output_json(p)
        else:
            _output_table(p)

    async def on_error(exc: Exception) -> None:
        sys.stderr.write(f"WebSocket error: {exc}\n")

    await ws.connect_and_subscribe(on_portfolio, on_error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freedom Broker (Tradernet) portfolio viewer")
    parser.add_argument("--live", action="store_true", help="Subscribe to live WebSocket feed")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--demo", action="store_true", help="Use demo/beta WebSocket server")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    public_key, secret_key = _load_env()

    if args.live:
        try:
            asyncio.run(_run_live(public_key, secret_key, demo=args.demo, json_output=args.json))
        except KeyboardInterrupt:
            return 0
        return 0

    client = TradernetClient(public_key, secret_key)
    try:
        portfolio = client.get_portfolio()
    except InvalidSignatureError as exc:
        sys.stderr.write(f"ERROR: invalid signature — check TRADERNET_SECRET_KEY: {exc}\n")
        return 3
    except AuthenticationError as exc:
        sys.stderr.write(f"ERROR: authentication failed — check TRADERNET_PUBLIC_KEY: {exc}\n")
        return 4
    except BrokerAPIError as exc:
        sys.stderr.write(f"ERROR: Tradernet API failure: {exc}\n")
        return 5

    if args.json:
        _output_json(portfolio)
    else:
        _output_table(portfolio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
