"""
FreedomConnector — live Tradernet/Freedom Broker API client.

Authentication uses HMAC-SHA256 signing:
  - Header ``X-Nt-Api-Key``:       Public API key
  - Header ``X-Nt-Api-Timestamp``: Unix epoch seconds
  - Header ``X-Nt-Api-Sig``:       HMAC-SHA256(payload + timestamp, secret_key)

Demo mode: when api_key == 'demo', returns a hardcoded mock portfolio so
the MAC3 engine can run without real credentials.

Service-level credentials can be set via FREEDOM_API_KEY / FREEDOM_API_SECRET
environment variables and are used as a fallback when no per-user vault keys
are present.
"""

import hashlib
import hmac
import json
import logging
import os
import time

import pandas as pd
import requests

logger = logging.getLogger("BrokerAPI")

# Configurable via env var; defaults to the Kazakhstan regional endpoint.
# tradernet.com returns 403 for KZ accounts — use tradernet.kz instead.
TRADERNET_URL    = os.getenv("TRADERNET_URL", "https://tradernet.kz/api/")
REQUEST_TIMEOUT  = 30   # seconds
DEMO_KEY         = "demo"

# Service-level Freedom Broker credentials (injected via Secret Manager).
# Used as a fallback when no per-user vault keys are available.
FREEDOM_API_KEY    = os.getenv("FREEDOM_API_KEY", "")
FREEDOM_API_SECRET = os.getenv("FREEDOM_API_SECRET", "")

# Commands to try, in priority order.  The Freedom Broker / Tradernet API
# currently uses "getPortfolio" as the primary command; "getPositionJson"
# is retained as a fallback for older account types.
_PORTFOLIO_CMDS = ("getPortfolio", "getPositionJson")
_BALANCE_CMDS   = ("getBalance", "getClientInfo")


class FreedomConnector:
    def __init__(self, api_key: str = "", secret_key: str = ""):
        # Fall back to service-level env var credentials when not explicitly supplied.
        self.api_key    = api_key    or FREEDOM_API_KEY
        self.secret_key = secret_key or FREEDOM_API_SECRET

    # ── HMAC-SHA256 request signing ──────────────────────────────────────────

    @staticmethod
    def _sign(payload_str: str, timestamp: int, secret_key: str) -> str:
        """Tradernet signing: HMAC-SHA256(payload + str(timestamp), secret_key)"""
        message = (payload_str + str(timestamp)).encode("utf-8")
        return hmac.new(
            secret_key.encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()

    # ── Internal request helper ──────────────────────────────────────────────

    def _post(self, cmd: str, extra_params: dict | None = None) -> dict:
        """
        Send a signed command to the Tradernet API and return the parsed JSON.
        Raises ``RuntimeError`` on API-level errors.
        """
        params = {}
        if extra_params:
            params.update(extra_params)

        cmd_payload  = {"cmd": cmd, "params": params}
        payload_str  = json.dumps(cmd_payload, separators=(",", ":"), ensure_ascii=False)
        timestamp    = int(time.time())

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        if self.secret_key:
            sig = self._sign(payload_str, timestamp, self.secret_key)
            headers.update({
                "X-Nt-Api-Key":       self.api_key,
                "X-Nt-Api-Timestamp": str(timestamp),
                "X-Nt-Api-Sig":       sig,
            })

        form_data = {"q": payload_str}

        logger.info("POST %s  [cmd=%s]", TRADERNET_URL, cmd)
        logger.debug("Request form_data: %s", form_data)
        logger.debug("Headers: X-Nt-Api-Key=%s, ts=%s", self.api_key, timestamp)

        resp = requests.post(
            TRADERNET_URL,
            data=form_data,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()

        if "error" in raw:
            raise RuntimeError(raw["error"])

        return raw

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_portfolio(self) -> pd.DataFrame:
        """
        Fetch live positions from Freedom Broker Tradernet API.
        Falls back to a mock portfolio when running in demo mode or when
        the broker is unreachable.
        """
        if self.api_key == DEMO_KEY:
            logger.info("Демо-режим: используется шаблонный портфель.")
            return self._mock_portfolio()

        last_error: Exception | None = None

        for cmd in _PORTFOLIO_CMDS:
            try:
                raw = self._post(cmd)
                return self._parse(raw)
            except RuntimeError as exc:
                logger.warning("Команда '%s' не поддерживается: %s", cmd, exc)
                last_error = exc
            except requests.RequestException as exc:
                logger.error("Freedom API недоступен [cmd=%s]: %s", cmd, exc)
                last_error = exc

        logger.error(
            "Все команды API не сработали (последняя ошибка: %s) — "
            "переключаюсь на mock-портфель.",
            last_error,
        )
        return self._mock_portfolio()

    def fetch_balance(self) -> dict:
        """
        Fetch account cash balance from Freedom Broker Tradernet API.
        Returns a dict with ``available``, ``blocked``, and ``currency`` keys.
        Falls back to an empty dict in demo mode or on API error.
        """
        if self.api_key == DEMO_KEY:
            logger.info("Демо-режим: возвращается фиктивный баланс.")
            return {"currency": "USD", "available": 10_000.0, "blocked": 0.0}

        last_error: Exception | None = None

        for cmd in _BALANCE_CMDS:
            try:
                raw  = self._post(cmd)
                data = raw.get("result", raw)
                return {
                    "currency":  data.get("currency", "USD"),
                    "available": float(data.get("equity", data.get("available", 0))),
                    "blocked":   float(data.get("blocked", 0)),
                }
            except RuntimeError as exc:
                logger.warning("Команда баланса '%s' не поддерживается: %s", cmd, exc)
                last_error = exc
            except (requests.RequestException, ValueError) as exc:
                logger.error("Ошибка получения баланса [cmd=%s]: %s", cmd, exc)
                last_error = exc

        logger.error("Не удалось получить баланс: %s", last_error)
        return {}

    # ── Parsers ───────────────────────────────────────────────────────────────

    def _parse(self, raw_json: dict) -> pd.DataFrame:
        """Map Tradernet portfolio JSON → MAC3 DataFrame (Ticker / Quantity / Purchase_Price)."""
        data_root = raw_json.get("result", raw_json)
        items = (
            data_root.get("pos")
            or data_root.get("portfolio")
            or data_root.get("positions")
            or []
        )

        positions = []
        for item in items:
            ticker = (
                item.get("i")
                or item.get("t")
                or item.get("ticker")
                or item.get("symbol")
            )
            qty = item.get("q") or item.get("quantity") or 0
            price = (
                item.get("price_a")
                or item.get("avg_price")
                or item.get("mkt_p")
                or item.get("price")
                or 0
            )
            if not ticker or float(qty) == 0:
                continue
            positions.append({
                "Ticker":         str(ticker).upper().strip(),
                "Quantity":       float(qty),
                "Purchase_Price": float(price),
            })

        if not positions:
            logger.warning("Freedom API вернул пустой портфель, переключаюсь на mock.")
            return self._mock_portfolio()

        df = pd.DataFrame(positions)
        logger.info("Загружено %d позиций из Freedom Broker.", len(df))
        return df

    @staticmethod
    def _mock_portfolio() -> pd.DataFrame:
        """Deterministic template portfolio for demo / offline mode."""
        return pd.DataFrame([
            {"Ticker": "AAPL",    "Quantity": 10,  "Purchase_Price": 150.0},
            {"Ticker": "KSPI",    "Quantity": 100, "Purchase_Price": 12.5},
            {"Ticker": "BTC-USD", "Quantity": 0.5, "Purchase_Price": 45_000.0},
        ])
