"""
FreedomConnector — live Tradernet/Freedom Broker API client.

Authentication uses HMAC-SHA256 signing:
  - Header ``X-Nt-Api-Key``:  Public API key
  - Header ``X-Nt-Api-Sig``:  HMAC-SHA256(secret_key, q).hexdigest()

The ``q`` parameter is a compact JSON string (no extra spaces) of the
command payload. The signature is computed over the raw ``q`` bytes only.

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

import pandas as pd
import requests

logger = logging.getLogger("BrokerAPI")

# Freedom Finance public API v2 endpoint (getPositionJson and all current commands
# live on v2; v1 at /api/ returns code 5 "Command not found" for these commands).
TRADERNET_URL   = os.getenv("TRADERNET_URL", "https://tradernet.kz/api/v2/")
REQUEST_TIMEOUT = 30   # seconds
DEMO_KEY        = "demo"

# Service-level Freedom Broker credentials (injected via Secret Manager).
# .strip() at source prevents hidden newlines from Secret Manager corrupting
# HMAC signatures or being passed into per-user vault comparisons.
FREEDOM_API_KEY    = os.getenv("FREEDOM_API_KEY", "").strip()
FREEDOM_API_SECRET = os.getenv("FREEDOM_API_SECRET", "").strip()

# Commands to try in priority order.
# getPositions is the confirmed working command for this account/API version.
# getPortfolioFull is tried as a second option; legacy names follow.
_PORTFOLIO_CMDS = ("getPositions", "getPortfolio", "getPortfolioFull", "getPositionJson")
_BALANCE_CMDS   = ("getBalance", "getClientInfo")

# API error substrings that indicate credential/auth rejection.
_AUTH_ERROR_PHRASES = (
    "invalid credentials",
    "unauthorized",
    "access denied",
    "forbidden",
    "bad credentials",
    "authentication",
)


class BrokerAuthError(RuntimeError):
    """Raised when Freedom Broker API rejects the request due to bad credentials."""


class FreedomConnector:
    def __init__(self, api_key: str = "", secret_key: str = ""):
        # Fall back to service-level env var credentials when not explicitly supplied.
        # .strip() removes hidden newlines/spaces injected by Secret Manager.
        self.api_key    = (api_key    or FREEDOM_API_KEY).strip()
        self.secret_key = (secret_key or FREEDOM_API_SECRET).strip()

    # ── HMAC-SHA256 request signing ──────────────────────────────────────────

    def _generate_signature(self, payload: dict) -> tuple[str, str]:
        """
        Tradernet signing: HMAC-SHA256(secret_key, q_bytes).
        q is compact JSON — separators=(',',':') guarantees no spaces.
        Returns (q_string, hex_digest).
        """
        q_str = json.dumps(payload, separators=(',', ':'))
        sig   = hmac.new(
            self.secret_key.encode('utf-8'),
            q_str.encode('utf-8'),
            hashlib.sha256,
        ).hexdigest()
        logger.debug("DEBUG: Final q_str used for signing: %r", q_str)
        return q_str, sig

    # ── Internal request helper ──────────────────────────────────────────────

    def _post(self, cmd: str, extra_params: dict | None = None) -> dict:
        """
        Send a signed command to the Tradernet API and return the parsed JSON.
        Raises ``BrokerAuthError`` on credential rejection.
        Raises ``RuntimeError`` on other API-level errors.
        """
        params = {}
        if extra_params:
            params.update(extra_params)

        # Official API spec requires "params":{} even when empty.
        cmd_payload = {"cmd": cmd, "params": params}
        q_str, sig  = self._generate_signature(cmd_payload)

        headers = {
            "Content-Type":  "application/x-www-form-urlencoded",
            "X-Nt-Api-Key":  self.api_key,
            "X-Nt-Api-Sig":  sig,
        }

        form_data = {"q": q_str}

        logger.info("POST %s  [cmd=%s]", TRADERNET_URL, cmd)
        logger.info(
            "SIGN key=%s… secret_len=%d secret_prefix=%s… q=%r",
            self.api_key[:6]    if self.api_key    else "EMPTY",
            len(self.secret_key),
            self.secret_key[:4] if self.secret_key else "EMPTY",
            q_str,
        )
        logger.info(
            "HEADERS Content-Type=%s X-Nt-Api-Key=%s… X-Nt-Api-Sig=…%s",
            headers.get("Content-Type", "MISSING"),
            headers.get("X-Nt-Api-Key", "MISSING")[:6],
            headers.get("X-Nt-Api-Sig", "MISSING")[-8:],
        )

        resp = requests.post(
            TRADERNET_URL,
            data=form_data,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()

        if "error" in raw:
            err_msg = str(raw["error"])
            logger.error(
                "Freedom API error [cmd=%s]: status=%s raw_body=%s",
                cmd, resp.status_code, resp.text[:500],
            )
            if any(p in err_msg.lower() for p in _AUTH_ERROR_PHRASES):
                raise BrokerAuthError(err_msg)
            raise RuntimeError(err_msg)

        return raw

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_portfolio(self) -> pd.DataFrame:
        """
        Fetch live positions from Freedom Broker Tradernet API.
        Raises ``BrokerAuthError`` on credential rejection — does NOT silently
        fall back to mock data in that case.
        Falls back to mock only for non-auth network/command errors.
        """
        if self.api_key == DEMO_KEY:
            logger.info("Демо-режим: используется шаблонный портфель.")
            return self._mock_portfolio()

        last_error: Exception | None = None

        for cmd in _PORTFOLIO_CMDS:
            try:
                raw = self._post(cmd)
                df  = self._parse(raw)
                logger.info("Команда '%s' выполнена успешно.", cmd)
                return df
            except BrokerAuthError:
                raise  # Never fall back to mock on credential rejection
            except RuntimeError as exc:
                logger.info("Команда '%s' не поддерживается (code 5), пробую следующую: %s", cmd, exc)
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
        Falls back to an empty dict on error (non-auth).
        Raises ``BrokerAuthError`` on credential rejection.
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
            except BrokerAuthError:
                raise
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
