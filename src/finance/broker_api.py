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

# Known stock-exchange suffixes that Freedom Finance appends to tickers.
# "AAPL.US" → "AAPL", "KSPI.KZ" → "KSPI", "KAP.IL" → "KAP".
# Compound tickers like "BRK.B" are not affected because "B" is not in this set.
_EXCHANGE_SUFFIXES = frozenset({
    "US", "KZ", "ME", "LN", "GR", "PA", "HK", "TO", "AX",
    "DE", "JP", "CN", "SG", "IL", "SW", "AS", "MI",
})


def _coerce_float(value) -> "float | None":
    """Convert *value* to float; return None if absent or unconvertible."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

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
# getPositionJson is the confirmed working command — it is a classic Tradernet v1 command
# that is still supported on the v2 endpoint and uses the standard q-JSON signing format.
# The pure v2 commands (getPositions, getPortfolio, getPortfolioFull) are kept as fallbacks
# for accounts on newer API tiers, but they return code 5 "Command not found" on most accounts.
_PORTFOLIO_CMDS = ("getPositionJson", "getPositions", "getPortfolio", "getPortfolioFull")
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


class BrokerEmptyPortfolioError(RuntimeError):
    """Raised when Freedom Broker returns an authenticated account with no open positions."""


class FreedomConnector:
    def __init__(self, api_key: str = "", secret_key: str = ""):
        # Fall back to service-level env var credentials when not explicitly supplied.
        # .strip() removes hidden newlines/spaces injected by Secret Manager.
        self.api_key    = (api_key    or FREEDOM_API_KEY).strip()
        self.secret_key = (secret_key or FREEDOM_API_SECRET).strip()

    # ── Internal request helper ──────────────────────────────────────────────

    def _post(self, cmd: str, extra_params: dict | None = None) -> dict:
        """
        Send a signed command to the Freedom Finance / Tradernet API and return parsed JSON.

        Signing format (standard Tradernet v1/v2):
          q       = compact JSON string {"cmd": <cmd>, "params": <params>}
          sig     = HMAC-SHA256(secret_key, q).hexdigest()
          POST body fields: cmd, q, apiKey, sig
        """
        params = extra_params or {}

        q_payload = json.dumps({"cmd": cmd, "params": params}, separators=(',', ':'))
        sig = hmac.new(
            self.secret_key.encode('utf-8'),
            q_payload.encode('utf-8'),
            hashlib.sha256,
        ).hexdigest()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # apiKey and sig MUST be POST form fields — the server looks up the user's
        # secret by apiKey from the body; passing them only in headers yields code 12.
        form_data = {
            "cmd":    cmd,
            "q":      q_payload,
            "apiKey": self.api_key,
            "sig":    sig,
        }

        logger.info("POST %s  [cmd=%s]", TRADERNET_URL, cmd)
        logger.info(
            "SIGN key=%s… secret_len=%d secret_prefix=%s… q=%r",
            self.api_key[:6]    if self.api_key    else "EMPTY",
            len(self.secret_key),
            self.secret_key[:4] if self.secret_key else "EMPTY",
            q_payload,
        )

        resp = requests.post(
            TRADERNET_URL,
            data=form_data,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()

        # Freedom Finance API signals errors via "errMsg" + "code", not "error".
        # Also handle legacy "error" key for defensive coverage.
        err_msg  = raw.get("errMsg") or raw.get("error") or raw.get("err")
        err_code = raw.get("code")
        if err_msg or (isinstance(err_code, int) and err_code != 0):
            err_text = str(err_msg) if err_msg else f"API error code {err_code}"
            logger.error(
                "Freedom API error [cmd=%s]: %s (code=%s) raw=%s",
                cmd, err_text, err_code, resp.text[:500],
            )
            if any(p in err_text.lower() for p in _AUTH_ERROR_PHRASES):
                raise BrokerAuthError(err_text)
            raise RuntimeError(err_text)

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
            except BrokerEmptyPortfolioError:
                raise  # Real account with no positions — propagate, do not fall back to mock
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
            # Freedom API canonical ticker field is "i"; others are legacy/alternate commands.
            ticker_raw = (
                item.get("i")
                or item.get("t")
                or item.get("ticker")
                or item.get("symbol")
            )
            if not ticker_raw:
                continue

            ticker = str(ticker_raw).upper().strip()
            # Strip exchange suffix: "AAPL.US" → "AAPL", "KSPI.KZ" → "KSPI".
            # rpartition on the rightmost "." so "BRK.B" stays intact when "B"
            # is not a known exchange code.
            if "." in ticker:
                base, _, suffix = ticker.rpartition(".")
                if suffix in _EXCHANGE_SUFFIXES:
                    ticker = base

            # Quantity — "q" is the canonical Freedom field.
            qty = _coerce_float(item.get("q"))
            if qty is None:
                qty = _coerce_float(item.get("quantity"))
            if not qty:  # None or 0 — skip zero-quantity rows
                continue

            # Purchase price — "price_a" / "bal_price_a" = entry price (preferred).
            # Falls back to market price fields when no entry price is present.
            price = next(
                (
                    f
                    for key in ("price_a", "bal_price_a", "avg_price", "mkt_p", "mkt_price", "price")
                    if (f := _coerce_float(item.get(key))) is not None
                ),
                0.0,
            )

            # Broker-provided current market price — used as fallback for instruments
            # (e.g. KZ bonds with ISIN tickers) that have no yfinance history.
            broker_current = next(
                (
                    f
                    for key in ("mkt_price", "close_price")
                    if (f := _coerce_float(item.get(key))) is not None and f > 0
                ),
                None,
            )

            positions.append({
                "Ticker":               ticker,
                "Quantity":             qty,
                "Purchase_Price":       price,
                "Broker_Current_Price": broker_current,
            })

        # Cash balances from "acc" section (Freedom API).
        # "s" = свободные средства (available cash) in the account currency.
        # Each currency account is added as a separate cash position.
        for acc in data_root.get("acc", []):
            curr = str(acc.get("curr", "")).upper().strip()
            if not curr:
                continue
            cash_qty = _coerce_float(acc.get("s"))
            if not cash_qty:  # None or 0 — skip empty/zero balances
                continue
            positions.append({
                "Ticker":               curr,      # "USD", "EUR", "KZT", etc.
                "Quantity":             cash_qty,
                "Purchase_Price":       1.0,
                "Broker_Current_Price": 1.0,
            })
            logger.info("Добавлен кэш: %s = %.2f", curr, cash_qty)

        if not positions:
            raise BrokerEmptyPortfolioError(
                "Брокерский счёт не содержит открытых позиций и свободных средств. "
                "Откройте позиции в Freedom Broker или переключитесь на демо-режим (/start)."
            )

        df = pd.DataFrame(positions)
        logger.info("Загружено %d позиций из Freedom Broker (включая кэш).", len(df))
        return df

    @staticmethod
    def _mock_portfolio() -> pd.DataFrame:
        """Deterministic template portfolio for demo / offline mode."""
        return pd.DataFrame([
            {"Ticker": "AAPL",    "Quantity": 10,  "Purchase_Price": 150.0},
            {"Ticker": "KSPI",    "Quantity": 100, "Purchase_Price": 12.5},
            {"Ticker": "BTC-USD", "Quantity": 0.5, "Purchase_Price": 45_000.0},
        ])
