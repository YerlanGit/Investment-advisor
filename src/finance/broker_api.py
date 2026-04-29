"""
FreedomConnector — adapter between the read-only ``freedom_portfolio`` package
and the Telegram bot / MAC3 engine which expect a ``pandas.DataFrame`` shaped
``[Ticker, Quantity, Purchase_Price, Broker_Current_Price]`` plus the legacy
``df.attrs['_ramp_*']`` source markers.

This module preserves the exact public surface used by the rest of the project
(``FreedomConnector``, ``BrokerAuthError``, ``BrokerEmptyPortfolioError``,
``RealPortfolioRequired``) while delegating all wire-protocol work to
``freedom_portfolio``.

Wire protocol (per official Tradernet docs https://github.com/Tradernet/tn.api):
    URL       https://tradernet.com/api/   (international endpoint)
    Body      JSON {apiKey, cmd, nonce, params, sig}
    Signing   sig = md5(sorted("k=v"…concat) + secret_key)        # NOT HMAC
    Fallback  unsigned q={"cmd":…, "params":{"apiKey":…}} for accounts that
              do not have a separate secret (single Public API key).
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from freedom_portfolio.client import (
    AuthenticationError,
    BrokerAPIError,
    EmptyPortfolioError,
    InvalidSignatureError,
    TradernetClient,
)
from freedom_portfolio.models import Portfolio

logger = logging.getLogger("BrokerAPI")

# Known stock-exchange suffixes that Freedom Finance appends to tickers.
# "AAPL.US" → "AAPL", "KSPI.KZ" → "KSPI", "KAP.IL" → "KAP".
# Compound tickers like "BRK.B" are not affected because "B" is not in this set.
_EXCHANGE_SUFFIXES = frozenset({
    "US", "KZ", "ME", "LN", "GR", "PA", "HK", "TO", "AX",
    "DE", "JP", "CN", "SG", "IL", "SW", "AS", "MI", "AIX",
})

DEMO_KEY = "demo"

# Service-level Freedom Broker credentials (injected via Secret Manager).
# .strip() at source prevents hidden newlines from corrupting signatures.
FREEDOM_API_KEY    = os.getenv("FREEDOM_API_KEY",    "").strip()
FREEDOM_API_SECRET = os.getenv("FREEDOM_API_SECRET", "").strip()
FREEDOM_LOGIN      = os.getenv("FREEDOM_LOGIN",      "").strip()

# Optional override (helpful for tests/staging).  Spec mandates tradernet.com.
TRADERNET_URL = os.getenv("TRADERNET_URL", "https://tradernet.com/api/")


# ── Backwards-compatible exception aliases ───────────────────────────────────
# Existing code (tg_bot.py, advisor_bot.py, investment_logic.py) imports these
# names directly — keep the symbol stable while remapping to the new hierarchy.

class BrokerAuthError(AuthenticationError):
    """Raised when Freedom Broker API rejects the request due to bad credentials."""


class BrokerEmptyPortfolioError(EmptyPortfolioError):
    """Raised when Freedom Broker returns an authenticated account with no open positions."""


class RealPortfolioRequired(RuntimeError):
    """
    Raised by the MAC3 engine gate when the portfolio DataFrame is a
    fallback-mock produced by broker API failure.
    Distinct from intentional demo mode (api_key == 'demo').
    """


# ── Helpers ──────────────────────────────────────────────────────────────────


def _coerce_float(value) -> float | None:
    """Convert *value* to float; return None if absent or unconvertible."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _strip_exchange_suffix(ticker: str) -> str:
    """``AAPL.US`` → ``AAPL``; leaves compound tickers like ``BRK.B`` intact."""
    if "." not in ticker:
        return ticker
    base, _, suffix = ticker.rpartition(".")
    if suffix in _EXCHANGE_SUFFIXES:
        return base
    return ticker


# ── FreedomConnector ─────────────────────────────────────────────────────────


class FreedomConnector:
    """
    Bot-facing wrapper around ``freedom_portfolio.TradernetClient``.

    Two wire phases tried in order — first success wins:
        1. Signed REST    (md5 sig, when api_key + secret_key are both set)
        2. Unsigned REST  (q=JSON form, when only api_key is available)

    Phase 3 session login (login + password → SID) was removed in 2026-04 —
    spec forbids it, and it is not needed for read-only API-key access.
    """

    def __init__(self, api_key: str = "", secret_key: str = "", login: str = "") -> None:
        # Fall back to service-level env vars when not explicitly supplied.
        # The ``login`` argument is preserved for ABI compat but no longer used.
        self.api_key    = (api_key    or FREEDOM_API_KEY).strip()
        self.secret_key = (secret_key or FREEDOM_API_SECRET).strip()
        self.login      = (login      or FREEDOM_LOGIN).strip()

    # ── Public surface ───────────────────────────────────────────────────────

    def fetch_portfolio(self) -> pd.DataFrame:
        """
        Fetch live positions from Tradernet and convert to the MAC3 DataFrame.

        Returns a DataFrame with columns
            Ticker | Quantity | Purchase_Price | Broker_Current_Price
        plus ``df.attrs`` markers consumed by ``investment_logic.analyze_all``:
            _ramp_source='demo'        — intentional demo mode
            _ramp_is_mock=True         — fallback mock (always set on mock)
            _ramp_is_fallback=True     — broker call failed; MAC3 must halt

        Raises
        ------
        BrokerAuthError
            All transport phases reported credential rejection.
        BrokerEmptyPortfolioError
            Authenticated successfully but no positions and no cash.
        """
        if self.api_key == DEMO_KEY:
            logger.info("Демо-режим: используется шаблонный портфель.")
            df = self._mock_portfolio()
            df.attrs["_ramp_source"] = "demo"
            return df

        if not self.api_key:
            raise BrokerAuthError(
                "Freedom Broker API key is empty. Configure FREEDOM_API_KEY or "
                "register your key via /start → 🔗 Freedom Broker API."
            )

        client = TradernetClient(self.api_key, self.secret_key, base_url=TRADERNET_URL)
        try:
            portfolio = client.get_portfolio()
        except (InvalidSignatureError, AuthenticationError) as exc:
            # Re-raise as the legacy alias the bot already catches.
            raise BrokerAuthError(str(exc)) from exc
        except BrokerAPIError as exc:
            logger.error("Tradernet API failure: %s — returning fallback mock.", exc)
            df = self._mock_portfolio()
            df.attrs["_ramp_is_fallback"] = True
            return df
        except Exception as exc:
            # Catches pydantic.ValidationError if the API response is malformed.
            logger.error("Unexpected error parsing Tradernet response: %s — returning fallback mock.", exc)
            df = self._mock_portfolio()
            df.attrs["_ramp_is_fallback"] = True
            return df

        df = self._to_dataframe(portfolio)
        if df.empty:
            raise BrokerEmptyPortfolioError(
                "Брокерский счёт не содержит открытых позиций и свободных средств. "
                "Откройте позиции в Freedom Broker или переключитесь на демо-режим (/start)."
            )
        logger.info("Загружено %d позиций из Freedom Broker (включая кэш).", len(df))
        return df

    def fetch_balance(self) -> dict:
        """
        Fetch account cash balance.  Falls back to {} on non-auth errors.
        Returns ``{currency, available, blocked}``.
        """
        if self.api_key == DEMO_KEY:
            logger.info("Демо-режим: возвращается фиктивный баланс.")
            return {"currency": "USD", "available": 10_000.0, "blocked": 0.0}

        if not self.api_key:
            return {}

        client = TradernetClient(self.api_key, self.secret_key, base_url=TRADERNET_URL)
        try:
            portfolio = client.get_portfolio()
        except (InvalidSignatureError, AuthenticationError) as exc:
            raise BrokerAuthError(str(exc)) from exc
        except BrokerAPIError as exc:
            logger.error("Не удалось получить баланс: %s", exc)
            return {}

        # Prefer USD, otherwise fall back to the first non-empty account line.
        usd = next((a for a in portfolio.acc if a.curr.upper() == "USD"), None)
        primary = usd or (portfolio.acc[0] if portfolio.acc else None)
        if primary is None:
            return {}
        return {
            "currency":  primary.curr or "USD",
            "available": float(primary.s),
            "blocked":   0.0,
        }

    # ── DataFrame conversion ─────────────────────────────────────────────────

    @staticmethod
    def _to_dataframe(portfolio: Portfolio) -> pd.DataFrame:
        """Convert a Pydantic Portfolio to the MAC3 DataFrame contract."""
        rows: list[dict] = []

        for p in portfolio.pos:
            qty = _coerce_float(p.q) or 0.0
            if qty == 0.0:
                continue

            ticker = _strip_exchange_suffix(p.i.upper().strip())

            # Purchase price priority: explicit entry-price fields → cost basis
            # split → market price → 0.0.  open_bal / q is the cost basis when
            # no avg-price field is exposed by the API.
            purchase_price = (
                _coerce_float(p.price_a)
                or _coerce_float(p.bal_price_a)
                or (p.open_bal / qty if p.open_bal and qty else None)
                or _coerce_float(p.mkt_price)
                or 0.0
            )

            broker_current = _coerce_float(p.mkt_price)
            if broker_current is not None and broker_current <= 0:
                broker_current = None

            rows.append({
                "Ticker":               ticker,
                "Quantity":             qty,
                "Purchase_Price":       purchase_price,
                "Broker_Current_Price": broker_current,
            })

        # Cash positions from the ``acc`` array.  Each currency line becomes a
        # synthetic ticker (e.g. "USD") with quantity = free cash, price = 1.0.
        for a in portfolio.acc:
            curr = a.curr.upper().strip()
            if not curr:
                continue
            cash_qty = _coerce_float(a.s)
            if not cash_qty:
                continue
            rows.append({
                "Ticker":               curr,
                "Quantity":             cash_qty,
                "Purchase_Price":       1.0,
                "Broker_Current_Price": 1.0,
            })
            logger.info("Добавлен кэш: %s = %.2f", curr, cash_qty)

        return pd.DataFrame(rows, columns=["Ticker", "Quantity", "Purchase_Price", "Broker_Current_Price"])

    @staticmethod
    def _mock_portfolio() -> pd.DataFrame:
        """Deterministic template portfolio for demo / offline mode."""
        df = pd.DataFrame([
            {"Ticker": "AAPL",    "Quantity": 10,  "Purchase_Price": 150.0},
            {"Ticker": "KSPI",    "Quantity": 100, "Purchase_Price": 12.5},
            {"Ticker": "BTC-USD", "Quantity": 0.5, "Purchase_Price": 45_000.0},
        ])
        df.attrs["_ramp_is_mock"] = True
        return df
