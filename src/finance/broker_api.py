"""
FreedomConnector — live Tradernet/Freedom Broker API client.

Authentication (tn-crypto.js algorithm):
  sig = HMAC-SHA256(secret_key, preSign({cmd, params}))

  preSign(obj): sort keys alphabetically, format each as "key=value"
  (recursing into nested objects), join with "&".

  Example for getPositionJson (no params):
    preSign({"cmd": "getPositionJson", "params": {}})
    → "cmd=getPositionJson&params="   (empty dict → empty string)
    sig = HMAC-SHA256(secret, "cmd=getPositionJson&params=")

  Header X-NtApi-PublicKey carries the public key; X-NtApi-Sig carries sig.
  q (the JSON-serialised full payload) is sent as a form field for routing.

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

# Two Freedom Finance API modes:
#
#   v2 HMAC-signed  — POST /api/v2/ with X-NtApi-PublicKey + X-NtApi-Sig headers.
#                     Requires a *private* key pair issued via the broker's
#                     "API access" settings — most retail accounts never get this.
#
#   v1 unsigned     — POST /api/ with {"cmd":…,"params":{"apiKey":…}} inside the
#                     form field `q`.  Works with the standard "Public API key"
#                     that every Freedom Broker account can generate.
#
# fetch_portfolio() tries v2 first; on code=12 "Invalid signature" it falls back
# to v1 automatically so both account types work without user reconfiguration.
TRADERNET_URL    = os.getenv("TRADERNET_URL",    "https://tradernet.kz/api/v2/")
TRADERNET_URL_V1 = os.getenv("TRADERNET_URL_V1", "https://tradernet.kz/api/")
REQUEST_TIMEOUT  = 30   # seconds
DEMO_KEY        = "demo"

# Service-level Freedom Broker credentials (injected via Secret Manager).
# .strip() at source prevents hidden newlines from Secret Manager corrupting
# HMAC signatures or being passed into per-user vault comparisons.
FREEDOM_API_KEY    = os.getenv("FREEDOM_API_KEY", "").strip()
FREEDOM_API_SECRET = os.getenv("FREEDOM_API_SECRET", "").strip()

# getPositionJson is the only confirmed-working command on standard Freedom accounts.
# getPositions / getPortfolio / getPortfolioFull all return code 5 on standard tiers
# and were removed to avoid 90 s of pointless timeout before the mock fallback.
_PORTFOLIO_CMDS = ("getPositionJson",)
_BALANCE_CMDS   = ("getBalance", "getClientInfo")

# API error substrings that indicate credential/auth rejection.
# "invalid signature" / code=12 means the HMAC is wrong — treat as auth failure
# so the bot surfaces the error immediately rather than falling back to mock data.
_AUTH_ERROR_PHRASES = (
    "invalid credentials",
    "invalid signature",
    "invalid sig",
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


class RealPortfolioRequired(RuntimeError):
    """
    Raised by the MAC3 engine gate when the portfolio DataFrame is a
    fallback-mock produced by broker API failure (code=5 / network error).
    Distinct from intentional demo mode (api_key == 'demo').
    """


class FreedomConnector:
    def __init__(self, api_key: str = "", secret_key: str = ""):
        # Fall back to service-level env var credentials when not explicitly supplied.
        # .strip() removes hidden newlines/spaces injected by Secret Manager.
        self.api_key    = (api_key    or FREEDOM_API_KEY).strip()
        self.secret_key = (secret_key or FREEDOM_API_SECRET).strip()

    # ── Internal request helper ──────────────────────────────────────────────

    @staticmethod
    def _pre_sign(data: dict) -> str:
        """
        Port of tn-crypto.js preSign(): sort keys alphabetically, join as
        "key=value" pairs with "&", recursing into nested dicts.
        Empty dicts produce an empty string for their value.
        """
        parts = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, dict):
                value = FreedomConnector._pre_sign(value)
            else:
                value = "" if value is None else str(value)
            parts.append(f"{key}={value}")
        return "&".join(parts)

    def _post(self, cmd: str, extra_params: dict | None = None) -> dict:
        """
        Send a signed command to the Freedom Finance / Tradernet API.

        Signing (tn-crypto.js algorithm):
          sign_input = preSign({"cmd": cmd, "params": params})
          sig        = HMAC-SHA256(secret_key, sign_input).hexdigest()

        For getPositionJson with empty params:
          sign_input = "cmd=getPositionJson&params="
        """
        params = extra_params or {}

        sign_input = self._pre_sign({"cmd": cmd, "params": params})
        sig = hmac.new(
            self.secret_key.encode('utf-8'),
            sign_input.encode('utf-8'),
            hashlib.sha256,
        ).hexdigest()

        # q is the full JSON payload sent as a form field for server-side routing.
        q_payload = json.dumps({"cmd": cmd, "params": params}, separators=(',', ':'))

        headers = {
            "Content-Type":      "application/x-www-form-urlencoded",
            "X-NtApi-PublicKey": self.api_key,
            "X-NtApi-Sig":       sig,
        }

        form_data = {"cmd": cmd, "q": q_payload}

        logger.info("POST %s  [cmd=%s]", TRADERNET_URL, cmd)
        logger.info(
            "SIGN key=%s… secret_len=%d secret_prefix=%s… sign_input=%r",
            self.api_key[:6]    if self.api_key    else "EMPTY",
            len(self.secret_key),
            self.secret_key[:4] if self.secret_key else "EMPTY",
            sign_input,
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

    def _post_unsigned(self, cmd: str, extra_params: dict | None = None) -> dict:
        """
        Unsigned POST to the v1 endpoint — Freedom Broker 'Public API' style.

        The API key is embedded inside the params JSON (no HMAC, no special headers).
        This is the authentication method for standard retail accounts that generate
        a single 'API Key' in account settings without a separate private/secret key.

        Request shape:
            POST /api/
            Content-Type: application/x-www-form-urlencoded
            Body: q={"cmd":"getPositionJson","params":{"apiKey":"<key>"}}
        """
        params = {**(extra_params or {}), "apiKey": self.api_key}
        q_payload = json.dumps({"cmd": cmd, "params": params}, separators=(',', ':'))

        logger.info("POST (unsigned/v1) %s  [cmd=%s]", TRADERNET_URL_V1, cmd)

        resp = requests.post(
            TRADERNET_URL_V1,
            data={"q": q_payload},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()

        err_msg  = raw.get("errMsg") or raw.get("error") or raw.get("err")
        err_code = raw.get("code")
        if err_msg or (isinstance(err_code, int) and err_code != 0):
            err_text = str(err_msg) if err_msg else f"API error code {err_code}"
            logger.error(
                "Freedom API error (unsigned) [cmd=%s]: %s (code=%s) raw=%s",
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
            df = self._mock_portfolio()
            df.attrs['_ramp_source'] = 'demo'
            return df

        last_error: Exception | None = None

        for cmd in _PORTFOLIO_CMDS:
            # ── Step 1: HMAC-signed v2 (private key pair accounts) ──────────
            try:
                raw = self._post(cmd)
                df  = self._parse(raw)
                logger.info("'%s' via HMAC/v2 — успешно.", cmd)
                return df
            except BrokerAuthError as hmac_err:
                # code=12 "Invalid signature" — this account uses the unsigned
                # Public API, not the private-key-pair HMAC API.  Try v1 next.
                logger.warning(
                    "HMAC/v2 подпись отклонена [cmd=%s]: %s — "
                    "переключаюсь на unsigned/v1 Public API.",
                    cmd, hmac_err,
                )
            except BrokerEmptyPortfolioError:
                raise
            except RuntimeError as exc:
                logger.warning("HMAC/v2 [cmd=%s] не поддерживается: %s", cmd, exc)
                last_error = exc
                continue
            except requests.RequestException as exc:
                logger.error("Freedom API недоступен (HMAC) [cmd=%s]: %s", cmd, exc)
                last_error = exc
                continue

            # ── Step 2: unsigned v1 (standard Public API key) ───────────────
            try:
                raw = self._post_unsigned(cmd)
                df  = self._parse(raw)
                logger.info("'%s' via unsigned/v1 — успешно.", cmd)
                return df
            except BrokerAuthError:
                raise  # Bad API key — surface immediately, do not fall back to mock
            except BrokerEmptyPortfolioError:
                raise
            except RuntimeError as exc:
                logger.warning("unsigned/v1 [cmd=%s] ошибка: %s", cmd, exc)
                last_error = exc
            except requests.RequestException as exc:
                logger.error("Freedom API недоступен (unsigned) [cmd=%s]: %s", cmd, exc)
                last_error = exc

        logger.error(
            "Все команды API не сработали (последняя ошибка: %s) — "
            "возвращаю помеченный fallback-портфель; MAC3 движок заблокирует анализ.",
            last_error,
        )
        df = self._mock_portfolio()
        df.attrs['_ramp_is_fallback'] = True   # signals RealPortfolioRequired in analyze_all
        return df

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
        df = pd.DataFrame([
            {"Ticker": "AAPL",    "Quantity": 10,  "Purchase_Price": 150.0},
            {"Ticker": "KSPI",    "Quantity": 100, "Purchase_Price": 12.5},
            {"Ticker": "BTC-USD", "Quantity": 0.5, "Purchase_Price": 45_000.0},
        ])
        df.attrs['_ramp_is_mock'] = True
        return df
