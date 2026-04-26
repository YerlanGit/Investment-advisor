"""
FreedomConnector — live Tradernet/Freedom Broker API client.

Three authentication methods are tried in order:

  Phase 1 — HMAC-signed v2  (POST /api/v2/cmd/{cmd})
    Body:    url-form-encoded  apiKey=&cmd=&nonce=&params[...]=
    Sig:     HMAC-SHA256(secret_key, convert_to_query_string(body_dict))
    Header:  X-NtApi-Sig  (no X-NtApi-PublicKey — apiKey goes in body)
    Requires a *private key pair* generated in broker account settings.
    Most retail accounts never get this — they get a single API key instead.

  Phase 2 — Unsigned v1  (POST /api/)
    Embeds apiKey inside the params JSON (q field) — no signing needed.
    Works with the standard 'Public API Key' available to all accounts.
    Uses getPositionJson command (getPortfolio does not exist on tradernet.kz).

  Phase 3 — Session login  (POST /api/ cmd=login → sid)
    Uses account username (login) + password (secret_key) to obtain a session.
    This is the standard retail auth flow for Freedom Finance accounts that
    have neither an HMAC key pair nor a standalone Public API key.
    The 'login' stored in the vault is used as the username; 'secret_key'
    is treated as the account password for this phase only.

Demo mode: when api_key == 'demo', returns a hardcoded mock portfolio.

Service-level credentials: FREEDOM_API_KEY / FREEDOM_API_SECRET / FREEDOM_LOGIN
environment variables are used when no per-user vault keys are present.
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
#   v2 HMAC-signed  — POST /api/v2/cmd/{cmd}
#                     Body: form-encoded  apiKey=&cmd=&nonce=&params[...]=
#                     Sig:  HMAC-SHA256(secret, convert_to_query_string(body_dict))
#                     Header: X-NtApi-Sig (no X-NtApi-PublicKey — apiKey goes in body)
#                     Requires a *private* key pair issued via broker "API access" settings.
#
#   v1 unsigned     — POST /api/ with {"cmd":…,"params":{"apiKey":…}} inside
#                     form field `q`.  Works with the standard "Public API key"
#                     that every Freedom Broker account can generate.
#
# fetch_portfolio() tries v2 first; on code=12 "Invalid signature" it falls back
# to v1 automatically so both account types work without user reconfiguration.
TRADERNET_URL_V2 = os.getenv("TRADERNET_URL_V2", "https://tradernet.kz/api/v2")
TRADERNET_URL_V1 = os.getenv("TRADERNET_URL_V1", "https://tradernet.kz/api/")
REQUEST_TIMEOUT  = 30   # seconds
DEMO_KEY        = "demo"

# Keep for backwards-compat with any env override that set TRADERNET_URL
_TRADERNET_URL_LEGACY = os.getenv("TRADERNET_URL", "")
if _TRADERNET_URL_LEGACY:
    TRADERNET_URL_V2 = _TRADERNET_URL_LEGACY.rstrip("/").removesuffix("/v2/cmd").removesuffix("/v2")

# Service-level Freedom Broker credentials (injected via Secret Manager).
# .strip() at source prevents hidden newlines from Secret Manager corrupting
# HMAC signatures or being passed into per-user vault comparisons.
FREEDOM_API_KEY    = os.getenv("FREEDOM_API_KEY",    "").strip()
FREEDOM_API_SECRET = os.getenv("FREEDOM_API_SECRET", "").strip()
FREEDOM_LOGIN      = os.getenv("FREEDOM_LOGIN",      "").strip()

# Commands tried per authentication phase.
# Phase 1 HMAC: getPositionJson is the confirmed v2 command.
# Phase 2 unsigned: getPositionJson only — getPortfolio returns code=5 "Command not found"
#   at tradernet.kz/api/; it does not exist on that endpoint.
# Phase 3 session: getPositionJson first.
_PORTFOLIO_CMDS_HMAC     = ("getPositionJson",)
_PORTFOLIO_CMDS_UNSIGNED = ("getPositionJson",)
_PORTFOLIO_CMDS_SESSION  = ("getPositionJson",)
_BALANCE_CMDS            = ("getBalance", "getClientInfo")

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
    def __init__(self, api_key: str = "", secret_key: str = "", login: str = ""):
        # Fall back to service-level env var credentials when not explicitly supplied.
        # .strip() removes hidden newlines/spaces injected by Secret Manager.
        self.api_key    = (api_key    or FREEDOM_API_KEY).strip()
        self.secret_key = (secret_key or FREEDOM_API_SECRET).strip()
        self.login      = (login      or FREEDOM_LOGIN).strip()

    # ── Internal request helpers ─────────────────────────────────────────────

    @staticmethod
    def _convert_to_query_string(data: dict) -> str:
        """
        Tradernet v2 signing serialiser (port of the Python reference client).

        Sorts keys alphabetically, joins as "key=value" pairs with "&",
        recursing into nested dicts so that nested values become the inner
        serialised string (e.g. params={} → "params=").
        This output is what is HMAC-SHA256-signed.
        """
        result = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, dict):
                value = FreedomConnector._convert_to_query_string(value)
            else:
                value = "" if value is None else str(value)
            result.append(f"{key}={value}")
        return "&".join(result)

    @staticmethod
    def _url_form_encoded(data: dict, root_name: str = "") -> str:
        """
        Tradernet v2 body serialiser — produces bracket-notation form encoding.

        Top-level scalar fields → "key=value".
        Nested dict fields     → "parentKey[childKey]=value".
        Empty nested dicts are omitted (no output for that key).
        This is sent as the POST body (Content-Type: application/x-www-form-urlencoded).
        """
        result = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, dict):
                inner = FreedomConnector._url_form_encoded(value, key)
                if inner:
                    result.append(inner)
            else:
                if root_name:
                    result.append(f"{root_name}[{key}]={value}")
                else:
                    result.append(f"{key}={value}")
        return "&".join(result)

    def _post(self, cmd: str, extra_params: dict | None = None) -> dict:
        """
        Send a HMAC-signed v2 command to the Freedom Finance / Tradernet API.

        Correct v2 format (from official Python reference client):
          URL:     POST /api/v2/cmd/{cmd}
          Body:    url-form-encoded  apiKey=&cmd=&nonce=&params[...]=
          Signing: HMAC-SHA256(secret_key, _convert_to_query_string(body_dict))
          Header:  X-NtApi-Sig (no X-NtApi-PublicKey — apiKey is in body)

        For getPositionJson with empty params, the signed string is:
          "apiKey=<key>&cmd=getPositionJson&nonce=<ts>&params="
        """
        params = extra_params or {}
        nonce  = int(time.time() * 10000)

        request_dict = {
            "apiKey": self.api_key,
            "cmd":    cmd,
            "nonce":  nonce,
            "params": params,
        }

        sign_input = self._convert_to_query_string(request_dict)
        sig = hmac.new(
            self.secret_key.encode('utf-8'),
            sign_input.encode('utf-8'),
            hashlib.sha256,
        ).hexdigest()

        body = self._url_form_encoded(request_dict)
        url  = f"{TRADERNET_URL_V2}/cmd/{cmd}"

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-NtApi-Sig":  sig,
        }

        logger.info(
            "POST (HMAC/v2) %s  apiKey_prefix=%s…  secret_len=%d  sign_input=%r",
            url,
            self.api_key[:8]    if self.api_key    else "EMPTY",
            len(self.secret_key),
            sign_input[:120],
        )

        resp = requests.post(url, data=body, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()

        # Freedom Finance API signals errors via "errMsg" + "code", not "error".
        # Also handle legacy "error" key for defensive coverage.
        err_msg  = raw.get("errMsg") or raw.get("error") or raw.get("err")
        err_code = raw.get("code")
        if err_msg or (isinstance(err_code, int) and err_code != 0):
            err_text = str(err_msg) if err_msg else f"API error code {err_code}"
            logger.error(
                "Freedom API error (HMAC/v2) [cmd=%s]: %s (code=%s) raw=%s",
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

        logger.info(
            "POST (unsigned/v1) %s  [cmd=%s]  key_prefix=%s…  key_len=%d",
            TRADERNET_URL_V1, cmd,
            self.api_key[:8] if self.api_key else "EMPTY",
            len(self.api_key),
        )

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

    def _get_session(self) -> str:
        """
        Phase-3 auth: POST cmd=login with account username + password → session SID.

        'login' (vault field) is used as the Freedom Finance account username.
        Falls back to api_key as username if login is not stored.
        'secret_key' is treated as the account PASSWORD for this phase only
        (not as an HMAC secret).

        This is the standard retail auth flow for accounts that have neither
        an HMAC key pair nor a standalone Public API key.
        """
        username = self.login or self.api_key
        password = self.secret_key
        if not username or not password:
            raise BrokerAuthError(
                "Session login requires account username (login) and password (secret_key)"
            )

        q_payload = json.dumps(
            {"cmd": "login", "params": {"login": username, "password": password}},
            separators=(',', ':'),
        )
        logger.info(
            "POST (session/login) %s  [username=%s…]",
            TRADERNET_URL_V1,
            username[:6],
        )

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
            err_text = str(err_msg) if err_msg else f"Login error code {err_code}"
            logger.error("Session login failed: %s (code=%s) raw=%s", err_text, err_code, resp.text[:300])
            raise BrokerAuthError(f"Авторизация в Freedom Broker не прошла: {err_text}")

        result = raw.get("result", raw)
        sid = (
            result.get("sid")
            or result.get("token")
            or result.get("sessionId")
            or result.get("session")
        )
        if not sid:
            raise BrokerAuthError(f"Сервер не вернул session ID. Ответ: {str(raw)[:200]}")

        logger.info("Session login OK — SID prefix: %s…", str(sid)[:8])
        return str(sid)

    def _post_with_session(self, cmd: str, sid: str) -> dict:
        """Send a portfolio command authenticated with a session SID."""
        q_payload = json.dumps(
            {"cmd": cmd, "params": {"sid": sid}},
            separators=(',', ':'),
        )
        logger.info("POST (session/v1) %s  [cmd=%s]", TRADERNET_URL_V1, cmd)

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
                "Freedom API error (session) [cmd=%s]: %s (code=%s) raw=%s",
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

        Three auth phases tried in order — first success wins:
          1. HMAC-signed v2    (X-NtApi-PublicKey + signature)
          2. Unsigned v1       (apiKey embedded in params JSON)
          3. Session login v1  (login + password → SID → portfolio command)

        Raises BrokerAuthError when all three phases confirm bad credentials.
        Falls back to a marked mock DataFrame only on network/command errors
        so the MAC3 gate in analyze_all() halts before yfinance.
        """
        if self.api_key == DEMO_KEY:
            logger.info("Демо-режим: используется шаблонный портфель.")
            df = self._mock_portfolio()
            df.attrs['_ramp_source'] = 'demo'
            return df

        last_error: Exception | None = None

        # ── Phase 1: HMAC-signed v2 ──────────────────────────────────────────
        _hmac_rejected = False
        for cmd in _PORTFOLIO_CMDS_HMAC:
            try:
                raw = self._post(cmd)
                df  = self._parse(raw)
                logger.info("'%s' via HMAC/v2 — успешно.", cmd)
                return df
            except BrokerAuthError as exc:
                logger.warning(
                    "HMAC/v2 отклонён [cmd=%s]: %s — переходим к unsigned/v1.",
                    cmd, exc,
                )
                last_error     = exc
                _hmac_rejected = True
                break   # auth rejected — no point trying other HMAC commands
            except BrokerEmptyPortfolioError:
                raise
            except (RuntimeError, requests.RequestException) as exc:
                logger.warning("HMAC/v2 ошибка [cmd=%s]: %s", cmd, exc)
                last_error = exc

        # ── Phase 2: unsigned v1 (apiKey in params) ──────────────────────────
        _unsigned_rejected = False
        for cmd in _PORTFOLIO_CMDS_UNSIGNED:
            try:
                raw = self._post_unsigned(cmd)
                df  = self._parse(raw)
                logger.info("'%s' via unsigned/v1 — успешно.", cmd)
                return df
            except BrokerAuthError as exc:
                logger.warning(
                    "unsigned/v1 отклонён [cmd=%s]: %s — переходим к session/v1.",
                    cmd, exc,
                )
                last_error         = exc
                _unsigned_rejected = True
                break   # auth rejected — try session login instead
            except BrokerEmptyPortfolioError:
                raise
            except (RuntimeError, requests.RequestException) as exc:
                logger.warning("unsigned/v1 ошибка [cmd=%s]: %s", cmd, exc)
                last_error = exc

        # ── Phase 3: session login (account username + password) ─────────────
        try:
            sid = self._get_session()
            for cmd in _PORTFOLIO_CMDS_SESSION:
                try:
                    raw = self._post_with_session(cmd, sid)
                    df  = self._parse(raw)
                    logger.info("'%s' via session/v1 — успешно.", cmd)
                    return df
                except BrokerAuthError:
                    raise   # session expired mid-request — surface immediately
                except BrokerEmptyPortfolioError:
                    raise
                except (RuntimeError, requests.RequestException) as exc:
                    logger.warning("session/v1 ошибка [cmd=%s]: %s", cmd, exc)
                    last_error = exc
        except BrokerAuthError:
            raise   # login itself failed (bad username/password) — surface to user
        except Exception as exc:
            logger.error("Phase 3 session auth недоступна: %s", exc)
            last_error = exc

        # All three phases failed — return marked fallback; MAC3 gate blocks analysis
        logger.error(
            "Все три фазы аутентификации не сработали (последняя ошибка: %s) — "
            "возвращаю помеченный fallback-портфель; MAC3 движок заблокирует анализ.",
            last_error,
        )
        df = self._mock_portfolio()
        df.attrs['_ramp_is_fallback'] = True
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
