"""
Tradernet REST client (read-only).

Three transport modes are tried in order:

  v2 signed (primary)  — POST form-urlencoded to /api/v2/cmd/{cmd} with
                         X-NtApi-Sig (HMAC-SHA256) + X-NtApi-PublicKey headers.
                         This is the path that modern apiKey/apiSecret retail
                         keys (32-char public + 40-char secret) authenticate
                         against.  Mirrors kofeinstyle/tradernet-sdk exactly.

  v1 signed (fallback) — POST q=<json> to /api/ with md5(...) signature.
                         Used by legacy uid-based authentication where the
                         server reads ``uid`` (account number) instead of
                         ``apiKey``.  Almost never the right answer for retail
                         keys, but kept for completeness.

  v1 unsigned (fallback) — POST q={cmd, params:{apiKey}} form-encoded to /api/.
                         Works for accounts that only expose a single Public
                         API key without a separate secret.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from freedom_portfolio.auth import build_request, build_v2_request
from freedom_portfolio.models import Portfolio

logger = logging.getLogger(__name__)

# Known Tradernet / Freedom Finance endpoints.
# get_portfolio() tries BASE_URL_LIVE first; if that rejects credentials it
# cycles through _FALLBACK_KZ_URLS in order until one succeeds or all fail.
BASE_URL_LIVE = "https://tradernet.com/api/"
BASE_URL_DEMO = "https://tradernet.com/api/"
DEFAULT_TIMEOUT = 30

# Browser-like User-Agent to prevent Cloudflare WAF from blocking server-side
# requests.  Google Cloud Run IP ranges (35.x / 34.x) are flagged by default
# when the User-Agent is "python-requests/...".
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# KZ-registered accounts may have keys that only authenticate on one of these
# Kazakhstan-specific deployments.  Both use the same /api/ + q= protocol.
_FALLBACK_KZ_URLS: tuple[str, ...] = (
    "https://tradernet.kz/api/",
    "https://freedombroker.kz/api/",
)


# ── Typed exceptions ──────────────────────────────────────────────────────────


class BrokerAPIError(RuntimeError):
    """Generic Tradernet API failure (non-zero ``code`` or ``errMsg`` set)."""


class CloudflareBlockError(BrokerAPIError):
    """Cloudflare WAF blocked the request (HTTP 403 + HTML challenge page)."""


class AuthenticationError(BrokerAPIError):
    """API rejected credentials (typically code=12)."""


class InvalidSignatureError(AuthenticationError):
    """API rejected the request signature (typically code=4)."""


class EmptyPortfolioError(BrokerAPIError):
    """Authenticated successfully but the account has no positions and no cash."""


# Code → exception map (per Tradernet error reference).
_CODE_EXCEPTIONS: dict[int, type[BrokerAPIError]] = {
    4:  InvalidSignatureError,
    12: AuthenticationError,
}


def _raise_on_error(raw: dict, *, raw_text: str = "") -> None:
    """
    Inspect a Tradernet response dict and raise the appropriate typed exception
    if it carries an error.  No-op on success.
    """
    err_msg  = raw.get("errMsg") or raw.get("error") or raw.get("err")
    err_code = raw.get("code")

    if not err_msg and (not isinstance(err_code, int) or err_code == 0):
        return

    text = str(err_msg) if err_msg else f"API error code {err_code}"
    exc_cls = _CODE_EXCEPTIONS.get(err_code, BrokerAPIError) if isinstance(err_code, int) else BrokerAPIError
    logger.error(
        "Tradernet API error: %s (code=%s) raw=%s",
        text, err_code, raw_text[:500] or json.dumps(raw)[:500],
    )
    raise exc_cls(text)


# ── Client ───────────────────────────────────────────────────────────────────


class TradernetClient:
    """
    Minimal read-only Tradernet REST client.

    ``public_key`` is required.  ``secret_key`` is optional — when it is empty
    the client falls back to the unsigned ``q=`` flow that works with the
    "Public API key" issued by retail Freedom Broker accounts.
    """

    BASE_URL = BASE_URL_LIVE

    def __init__(
        self,
        public_key: str,
        secret_key: str = "",
        *,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.public_key = public_key.strip()
        self.secret_key = secret_key.strip()
        self.base_url   = (base_url or self.BASE_URL).rstrip("/") + "/"
        self.timeout    = timeout
        self._session   = session or self._make_session()

    @staticmethod
    def _make_session() -> requests.Session:
        """Create a Session with browser-like headers and retry adapter."""
        s = requests.Session()
        s.headers.update(_DEFAULT_HEADERS)
        # Retry on 429/500/502/503 — but NOT 403 (handled explicitly).
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    # ── Public commands ──────────────────────────────────────────────────────

    def get_portfolio(self) -> Portfolio:
        """
        Fetch the user's portfolio.  Tries multiple auth modes and endpoints.

        Phase 1 — v2 signed on primary URL:
            POST /api/v2/cmd/{cmd} with HMAC-SHA256 sig + X-NtApi-PublicKey.
            This is the path retail apiKey/apiSecret pairs authenticate on.

        Phase 2 — v1 md5-signed on primary URL:
            Legacy uid-based path; almost never matches apiKey credentials but
            tried for backwards compatibility.

        Phase 3 — Unsigned on primary URL:
            ``apiKey`` in params JSON.  Falls through to Phase 4 on auth rejection.

        Phase 4 — Unsigned on KZ fallback endpoints:
            Tried in order: ``tradernet.kz``, ``freedombroker.kz``.
            Retail Freedom Finance Kazakhstan accounts issue keys that only
            authenticate on one of these KZ deployments.
            Auth rejection on all endpoints is truly fatal.
        """
        last_exc: Exception | None = None

        # ── Phase 1: v2 signed on primary URL (HMAC-SHA256, /v2/cmd/) ────────
        if self.secret_key:
            for cmd in ("getPositionJson", "getPortfolio"):
                try:
                    raw = self._with_cf_retry(self._post_v2_signed, cmd, {})
                    return self._parse_portfolio(raw)
                except (InvalidSignatureError, AuthenticationError) as exc:
                    logger.warning(
                        "Tradernet v2 signed auth rejected [cmd=%s apiKey_prefix=%s… secret_len=%d]: %s "
                        "— falling back to v1 md5 signing",
                        cmd,
                        self.public_key[:8] if self.public_key else "EMPTY",
                        len(self.secret_key),
                        exc,
                    )
                    last_exc = exc
                    break
                except BrokerAPIError as exc:
                    last_exc = exc
                    logger.warning("Tradernet '%s' (v2 signed) failed: %s — trying next", cmd, exc)

        # ── Phase 2: v1 md5-signed on primary URL ────────────────────────────
        if self.secret_key:
            for cmd in ("getPositionJson", "getPortfolio"):
                try:
                    raw = self._with_cf_retry(self._post_signed, cmd, {})
                    return self._parse_portfolio(raw)
                except (InvalidSignatureError, AuthenticationError) as exc:
                    logger.warning(
                        "Tradernet v1 md5 auth rejected [cmd=%s]: %s — falling back to unsigned",
                        cmd, exc,
                    )
                    last_exc = exc
                    break
                except BrokerAPIError as exc:
                    last_exc = exc
                    logger.warning("Tradernet '%s' (v1 signed) failed: %s — trying next", cmd, exc)

        # ── Phase 2: unsigned on primary URL ─────────────────────────────────
        _primary_rejected = False
        for cmd in ("getPositionJson", "getPortfolio"):
            try:
                raw = self._with_cf_retry(self._post_unsigned, cmd, {})
                return self._parse_portfolio(raw)
            except (InvalidSignatureError, AuthenticationError) as exc:
                logger.warning(
                    "Tradernet unsigned auth rejected on %s [cmd=%s]: %s "
                    "— trying KZ fallback endpoints",
                    self.base_url, cmd, exc,
                )
                last_exc = exc
                _primary_rejected = True
                break
            except BrokerAPIError as exc:
                last_exc = exc
                logger.warning("Tradernet '%s' (unsigned/%s) failed: %s — trying next",
                               cmd, self.base_url, exc)

        # ── Phase 3: unsigned on KZ fallback endpoints ────────────────────────
        # KZ-registered accounts may have keys that only work on tradernet.kz
        # or freedombroker.kz, not on the international tradernet.com gateway.
        if _primary_rejected:
            primary = self.base_url.rstrip("/")
            for fallback_url in _FALLBACK_KZ_URLS:
                if fallback_url.rstrip("/") == primary:
                    continue   # skip if this IS the primary URL
                logger.info("Trying KZ fallback endpoint: %s", fallback_url)
                for cmd in ("getPositionJson", "getPortfolio"):
                    try:
                        fb_client = TradernetClient(
                            self.public_key, "",
                            base_url=fallback_url,
                            timeout=self.timeout,
                            session=self._session,
                        )
                        raw = fb_client._with_cf_retry(fb_client._post_unsigned, cmd, {})
                        logger.info("'%s' succeeded on KZ fallback %s.", cmd, fallback_url)
                        return self._parse_portfolio(raw)
                    except (InvalidSignatureError, AuthenticationError) as exc:
                        logger.warning(
                            "KZ fallback %s rejected credentials [cmd=%s]: %s — trying next endpoint",
                            fallback_url, cmd, exc,
                        )
                        last_exc = exc
                        break   # move to next fallback URL
                    except BrokerAPIError as exc:
                        last_exc = exc
                        logger.warning("'%s' (unsigned/%s) failed: %s — trying next cmd",
                                       cmd, fallback_url, exc)

        assert last_exc is not None
        raise last_exc

    def get_user_info(self) -> dict[str, Any]:
        """Verify credentials by fetching the authenticated user's profile."""
        return self._call("getAuthInfo", {})

    # ── Internals ────────────────────────────────────────────────────────────

    _CF_MAX_RETRIES = 3
    _CF_BACKOFF_BASE = 2  # seconds

    def _with_cf_retry(self, fn, *args, **kwargs) -> dict:
        """
        Execute ``fn(*args, **kwargs)`` with Cloudflare-specific retry.

        If the first attempt hits a CloudflareBlockError, wait 2^attempt seconds
        and retry up to ``_CF_MAX_RETRIES`` times.  Non-Cloudflare exceptions
        propagate immediately.
        """
        for attempt in range(1, self._CF_MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except CloudflareBlockError:
                if attempt == self._CF_MAX_RETRIES:
                    raise
                wait = self._CF_BACKOFF_BASE ** attempt
                logger.info(
                    "Cloudflare retry %d/%d — ожидание %ds…",
                    attempt, self._CF_MAX_RETRIES, wait,
                )
                time.sleep(wait)
        raise RuntimeError("Unreachable")  # pragma: no cover

    def _call(self, cmd: str, params: dict) -> dict:
        """Dispatch ``cmd`` via signed or unsigned transport, return parsed JSON."""
        if self.secret_key:
            return self._post_signed(cmd, params)
        return self._post_unsigned(cmd, params)

    def _post_v2_signed(self, cmd: str, params: dict, *, base_override: str | None = None) -> dict:
        """
        v2 signed POST to ``/api/v2/cmd/{cmd}`` with HMAC-SHA256.

        Headers ``X-NtApi-Sig`` AND ``X-NtApi-PublicKey`` are both required —
        previous attempts in this project that used only ``X-NtApi-Sig`` were
        silently rejected by the server.  Body is form-urlencoded with bracket
        notation for nested params (matches kofeinstyle/tradernet-sdk).

        ``base_override`` lets callers route to a different regional host
        (e.g. ``https://tradernet.kz`` for KZ-registered accounts whose
        market-data subscription lives on the .kz endpoint).
        """
        payload, sig = build_v2_request(cmd, params, self.public_key, self.secret_key)
        body = self._to_form_urlencoded(payload)
        base = (base_override or self.base_url).rstrip("/")
        if base.endswith("/api"):
            base = base[: -len("/api")]
        url  = f"{base}/api/v2/cmd/{cmd}"
        headers = {
            "Content-Type":     "application/x-www-form-urlencoded",
            "X-NtApi-Sig":      sig,
            "X-NtApi-PublicKey": self.public_key,
        }
        logger.info(
            "Tradernet POST (v2 signed) %s [cmd=%s apiKey_prefix=%s… key_len=%d secret_len=%d sig_prefix=%s…]",
            url, cmd,
            self.public_key[:8] if self.public_key else "EMPTY",
            len(self.public_key),
            len(self.secret_key),
            sig[:8],
        )
        resp = self._session.post(url, data=body, headers=headers, timeout=self.timeout)
        return self._decode(resp)

    def _post_json_v2(self, cmd: str, params: dict, *, base_override: str | None = None) -> dict:
        """
        v2 JSON-body signed POST — used by ``getHloc``, ``get_trades_history``,
        ``get_orders_history`` etc.  This is the transport implemented by the
        official Tradernet Python SDK (``tradernet-sdk`` on PyPI, file
        ``tradernet/core.py:authorized_request``).

        Differences from ``_post_v2_signed``:
          * URL is ``{base}/api/{cmd}``  — NO ``/v2/cmd/`` segment.
          * Body is ``application/json`` (params object), NOT form-urlencoded.
          * Signature is ``HMAC-SHA256(secret, json_body + unix_timestamp)``.
          * Required header ``X-NtApi-Timestamp`` (unix seconds string).

        ``base_override`` lets callers target ``freedom24.com`` instead of the
        default ``tradernet.com`` — the official SDK hardcodes ``freedom24.com``
        for these commands, so callers should try it first and fall back to
        ``tradernet.com``.
        """
        import hashlib
        import hmac as _hmac
        import time as _time

        base = (base_override or self.base_url).rstrip("/")
        # Strip trailing /api/ if the base URL already includes it.
        if base.endswith("/api"):
            base = base[: -len("/api")]
        url = f"{base}/api/{cmd}"

        body      = json.dumps(params, separators=(",", ":"))
        timestamp = str(int(_time.time()))
        sig       = _hmac.new(
            self.secret_key.encode("utf-8"),
            (body + timestamp).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "Content-Type":      "application/json",
            "X-NtApi-PublicKey": self.public_key,
            "X-NtApi-Timestamp": timestamp,
            "X-NtApi-Sig":       sig,
        }
        logger.info(
            "Tradernet POST (v2 JSON) %s [cmd=%s ts=%s sig_prefix=%s…]",
            url, cmd, timestamp, sig[:8],
        )
        resp = self._session.post(url, data=body, headers=headers, timeout=self.timeout)
        return self._decode(resp)

    def _post_signed(self, cmd: str, params: dict, *, base_override: str | None = None) -> dict:
        body = build_request(cmd, params, self.public_key, self.secret_key)
        # Tradernet /api/ ALWAYS expects the request body in the `q` form field —
        # both signed and unsigned modes.  Sending the JSON directly as the
        # request body produces "Invalid 'q' provided" (code=5).  The signed
        # payload (apiKey/cmd/nonce/params/sig) is JSON-encoded and dropped
        # into a single form parameter.
        q_payload = json.dumps(body, separators=(",", ":"))
        url = self._url_with_override(base_override)
        logger.info(
            "Tradernet POST (signed) %s [cmd=%s apiKey=%s… key_len=%d secret_len=%d sig_prefix=%s…]",
            url, cmd,
            self.public_key[:12] if self.public_key else "EMPTY",
            len(self.public_key),
            len(self.secret_key),
            body.get("sig", "")[:8],
        )
        resp = self._session.post(
            url,
            data={"q": q_payload},
            timeout=self.timeout,
        )
        return self._decode(resp)

    def _post_unsigned(self, cmd: str, params: dict, *, base_override: str | None = None) -> dict:
        params_with_key = {**params, "apiKey": self.public_key}
        q_payload = json.dumps({"cmd": cmd, "params": params_with_key}, separators=(",", ":"))
        url = self._url_with_override(base_override)
        logger.info(
            "Tradernet POST (unsigned) %s [cmd=%s key_prefix=%s… key_len=%d]",
            url, cmd,
            self.public_key[:8] if self.public_key else "EMPTY",
            len(self.public_key),
        )
        resp = self._session.post(
            url,
            data={"q": q_payload},
            timeout=self.timeout,
        )
        return self._decode(resp)

    def _url_with_override(self, base_override: str | None) -> str:
        """Resolve the v1 ``/api/`` URL, honouring an optional regional override."""
        base = (base_override or self.base_url).rstrip("/")
        if not base.endswith("/api"):
            base = base + "/api"
        return base + "/"

    @staticmethod
    def _to_form_urlencoded(data: dict, prefix: str = "") -> str:
        """
        Serialize *data* to ``application/x-www-form-urlencoded`` with bracket
        notation for nested dicts.  Matches the JS reference implementation
        (kofeinstyle/tradernet-sdk).  Booleans become 1/0 to match the PHP
        backend's serialisation expectations.
        """
        parts: list[str] = []
        for key, value in data.items():
            encoded_key = f"{prefix}[{quote(str(key), safe='')}]" if prefix else quote(str(key), safe='')
            if isinstance(value, dict):
                # Empty nested dicts are still sent as "params=" — server expects the field.
                if not value:
                    parts.append(f"{encoded_key}=")
                else:
                    parts.append(TradernetClient._to_form_urlencoded(value, encoded_key))
            else:
                if value is True:
                    rendered = "1"
                elif value is False:
                    rendered = "0"
                elif value is None:
                    rendered = ""
                else:
                    rendered = str(value)
                parts.append(f"{encoded_key}={quote(rendered, safe='')}")
        return "&".join(parts)

    @staticmethod
    def _is_cloudflare_block(resp: requests.Response) -> bool:
        """Detect Cloudflare WAF challenge page (HTML with IE conditionals)."""
        if resp.status_code != 403:
            return False
        ct = resp.headers.get("content-type", "")
        body = resp.text[:500]
        return (
            "text/html" in ct
            or "<!DOCTYPE" in body
            or "no-js" in body
            or "cloudflare" in body.lower()
        )

    @staticmethod
    def _decode(resp: requests.Response) -> dict:
        # Cloudflare detection — raise a distinct exception so callers can retry.
        if TradernetClient._is_cloudflare_block(resp):
            logger.warning(
                "Cloudflare WAF заблокировал запрос к %s (HTTP 403, HTML challenge). "
                "Вероятно, IP-адрес Cloud Run попал в фильтр.",
                resp.url,
            )
            raise CloudflareBlockError(
                f"Cloudflare WAF blocked request to {resp.url} — "
                f"server returned HTML challenge page instead of API response."
            )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise BrokerAPIError(f"HTTP {resp.status_code}: {resp.text[:200]}") from exc
        try:
            data = resp.json()
        except ValueError as exc:
            raise BrokerAPIError(f"Non-JSON response: {resp.text[:200]}") from exc
        _raise_on_error(data, raw_text=resp.text)
        return data

    # ── Parsing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_portfolio(raw: dict) -> Portfolio:
        """
        Locate the ``ps`` payload inside a Tradernet response and return a Portfolio.

        Tradernet wraps the portfolio under ``result.ps`` for some commands and
        under top-level ``ps`` for others; this helper handles both.
        """
        body = raw.get("result", raw)
        ps   = body.get("ps") or body
        return Portfolio(**ps)
