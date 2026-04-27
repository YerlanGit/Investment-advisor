"""
Tradernet REST client (read-only).

Two transport modes:

  Signed   — primary path.  POST JSON {apiKey, cmd, nonce, params, sig} to /api/.
             ``sig`` is built by ``auth.build_signature`` (md5 of sorted concat
             with secret appended).  Use this when you have an apiKey/secret pair.

  Unsigned — fallback for accounts that only expose a single "Public API key"
             (no separate secret).  POST form-encoded ``q={cmd, params:{apiKey}}``.
             This is the legacy /api/ flow used by the Tradernet web UI.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from freedom_portfolio.auth import build_request
from freedom_portfolio.models import Portfolio

logger = logging.getLogger(__name__)

# Two known Tradernet endpoints:
#   tradernet.com  — international gateway; official spec target.
#   tradernet.kz   — Kazakhstan-specific deployment; retail KZ accounts may have
#                    keys that only authenticate here, not on tradernet.com.
# get_portfolio() tries .com first; on "Invalid credentials" it retries on .kz.
BASE_URL_LIVE    = "https://tradernet.com/api/"
BASE_URL_LIVE_KZ = "https://tradernet.kz/api/"
BASE_URL_DEMO    = "https://tradernet.com/api/"
DEFAULT_TIMEOUT  = 30


# ── Typed exceptions ──────────────────────────────────────────────────────────


class BrokerAPIError(RuntimeError):
    """Generic Tradernet API failure (non-zero ``code`` or ``errMsg`` set)."""


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
        self._session   = session or requests.Session()

    # ── Public commands ──────────────────────────────────────────────────────

    def get_portfolio(self) -> Portfolio:
        """
        Fetch the user's portfolio.  Tries multiple auth modes and endpoints.

        Phase 1 — Signed on primary URL (``tradernet.com``):
            MD5-signed request.  Falls through to Phase 2 on auth rejection.

        Phase 2 — Unsigned on primary URL:
            ``apiKey`` in params JSON.  Falls through to Phase 3 on auth rejection.

        Phase 3 — Unsigned on KZ endpoint (``tradernet.kz``):
            Retail Freedom Finance Kazakhstan accounts issue keys that only
            authenticate on the KZ deployment, not the international gateway.
            Auth rejection here is truly fatal.
        """
        last_exc: Exception | None = None

        # ── Phase 1: signed on primary URL ───────────────────────────────────
        if self.secret_key:
            for cmd in ("getPositionJson", "getPortfolio"):
                try:
                    raw = self._post_signed(cmd, {})
                    return self._parse_portfolio(raw)
                except (InvalidSignatureError, AuthenticationError) as exc:
                    logger.warning(
                        "Tradernet signed auth rejected on %s [cmd=%s key_prefix=%s… secret_len=%d]: %s "
                        "— falling back to unsigned",
                        self.base_url, cmd,
                        self.public_key[:8] if self.public_key else "EMPTY",
                        len(self.secret_key),
                        exc,
                    )
                    last_exc = exc
                    break
                except BrokerAPIError as exc:
                    last_exc = exc
                    logger.warning("Tradernet '%s' (signed) failed: %s — trying next", cmd, exc)

        # ── Phase 2: unsigned on primary URL ─────────────────────────────────
        _com_rejected = False
        for cmd in ("getPositionJson", "getPortfolio"):
            try:
                raw = self._post_unsigned(cmd, {})
                return self._parse_portfolio(raw)
            except (InvalidSignatureError, AuthenticationError) as exc:
                logger.warning(
                    "Tradernet unsigned auth rejected on %s [cmd=%s]: %s "
                    "— trying tradernet.kz endpoint",
                    self.base_url, cmd, exc,
                )
                last_exc = exc
                _com_rejected = True
                break
            except BrokerAPIError as exc:
                last_exc = exc
                logger.warning("Tradernet '%s' (unsigned/%s) failed: %s — trying next",
                               cmd, self.base_url, exc)

        # ── Phase 3: unsigned on KZ endpoint ─────────────────────────────────
        # Only try if the primary URL rejected credentials — KZ keys are only
        # valid on tradernet.kz, not tradernet.com.
        if _com_rejected and self.base_url.rstrip("/") != BASE_URL_LIVE_KZ.rstrip("/"):
            kz_client = TradernetClient(
                self.public_key, "",   # unsigned — only public key
                base_url=BASE_URL_LIVE_KZ,
                timeout=self.timeout,
                session=self._session,
            )
            for cmd in ("getPositionJson", "getPortfolio"):
                try:
                    raw = kz_client._post_unsigned(cmd, {})
                    logger.info("Tradernet '%s' succeeded on tradernet.kz fallback.", cmd)
                    return self._parse_portfolio(raw)
                except (InvalidSignatureError, AuthenticationError):
                    raise   # KZ also rejected — nothing more to try
                except BrokerAPIError as exc:
                    last_exc = exc
                    logger.warning("Tradernet '%s' (unsigned/kz) failed: %s — trying next", cmd, exc)

        assert last_exc is not None
        raise last_exc

    def get_user_info(self) -> dict[str, Any]:
        """Verify credentials by fetching the authenticated user's profile."""
        return self._call("getAuthInfo", {})

    # ── Internals ────────────────────────────────────────────────────────────

    def _call(self, cmd: str, params: dict) -> dict:
        """Dispatch ``cmd`` via signed or unsigned transport, return parsed JSON."""
        if self.secret_key:
            return self._post_signed(cmd, params)
        return self._post_unsigned(cmd, params)

    def _post_signed(self, cmd: str, params: dict) -> dict:
        body = build_request(cmd, params, self.public_key, self.secret_key)
        # Tradernet /api/ ALWAYS expects the request body in the `q` form field —
        # both signed and unsigned modes.  Sending the JSON directly as the
        # request body produces "Invalid 'q' provided" (code=5).  The signed
        # payload (apiKey/cmd/nonce/params/sig) is JSON-encoded and dropped
        # into a single form parameter.
        q_payload = json.dumps(body, separators=(",", ":"))
        logger.info(
            "Tradernet POST (signed) %s [cmd=%s apiKey=%s… key_len=%d secret_len=%d sig_prefix=%s…]",
            self.base_url, cmd,
            self.public_key[:12] if self.public_key else "EMPTY",
            len(self.public_key),
            len(self.secret_key),
            body.get("sig", "")[:8],
        )
        resp = self._session.post(
            self.base_url,
            data={"q": q_payload},
            timeout=self.timeout,
        )
        return self._decode(resp)

    def _post_unsigned(self, cmd: str, params: dict) -> dict:
        params_with_key = {**params, "apiKey": self.public_key}
        q_payload = json.dumps({"cmd": cmd, "params": params_with_key}, separators=(",", ":"))
        logger.info(
            "Tradernet POST (unsigned) %s [cmd=%s key_prefix=%s… key_len=%d]",
            self.base_url, cmd,
            self.public_key[:8] if self.public_key else "EMPTY",
            len(self.public_key),
        )
        resp = self._session.post(
            self.base_url,
            data={"q": q_payload},
            timeout=self.timeout,
        )
        return self._decode(resp)

    @staticmethod
    def _decode(resp: requests.Response) -> dict:
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
