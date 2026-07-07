"""
FRED-backed FX feed for the MAC3 risk engine (H2 wiring).

This is the production implementation of the `currency.FxProvider`
contract: a callable ``(base_ccy, quote_ccy) -> pd.Series | None`` that
returns a daily-indexed FX history suitable for `align_fx_to_prices`.

Currently supported pairs
─────────────────────────
  USD → KZT   via FRED series DEXKZUS (Kazakhstan Tenges per 1 US Dollar)
  KZT → USD   via 1 / DEXKZUS

DEXKZUS is published by the Federal Reserve H.10 release.  Updates are
daily on US business days, lagged ~1-2 calendar days.

Why not Tradernet?
──────────────────
The Tradernet wrapper currently surfaces only equity/ETF close prices.
DEXKZUS is a public, free, well-maintained alternative that we already
authenticate against via the `FRED_API_KEY` secret in Cloud Run.

Design constraints
──────────────────
* Lazy — constructing the provider does NOT hit the network.  The first
  network call happens when the engine actually calls the provider for
  a missing FX pair.
* Cache-first — disk cache at /tmp/fx_cache (writable on Cloud Run);
  falls back to cache when FRED is unreachable.
* Test-injectable — `http_get` and `sleep` are constructor hooks, so
  the unit tests run completely offline.
* Pure pandas/numpy — no extra dependencies beyond what the engine
  already needs.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests


logger = logging.getLogger("FxFeed")


# ── FRED series registry ────────────────────────────────────────────────────
# (base, quote) → (FRED series id, +1 for direct or -1 for invert)
# Adding a new pair: register the series here and tests downstream pick it up.

FRED_FX_SERIES: dict[tuple[str, str], tuple[str, float]] = {
    ("USD", "KZT"): ("DEXKZUS", +1.0),
    ("KZT", "USD"): ("DEXKZUS", -1.0),   # inverted
}


FRED_API_ROOT       = "https://api.stlouisfed.org/fred/series/observations"
HTTP_TIMEOUT_SEC    = 8.0
HTTP_MAX_RETRIES    = 3
HTTP_BACKOFF_BASE   = 0.5
DEFAULT_DAYS        = 1900    # ≥ engine lookback (1825 кал.) + буфер: FX-серия
                              # ДОЛЖНА покрывать всё ценовое окно, иначе ранние
                              # даты мультивалютного портфеля не конвертируются →
                              # NaN → row-dropna усекает общее окно обратно.
DEFAULT_CACHE_DIR   = Path("/tmp/fx_cache")
_RETRY_STATUS       = frozenset({429, 500, 502, 503, 504})


# ── Low-level fetch ─────────────────────────────────────────────────────────

def _fred_fetch(series_id: str,
                api_key:   str,
                days:      int = DEFAULT_DAYS,
                http_get:  Optional[Callable] = None,
                sleep:     Optional[Callable[[float], None]] = None,
                ) -> list[dict]:
    """
    Pull a daily FRED series with retry on 429/5xx; return ``[{date, value}]``
    sorted oldest→newest.  Raises on terminal failure.
    """
    http_get = http_get or requests.get
    sleep    = sleep or time.sleep
    params = {
        "series_id":  series_id,
        "api_key":    api_key,
        "file_type":  "json",
        # Newest first, then we sort — same trick MacroFeed uses to avoid
        # FRED's "first 260 observations from 1977" gotcha.
        "sort_order": "desc",
        "limit":      max(int(days), 60),
    }
    resp = None
    for attempt in range(HTTP_MAX_RETRIES):
        resp = http_get(FRED_API_ROOT, params=params, timeout=HTTP_TIMEOUT_SEC)
        status = getattr(resp, "status_code", 200)
        if status in _RETRY_STATUS and attempt < HTTP_MAX_RETRIES - 1:
            delay = HTTP_BACKOFF_BASE * (2 ** attempt)
            logger.warning("FxFeed: %s → HTTP %s, retry %d/%d in %.1fs",
                           series_id, status, attempt + 1, HTTP_MAX_RETRIES, delay)
            sleep(delay)
            continue
        resp.raise_for_status()
        break
    body = resp.json() if resp is not None else {}
    if not isinstance(body, dict) or "observations" not in body:
        raise ValueError(f"FRED response missing 'observations' for {series_id}")
    out: list[dict] = []
    for ob in body.get("observations", []):
        raw = ob.get("value")
        d   = ob.get("date")
        try:
            v = float(raw) if raw not in ("", ".", None) else None
        except (TypeError, ValueError):
            v = None
        if v is not None and d:
            out.append({"date": d, "value": v})
    if not out:
        raise ValueError(f"FRED returned zero usable observations for {series_id}")
    out.sort(key=lambda r: r["date"])
    return out


# ── Public provider ─────────────────────────────────────────────────────────

class FredFxProvider:
    """
    FRED-backed FX provider, conforming to `currency.FxProvider`.

    Usage:
        from finance.investment_logic import MAC3RiskEngine
        from services.fx_feed import FredFxProvider
        engine = MAC3RiskEngine(reporting_currency="USD",
                                fx_provider=FredFxProvider())

    Behaviour:
      * Missing FRED_API_KEY → every call returns None (engine then
        skips conversion and logs a warning; no exception).
      * Unsupported pair (e.g. EUR↔CNY) → returns None.
      * Successful first call writes the raw observation list to disk
        cache at /tmp/fx_cache/<series>.json; subsequent process-local
        calls are memoised in-memory.
    """

    def __init__(self,
                 api_key:   Optional[str] = None,
                 days:      int = DEFAULT_DAYS,
                 cache_dir: Optional[Path] = None,
                 http_get:  Optional[Callable] = None,
                 sleep:     Optional[Callable[[float], None]] = None):
        self._api_key   = (api_key if api_key is not None
                            else os.getenv("FRED_API_KEY", "")).strip()
        self._days      = int(days)
        self._http_get  = http_get
        self._sleep     = sleep
        self._memo: dict[tuple[str, str], pd.Series] = {}

        root = Path(cache_dir or DEFAULT_CACHE_DIR)
        try:
            root.mkdir(parents=True, exist_ok=True)
            self._cache_dir: Optional[Path] = root
        except OSError as exc:
            logger.warning("FxFeed: cache dir %s not writable (%s); memory-only.",
                           root, exc)
            self._cache_dir = None

    def __call__(self, base: str, quote: str) -> Optional[pd.Series]:
        key = (str(base).upper(), str(quote).upper())
        if key[0] == key[1]:
            return None            # caller short-circuits same-ccy upstream
        if key in self._memo:
            return self._memo[key]
        if not self._api_key:
            logger.info("FxFeed: FRED_API_KEY not set — FX %s→%s unavailable", *key)
            return None
        spec = FRED_FX_SERIES.get(key)
        if spec is None:
            logger.info("FxFeed: no FRED series registered for %s→%s", *key)
            return None
        series_id, multiplier = spec
        obs = self._load_series(series_id)
        if not obs:
            return None
        # Build a tz-naive DatetimeIndex with float64 values.
        idx = pd.to_datetime([r["date"] for r in obs])
        vals = [float(r["value"]) for r in obs]
        s    = pd.Series(vals, index=idx, dtype=float).sort_index()
        if multiplier < 0:
            # KZT→USD = 1 / (KZT per USD).  Guard against zero divisor.
            s = s.replace(0.0, float("nan"))
            s = 1.0 / s
        self._memo[key] = s
        logger.info("FxFeed: loaded %s → %s, %d obs, last=%s @ %s",
                    *key, len(s), s.dropna().iloc[-1] if s.notna().any() else "n/a",
                    s.dropna().index[-1].date() if s.notna().any() else "n/a")
        return s

    # ── Internal ────────────────────────────────────────────────────────

    def _cache_path(self, series_id: str) -> Optional[Path]:
        return (self._cache_dir / f"{series_id}.json") if self._cache_dir else None

    def _load_series(self, series_id: str) -> Optional[list[dict]]:
        """Try fresh fetch first; fall back to disk cache on failure."""
        try:
            obs = _fred_fetch(series_id, self._api_key, self._days,
                              http_get=self._http_get, sleep=self._sleep)
        except Exception as exc:
            cache_p = self._cache_path(series_id)
            logger.warning("FxFeed: fetch %s failed (%s); trying disk cache.",
                           series_id, exc)
            if cache_p and cache_p.exists():
                try:
                    return json.loads(cache_p.read_text())
                except (OSError, ValueError) as cache_exc:
                    logger.warning("FxFeed: cache read failed for %s: %s",
                                   series_id, cache_exc)
            return None

        # Persist the fresh response.
        cache_p = self._cache_path(series_id)
        if cache_p is not None:
            try:
                cache_p.write_text(json.dumps(obs))
            except OSError as exc:
                logger.warning("FxFeed: cache write failed for %s: %s",
                               series_id, exc)
        return obs


def default_fx_provider() -> Optional[Callable]:
    """
    Factory used by `MAC3RiskEngine.__init__` when no explicit provider is
    passed.  Returns:
      * a `FredFxProvider` instance if `FRED_API_KEY` is in env, else
      * None — engine then falls into the "no FX data" graceful path
        (native-currency prices used, audit record marks fallback).

    The function is cheap and does NOT touch the network — fully safe
    to call at every engine instantiation.
    """
    if not (os.getenv("FRED_API_KEY") or "").strip():
        return None
    return FredFxProvider()


__all__ = [
    "FRED_FX_SERIES",
    "FredFxProvider",
    "default_fx_provider",
]
