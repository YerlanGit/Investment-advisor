"""
FRED macro data feed for the regime / drivers panel.

Pulls 4 macro time series from the St. Louis Fed (FRED) — all public-domain
and free.  Used to enrich the DEEP P5 "Рыночный режим" page with hard
macro signals beyond the ETF-based regime classifier:

  • T10Y2Y         10Y-2Y Treasury spread     (yield curve · growth axis)
  • BAMLH0A0HYM2   ICE BofA US HY OAS         (credit stress)
  • VIXCLS         CBOE VIX                   (volatility regime)
  • T10YIE         10Y Breakeven Inflation    (inflation impulse)

Note: the ISM Manufacturing PMI series (`NAPM`) was discontinued by FRED
(ISM licensing) and is no longer fetched — it returned HTTP 400 on every
run.  The regime classifier does not depend on it.

Design constraints
──────────────────
1. **Zero new dependencies.**  Uses `requests` (already in requirements.txt)
   for the HTTP call.  Parses raw JSON; no fredapi wrapper.
2. **Sklearn-free.**  Stays unit-testable in environments without the heavy
   engine deps.
3. **Network-isolated tests.**  All tests inject a mock HTTP callable; no
   real FRED calls are made in CI.
4. **Graceful degradation.**  Missing API key → returns None for every
   series with status="missing".  Network failure → falls back to cached
   data even if stale, status="stale".  Empty cache → status="error".
5. **Cloud-Run friendly.**  Default cache dir is /tmp/macro_cache (writable,
   ephemeral).  First call after a cold start re-fetches; subsequent calls
   within `ttl_hours` serve from cache.

API key location
────────────────
Read from `FRED_API_KEY` environment variable.  Local dev: add to `.env`
(gitignored).  Cloud Run: store as a secret and mount as env var.  Get a
free key at https://fred.stlouisfed.org/docs/api/api_key.html.

Output shape
────────────
Per series, the public method ``get_regime_drivers()`` returns:

    {
      "yield_curve_10y2y": {
          "series_id": "T10Y2Y",
          "value":          0.18,
          "as_of":          "2026-05-14",
          "fetched_at":     "2026-05-15T03:01:22Z",
          "freshness_days": 1,
          "status":         "ok" | "stale" | "missing" | "error",
          "unit":           "pp",
          "label":          "10Y−2Y Treasury spread",
          "history_30d":    [{"date": "...", "value": ...}, ...],
      },
      "hy_credit_spread":   {...},
      "vix":                {...},
      "breakeven_inflation":{...},
    }

Status semantics
────────────────
  ok      Last observation within freshness window AND value in sanity range.
  stale   Cache served but FRED is unreachable, OR last observation older
           than the per-series freshness window (FRED publishing lag).
  missing No FRED_API_KEY in env (or empty).
  error   First-ever fetch failed and no cache exists.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests


logger = logging.getLogger("MacroFeed")


# ── Series catalog ──────────────────────────────────────────────────────────
# Each entry pins (FRED series id, human label, unit, freshness window in
# CALENDAR days, sanity range).  Sanity is permissive — meant to catch
# outright corrupt feeds, not to flag normal market moves.

@dataclass(frozen=True)
class SeriesSpec:
    key:                  str          # short key used in the output dict
    series_id:            str          # FRED series id
    label:                str
    unit:                 str          # "pp" | "bps" | "index" | "%"
    freshness_calendar_days: int       # status="stale" if older
    sanity_range:         tuple[float, float]
    publish_cadence:      str          # "daily" | "monthly"


MACRO_SERIES_CATALOG: list[SeriesSpec] = [
    SeriesSpec(
        key                     = "yield_curve_10y2y",
        series_id               = "T10Y2Y",
        label                   = "10Y−2Y Treasury spread",
        unit                    = "pp",
        freshness_calendar_days = 5,    # daily; 5 cal days = ≈3 trading
        sanity_range            = (-5.0, 5.0),
        publish_cadence         = "daily",
    ),
    SeriesSpec(
        key                     = "hy_credit_spread",
        series_id               = "BAMLH0A0HYM2",
        label                   = "US HY OAS (ICE BofA)",
        unit                    = "pp",                # FRED reports in %, =bps/100
        freshness_calendar_days = 5,
        sanity_range            = (0.5, 30.0),         # 50 bps … 3000 bps
        publish_cadence         = "daily",
    ),
    SeriesSpec(
        key                     = "vix",
        series_id               = "VIXCLS",
        label                   = "CBOE VIX",
        unit                    = "index",
        freshness_calendar_days = 5,
        sanity_range            = (5.0, 120.0),
        publish_cadence         = "daily",
    ),
    SeriesSpec(
        key                     = "breakeven_inflation",
        series_id               = "T10YIE",
        label                   = "10Y Breakeven Inflation Rate",
        unit                    = "%",
        freshness_calendar_days = 5,
        sanity_range            = (-1.0, 10.0),
        publish_cadence         = "daily",
    ),
]


FRED_API_ROOT     = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_TTL_HOURS = 12
DEFAULT_CACHE_DIR = Path("/tmp/macro_cache")

# FRED echoes the api_key inside HTTP error URLs (e.g. requests' HTTPError
# message).  That string is surfaced in the CoVe data-lineage panel of the
# report, so it must never carry the secret.  Redact any api_key=... token
# before an error message is stored or logged.
_SECRET_RE = re.compile(r"(api_key=)[^&\s\"']+", re.IGNORECASE)


def _sanitize_secret(text: object) -> str:
    """Strip FRED api_key values out of an arbitrary message/URL string."""
    return _SECRET_RE.sub(r"\1***", str(text))


HISTORY_KEEP_DAYS = 90        # local cache trims to this window
HTTP_TIMEOUT_SEC  = 8.0


# ── Helpers ─────────────────────────────────────────────────────────────────

def _format_iso(dt: datetime) -> str:
    """Stable ISO-8601 'Z' format used for cache timestamps."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(v) -> Optional[float]:
    try:
        if v in ("", None, "."):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _calendar_days_between(iso_date: str, ref: Optional[datetime] = None) -> int:
    """Return absolute calendar-day difference (ref defaults to today UTC)."""
    try:
        d = datetime.fromisoformat(iso_date).date()
    except (TypeError, ValueError):
        return 99999
    today = (ref or datetime.now(timezone.utc)).date()
    return abs((today - d).days)


# ── Cache layer ─────────────────────────────────────────────────────────────

class _DiskCache:
    """Tiny disk-backed JSON cache, one file per FRED series id."""

    def __init__(self, root: Path):
        self._root = Path(root)
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("MacroFeed: cache dir %s not writable (%s); "
                            "running in memory-only mode.", self._root, exc)
            self._root = None  # type: ignore[assignment]

    def _path(self, series_id: str) -> Optional[Path]:
        if self._root is None:
            return None
        return self._root / f"{series_id}.json"

    def read(self, series_id: str) -> Optional[dict]:
        p = self._path(series_id)
        if p is None or not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("MacroFeed: cache read failed for %s: %s", series_id, exc)
            return None

    def write(self, series_id: str, payload: dict) -> None:
        p = self._path(series_id)
        if p is None:
            return
        try:
            tmp = p.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            tmp.replace(p)             # atomic on POSIX
        except OSError as exc:
            logger.warning("MacroFeed: cache write failed for %s: %s", series_id, exc)


# ── Public class ────────────────────────────────────────────────────────────

class MacroFeed:
    """
    Pulls macro time-series from FRED with a transparent disk cache.

    Args:
        api_key   : FRED API key. If None (or empty), all calls return
                      status="missing" without hitting the network.
        cache_dir : Where to keep per-series JSON files.  Defaults to
                      /tmp/macro_cache (Cloud-Run-friendly).
        ttl_hours : Cache freshness window for the FETCH itself.  Reaching
                      this threshold triggers a refetch; serving stale cache
                      on network failure still works regardless.
        http_get  : Override for testing — should mimic ``requests.get``
                      semantics (return an object with .raise_for_status(),
                      .json(), and .status_code).
        now       : Override for tests that need a fixed clock.
    """

    def __init__(self,
                  api_key:   Optional[str]   = None,
                  cache_dir: Optional[Path]  = None,
                  ttl_hours: float           = DEFAULT_TTL_HOURS,
                  http_get:  Optional[Callable] = None,
                  now:       Optional[Callable[[], datetime]] = None,
                  catalog:   Optional[list[SeriesSpec]] = None,
                  ):
        self._api_key   = (api_key if api_key is not None
                            else os.getenv("FRED_API_KEY", "")).strip()
        self._cache     = _DiskCache(Path(cache_dir or DEFAULT_CACHE_DIR))
        self._ttl_sec   = float(ttl_hours) * 3600.0
        self._get       = http_get or requests.get
        self._now       = now or (lambda: datetime.now(timezone.utc))
        self._catalog   = list(catalog or MACRO_SERIES_CATALOG)

    # ── Public API ──────────────────────────────────────────────────────

    def get_regime_drivers(self) -> dict:
        """Fetch the full 5-series macro pack for the regime page."""
        out: dict[str, dict] = {}
        for spec in self._catalog:
            out[spec.key] = self._get_one(spec)
        return out

    # ── Internal ────────────────────────────────────────────────────────

    def _get_one(self, spec: SeriesSpec) -> dict:
        """
        Resolve one series via cache + (maybe) FRED HTTP.
        Always returns a dict with stable keys.
        """
        if not self._api_key:
            return self._empty_result(spec, status="missing",
                                       note="FRED_API_KEY not set in env")

        cached = self._cache.read(spec.series_id)
        cache_fresh = cached and self._cache_fresh(cached)

        if cache_fresh:
            return self._format_from_cache(spec, cached, status_override=None)

        # Need to fetch.
        try:
            fetched = self._http_fetch(spec)
            payload = {
                "series_id":   spec.series_id,
                # Use the injected clock so tests can fix the timestamp.
                "fetched_at":  _format_iso(self._now()),
                "observations": fetched[-HISTORY_KEEP_DAYS:],   # trim
            }
            self._cache.write(spec.series_id, payload)
            return self._format_from_cache(spec, payload, status_override=None)
        except Exception as exc:
            # Network or parse error — fall back to whatever cache we have.
            safe_exc = _sanitize_secret(exc)
            if cached:
                logger.warning("MacroFeed: refresh failed for %s (%s); serving stale cache.",
                                spec.series_id, safe_exc)
                return self._format_from_cache(spec, cached, status_override="stale",
                                                 note=f"refresh failed: {safe_exc}")
            logger.warning("MacroFeed: first-ever fetch failed for %s: %s",
                            spec.series_id, safe_exc)
            return self._empty_result(spec, status="error", note=safe_exc)

    def _cache_fresh(self, cached: dict) -> bool:
        ts = cached.get("fetched_at")
        if not ts:
            return False
        try:
            t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return False
        return (self._now() - t).total_seconds() < self._ttl_sec

    def _http_fetch(self, spec: SeriesSpec) -> list[dict]:
        """Hit FRED and return [{date, value}, ...] (chronological, oldest→newest)."""
        params = {
            "series_id":    spec.series_id,
            "api_key":      self._api_key,
            "file_type":    "json",
            # Newest observations first — ``sort_order=asc`` + ``limit=260``
            # used to return FRED's FIRST 260 datapoints (i.e. the start of
            # the series, ~1977 for T10Y2Y), which then drove freshness to
            # decades-old "as_of" stamps.  Pull newest N, then reverse to
            # restore the chronological ordering downstream code assumes.
            "sort_order":   "desc",
            "limit":        HISTORY_KEEP_DAYS,
        }
        resp = self._get(FRED_API_ROOT, params=params, timeout=HTTP_TIMEOUT_SEC)
        resp.raise_for_status()
        body = resp.json()
        if not isinstance(body, dict) or "observations" not in body:
            raise ValueError(f"FRED response missing 'observations': {body!r}")

        out: list[dict] = []
        for ob in body.get("observations", []):
            v = _safe_float(ob.get("value"))
            d = ob.get("date")
            if v is not None and d:
                out.append({"date": d, "value": v})
        if not out:
            raise ValueError("FRED returned zero usable observations")
        # Always re-sort chronologically so obs[-1] is the freshest point
        # regardless of how FRED actually ordered the response.  Cheap
        # (len ≤ HISTORY_KEEP_DAYS) and removes one whole class of bug.
        out.sort(key=lambda r: r["date"])
        return out

    def _format_from_cache(self, spec: SeriesSpec, cached: dict,
                             *, status_override: Optional[str] = None,
                             note: str = "") -> dict:
        obs = cached.get("observations") or []
        if not obs:
            return self._empty_result(spec, status="error",
                                       note="cache had no observations")
        last = obs[-1]
        value = _safe_float(last.get("value"))
        as_of = str(last.get("date") or "")
        freshness = _calendar_days_between(as_of, self._now())

        # Determine status.
        in_range = (value is not None and
                    spec.sanity_range[0] <= value <= spec.sanity_range[1])
        if status_override:
            status = status_override
        elif value is None:
            status = "error"
        elif not in_range:
            status = "stale"           # outside sanity → degrade visibly
        elif freshness > spec.freshness_calendar_days:
            status = "stale"
        else:
            status = "ok"

        return {
            "series_id":      spec.series_id,
            "value":          value,
            "as_of":          as_of,
            "fetched_at":     cached.get("fetched_at"),
            "freshness_days": int(freshness),
            "status":         status,
            "unit":           spec.unit,
            "label":          spec.label,
            "publish_cadence": spec.publish_cadence,
            "history_30d":    obs[-30:],
            "note":           note,
        }

    def _empty_result(self, spec: SeriesSpec, *, status: str, note: str = "") -> dict:
        return {
            "series_id":      spec.series_id,
            "value":          None,
            "as_of":          None,
            "fetched_at":     None,
            "freshness_days": None,
            "status":         status,
            "unit":           spec.unit,
            "label":          spec.label,
            "publish_cadence": spec.publish_cadence,
            "history_30d":    [],
            "note":           note,
        }


__all__ = [
    "MacroFeed",
    "SeriesSpec",
    "MACRO_SERIES_CATALOG",
    "FRED_API_ROOT",
    "DEFAULT_TTL_HOURS",
    "HISTORY_KEEP_DAYS",
]
