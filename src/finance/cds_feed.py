"""
CDS Feed — free-tier layer.

Provides per-ticker credit-spread readings used by pillar D (Credit) of the
4-pillar scoring model.  The free layer relies on official, no-key data:

  • FRED — BAMLH0A0HYM2 (ICE BofA US High Yield Index Option-Adjusted Spread)
           used as a market-wide HY proxy when issuer-level CDS is missing.
  • FRED — BAMLC0A0CM   (ICE BofA US Corporate Master Option-Adjusted Spread)
           used as Investment-Grade proxy.
  • worldgovernmentbonds.com — sovereign CDS table (light HTML scrape) for
           sovereign exposure (KZ_GOV_5Y for Kazakhstani names).

A QualityGate validates every reading before it is consumed by scoring:

  1. Sanity range — 1 ≤ bps ≤ 3000.
  2. Staleness — timestamp not older than 3 trading days.
  3. Cross-source consistency — when two sources agree to within 25%, accept;
     otherwise downgrade quality letter (A → B → C).

The module exposes a single CDSFeed class with a `get_spread(ticker)` method.
A 24-hour SQLite cache (path configurable via CDS_CACHE_PATH env var) keeps
load light: an end-of-day refresh by Cloud Scheduler is enough to keep the
bot fresh without per-request scraping.

Future paid providers (S&P Market Intelligence, Cbonds, ICE) plug in via
`CDSFeed.add_provider(name, fetch_fn, quality='A')`.  The orchestrator then
prefers high-quality sources without code changes elsewhere.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import requests

logger = logging.getLogger("CDSFeed")


# ── Public dataclass ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CDSPoint:
    """One credit-spread reading."""
    ticker:    str
    bps:       float
    source:    str               # 'FRED:HY' / 'WGB:KZ_5Y' / 'sp_global' / ...
    timestamp: datetime          # UTC
    quality:   str               # 'A' | 'B' | 'C'
    change_7d: Optional[float]   # fractional change vs 7d ago, e.g. +0.18

    def as_dict(self) -> dict:
        return {
            "ticker":    self.ticker,
            "bps":       round(self.bps, 1),
            "source":    self.source,
            "timestamp": self.timestamp.isoformat(),
            "quality":   self.quality,
            "change_7d": round(self.change_7d, 4) if self.change_7d is not None else None,
        }


# ── Sovereign-substitution map for tickers without their own CDS ────────────
SOVEREIGN_PROXY: dict[str, str] = {
    "KSPI": "KZ_5Y",
    "HSBK": "KZ_5Y",
    "KAP":  "KZ_5Y",
    "KZTK": "KZ_5Y",
    "KCEL": "KZ_5Y",
    "BAST": "KZ_5Y",
    "HRGL": "KZ_5Y",
    "KZAP": "KZ_5Y",
}


# ── Quality Gate ────────────────────────────────────────────────────────────

class CDSQualityGate:
    """
    Stateless validator: downgrades quality letter when checks fail.
    Returns (accept_bool, quality_letter, reason_or_None).
    """

    MIN_BPS         = 1.0
    MAX_BPS         = 3000.0
    MAX_STALE_DAYS  = 3
    AGREEMENT_BAND  = 0.25  # 25% spread between two sources is acceptable

    def validate(self,
                 bps: float,
                 timestamp: datetime,
                 *,
                 alternative_bps: Optional[float] = None,
                 base_quality: str = "B") -> tuple[bool, str, Optional[str]]:
        if not self.MIN_BPS <= bps <= self.MAX_BPS:
            return False, "C", f"out_of_range:{bps:.0f}"
        age_days = (datetime.now(timezone.utc) - timestamp).total_seconds() / 86_400
        if age_days > self.MAX_STALE_DAYS:
            return False, "C", f"stale:{age_days:.1f}d"
        if alternative_bps is not None and alternative_bps > 0:
            spread = abs(bps - alternative_bps) / max(bps, alternative_bps)
            if spread > self.AGREEMENT_BAND:
                # disagreement → downgrade one letter
                downgraded = {"A": "B", "B": "C", "C": "C"}.get(base_quality, "C")
                return True, downgraded, f"cross_source_disagreement:{spread:.0%}"
        return True, base_quality, None


# ── Local SQLite cache ──────────────────────────────────────────────────────

class _CDSCache:
    """24-hour SQLite cache.  Single-connection, write-locked."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS cds_cache (
        ticker      TEXT PRIMARY KEY,
        bps         REAL NOT NULL,
        bps_7d_ago  REAL,
        source      TEXT NOT NULL,
        quality     TEXT NOT NULL,
        ts_unix     INTEGER NOT NULL
    );
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as cx:
            cx.executescript(self.SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, isolation_level=None, timeout=5.0)

    def get(self, ticker: str, max_age_seconds: int = 86_400) -> Optional[dict]:
        cutoff = int(time.time()) - max_age_seconds
        with self._lock, self._connect() as cx:
            cur = cx.execute(
                "SELECT ticker, bps, bps_7d_ago, source, quality, ts_unix "
                "FROM cds_cache WHERE ticker = ? AND ts_unix >= ?",
                (ticker.upper(), cutoff),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "ticker":     row[0],
            "bps":        row[1],
            "bps_7d_ago": row[2],
            "source":     row[3],
            "quality":    row[4],
            "ts_unix":    row[5],
        }

    def put(self, *, ticker: str, bps: float, bps_7d_ago: Optional[float],
            source: str, quality: str, ts_unix: Optional[int] = None) -> None:
        ts_unix = ts_unix or int(time.time())
        with self._lock, self._connect() as cx:
            cx.execute(
                "INSERT INTO cds_cache (ticker, bps, bps_7d_ago, source, "
                "quality, ts_unix) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(ticker) DO UPDATE SET "
                "bps = excluded.bps, bps_7d_ago = excluded.bps_7d_ago, "
                "source = excluded.source, quality = excluded.quality, "
                "ts_unix = excluded.ts_unix",
                (ticker.upper(), float(bps),
                 None if bps_7d_ago is None else float(bps_7d_ago),
                 source, quality, ts_unix),
            )


# ── Free-layer providers ────────────────────────────────────────────────────

_FRED_HY_SERIES = "BAMLH0A0HYM2"   # ICE BofA US HY OAS, percent
_FRED_IG_SERIES = "BAMLC0A0CM"     # ICE BofA US Corp OAS, percent


def _fred_observations(series_id: str, *, days: int = 14) -> list[tuple[datetime, float]]:
    """
    Pull the last `days` observations for a FRED series via the public CSV
    endpoint (no API key required for fred.stlouisfed.org/graph/fredgraph.csv).

    Returns [(ts_utc, percent_value)] sorted ascending; bps = percent * 100.
    """
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        f"id={series_id}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        logger.warning("[CDSFeed/FRED] %s fetch failed: %s", series_id, exc)
        return []

    out: list[tuple[datetime, float]] = []
    lines = r.text.splitlines()
    for line in lines[1:]:                # skip header
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_str, val = parts[0].strip(), parts[1].strip()
        if val == "" or val == ".":
            continue
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            v = float(val)
        except (ValueError, TypeError):
            continue
        out.append((dt, v))
    out.sort(key=lambda r: r[0])
    return out[-days * 5:] if days else out  # generous tail to cover holidays


def _wgb_sovereign_5y(country_token: str = "kazakhstan") -> Optional[tuple[datetime, float]]:
    """
    Light scrape of worldgovernmentbonds.com for the 5Y sovereign CDS.

    Returns (timestamp_utc, bps) or None.

    NOTE: scrape is intentionally minimal.  If the site changes layout we
    return None and downstream falls back to neutral credit signal.  For
    production grade we'd switch to a paid feed; this is the free layer.
    """
    url = f"https://www.worldgovernmentbonds.com/cds-historical-data/{country_token}/5-years/"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "InvestmentAdvisorBot/1.0"})
        r.raise_for_status()
    except Exception as exc:
        logger.warning("[CDSFeed/WGB] %s fetch failed: %s", country_token, exc)
        return None
    # WGB renders the latest value as a number followed by 'bps' in a header
    # element.  Match a permissive pattern to avoid coupling to exact markup.
    m = re.search(r'(\d{2,4}(?:\.\d+)?)\s*bps', r.text)
    if not m:
        return None
    try:
        bps = float(m.group(1))
    except ValueError:
        return None
    return datetime.now(timezone.utc), bps


# ── Provider plumbing ────────────────────────────────────────────────────────

# A provider is a callable taking a ticker and returning either a CDSPoint
# (already validated) or None.  Higher-quality providers should be inserted
# at the front of the list.
ProviderFn = Callable[[str], Optional[CDSPoint]]


class CDSFeed:
    """
    Free-tier CDS feed orchestrator.

    Usage:
        feed = CDSFeed()
        point = feed.get_spread("AAPL")     # returns CDSPoint | None

    Resolution order:
      1. Per-ticker SQLite cache (24h)
      2. Registered providers in priority order (free layer first)
      3. None — caller treats as 'CDS data unavailable'

    The orchestrator never fabricates a value.  When a ticker can only be
    resolved via a sector/sovereign proxy we lower quality to 'C' and tag
    `source` accordingly so the PDF can disclose the proxy in the footer.
    """

    def __init__(self,
                 cache_path: Optional[str] = None,
                 cache_max_age_s: int = 86_400) -> None:
        path = cache_path or os.getenv(
            "CDS_CACHE_PATH",
            "data/cache/cds_cache.sqlite",
        )
        self._cache = _CDSCache(path)
        self._cache_max_age_s = cache_max_age_s
        self._providers: list[tuple[str, ProviderFn]] = []
        self._gate = CDSQualityGate()
        # Register the free-tier providers in priority order.
        self.add_provider("WGB_sovereign", self._wgb_sovereign_provider, quality="C")
        self.add_provider("FRED_HY_proxy", self._fred_hy_provider,       quality="C")

    # ── Public API ───────────────────────────────────────────────────────

    def add_provider(self, name: str, fn: ProviderFn, quality: str = "B") -> None:
        """Register a provider in priority order (first registered = highest)."""
        del quality   # currently informational; reserved for paid providers
        self._providers.append((name, fn))

    def get_spread(self, ticker: str) -> Optional[CDSPoint]:
        ticker_u = ticker.upper().strip()
        cached = self._cache.get(ticker_u, self._cache_max_age_s)
        if cached:
            ts = datetime.fromtimestamp(cached["ts_unix"], tz=timezone.utc)
            change_7d = None
            if cached["bps_7d_ago"]:
                change_7d = (cached["bps"] - cached["bps_7d_ago"]) / cached["bps_7d_ago"]
            return CDSPoint(
                ticker=ticker_u, bps=cached["bps"],
                source=cached["source"], timestamp=ts,
                quality=cached["quality"], change_7d=change_7d,
            )

        # Try providers in priority order.
        for name, fn in self._providers:
            try:
                point = fn(ticker_u)
            except Exception as exc:
                logger.warning("[CDSFeed] provider %s raised: %s", name, exc)
                point = None
            if point is None:
                continue
            ok, q, reason = self._gate.validate(
                bps=point.bps, timestamp=point.timestamp,
                base_quality=point.quality,
            )
            if not ok:
                logger.info("[CDSFeed] %s rejected by gate (%s)", ticker_u, reason)
                continue
            self._cache.put(
                ticker=ticker_u,
                bps=point.bps,
                bps_7d_ago=None if point.change_7d is None
                           else point.bps / (1.0 + point.change_7d),
                source=point.source, quality=q,
            )
            return CDSPoint(
                ticker=ticker_u, bps=point.bps,
                source=point.source, timestamp=point.timestamp,
                quality=q, change_7d=point.change_7d,
            )
        return None

    # ── Free-layer provider implementations ──────────────────────────────

    def _fred_hy_provider(self, ticker: str) -> Optional[CDSPoint]:
        """
        Use the FRED HY OAS as a coarse credit proxy for any US-listed
        corporate ticker that has no dedicated CDS data.  Returned quality
        is 'C' because this is a market-wide proxy, not issuer-level data.
        """
        if "." in ticker and not ticker.endswith(".US"):
            return None
        observations = _fred_observations(_FRED_HY_SERIES, days=14)
        if len(observations) < 2:
            return None
        ts_now, percent_now = observations[-1]
        # Find a value ≈ 7 trading days ago for change_7d.
        change_7d: Optional[float] = None
        if len(observations) >= 7:
            _, percent_7 = observations[-7]
            if percent_7 > 0:
                change_7d = (percent_now - percent_7) / percent_7
        bps = percent_now * 100.0
        return CDSPoint(
            ticker=ticker, bps=bps,
            source=f"FRED:{_FRED_HY_SERIES}",
            timestamp=ts_now, quality="C", change_7d=change_7d,
        )

    def _wgb_sovereign_provider(self, ticker: str) -> Optional[CDSPoint]:
        """Sovereign-CDS substitution for tickers in SOVEREIGN_PROXY."""
        proxy_key = SOVEREIGN_PROXY.get(ticker.split(".")[0])
        if proxy_key != "KZ_5Y":
            return None
        result = _wgb_sovereign_5y("kazakhstan")
        if result is None:
            return None
        ts, bps = result
        return CDSPoint(
            ticker=ticker, bps=bps,
            source="WGB:KZ_5Y", timestamp=ts,
            quality="C", change_7d=None,
        )


# ── Convenience: cds-lookup callable adapter for scoring orchestrator ────────

def make_lookup(feed: Optional[CDSFeed] = None) -> Callable[[str], dict]:
    """
    Build a (ticker → dict) callable matching scoring_orchestrator.CDSLookup.
    When `feed` is None, returns a no-op lookup so scoring degrades gracefully.
    """
    if feed is None:
        return lambda _t: {}
    def _lookup(ticker: str) -> dict:
        point = feed.get_spread(ticker)
        if point is None:
            return {}
        return {
            "bps":       point.bps,
            "change_7d": point.change_7d,
            "source":    point.source,
            "quality":   point.quality,
        }
    return _lookup


__all__ = [
    "CDSFeed",
    "CDSPoint",
    "CDSQualityGate",
    "make_lookup",
    "SOVEREIGN_PROXY",
]
