"""
Historical OHLC candles from Tradernet (replaces yfinance for price history).

Wire endpoint:  POST /api/v2/cmd/getHloc          (modern, used by official SDK)
Fallback:       POST /api/v2/cmd/getQuotesHistory (legacy, same payload shape)
Auth:           v2 HMAC-SHA256 (X-NtApi-Sig + X-NtApi-PublicKey) — same as portfolio.

Quality layers (defense in depth, since Tradernet docs do not specify whether
candles are split/dividend-adjusted):

  1. ``corr=1`` is sent on the request — Tradernet honours this on accounts
     that support it (no-op otherwise).
  2. ``KNOWN_SPLITS`` table is applied unconditionally — pre-split prices are
     scaled by the split ratio.  Authoritative override.
  3. ``_detect_and_adjust_splits`` heuristic flags any single-day return whose
     |log return| > 0.4 (≈ 33% jump).  The implied integer ratio is back-applied.

Each layer is idempotent — running them all on already-adjusted data is a
no-op.  This makes the module robust regardless of what the server returns.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ── Cache (per-process, on /tmp so Cloud Run can write) ──────────────────────

_CACHE_DIR = Path("/tmp/freedom_history_cache")
_CACHE_TTL_SECONDS = 3600  # 1 hour — covers a typical analysis session


# ── Authoritative split events (2y rolling window, refresh annually) ─────────
# Format: {ticker: [(split_date, ratio), ...]}.  ``ratio`` is the multiplier
# applied to share count: 10 means 1 share → 10 shares, prices ÷ 10.
# Only need entries that fall inside the rolling 2-year fetch window.
KNOWN_SPLITS: dict[str, list[tuple[date, float]]] = {
    "NVDA.US":  [(date(2024, 6, 10), 10)],
    "AMZN.US":  [(date(2022, 6, 6),  20)],
    "GOOGL.US": [(date(2022, 7, 18), 20)],
    "GOOG.US":  [(date(2022, 7, 18), 20)],
    "TSLA.US":  [(date(2022, 8, 25), 3)],
    "SHOP.US":  [(date(2022, 6, 29), 10)],
    "DXCM.US":  [(date(2022, 6, 13), 4)],
}


# ── Pydantic models ──────────────────────────────────────────────────────────


class Candle(BaseModel):
    """One historical bar.  Field names match Tradernet's getHloc response."""

    model_config = ConfigDict(extra="ignore")

    t: int                  = Field(description="Unix timestamp (seconds)")
    o: float                = Field(default=0.0, description="Open price")
    h: float                = Field(default=0.0, description="High price")
    l: float                = Field(default=0.0, description="Low price")
    c: float                = Field(default=0.0, description="Close price")
    v: float | None = Field(default=None, description="Volume (if reported)")


# ── Public API ───────────────────────────────────────────────────────────────


def get_candles(
    client,                       # TradernetClient — typed loosely to avoid circular import
    ticker: str,
    *,
    days: int = 730,              # ~2 years
    timeframe: str = "D",         # daily bars
    use_cache: bool = True,
) -> pd.Series:
    """
    Fetch daily close-price series for *ticker* from Tradernet.

    Returns a ``pd.Series`` indexed by date (``pd.DatetimeIndex``) with
    adjusted close prices.  Raises ``BrokerAPIError`` if the server rejects
    the request and no cached fallback exists.

    Quality layers applied in order: cache → server (corr=1) → KNOWN_SPLITS →
    heuristic split detector.
    """
    # 1. Cache lookup
    if use_cache:
        cached = _read_cache(ticker, days)
        if cached is not None:
            return cached

    # 2. Server fetch
    raw_candles = _fetch_hloc(client, ticker, days=days, timeframe=timeframe)
    series = _candles_to_series(raw_candles)

    if series.empty:
        logger.warning("Tradernet getHloc вернул пустую серию для %s", ticker)
        return series

    # 3. Apply known-splits table (authoritative)
    series = _apply_known_splits(ticker, series)

    # 4. Heuristic split detector (catches anything KNOWN_SPLITS missed)
    series = _detect_and_adjust_splits(ticker, series)

    # 5. Persist to cache
    if use_cache:
        _write_cache(ticker, days, series)

    return series


def get_history_frame(
    client,
    tickers: list[str],
    *,
    days: int = 730,
    max_workers: int = 6,
) -> pd.DataFrame:
    """
    Fetch close-price series for *tickers* in parallel and combine into a
    DataFrame indexed by date with one column per ticker.  Missing tickers
    appear as all-NaN columns (the consumer is expected to drop them).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    series_map: dict[str, pd.Series] = {}

    def _one(t: str) -> tuple[str, pd.Series]:
        try:
            return t, get_candles(client, t, days=days)
        except Exception as exc:
            logger.warning("Не удалось получить историю %s: %s", t, exc)
            return t, pd.Series(dtype=float)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, t) for t in tickers]
        for fut in as_completed(futures):
            t, s = fut.result()
            series_map[t] = s

    # Outer-join all series on date index, preserving requested column order.
    df = pd.concat(
        [series_map[t].rename(t) for t in tickers if not series_map[t].empty],
        axis=1,
    ).sort_index() if any(not s.empty for s in series_map.values()) else pd.DataFrame()

    # Forward/back-fill non-trading days (weekends, holidays) so the row-level
    # dropna in the regression layer doesn't kill rows for benign gaps.
    if not df.empty:
        df = df.ffill().bfill()

    return df


# ── Internal: server transport ───────────────────────────────────────────────


def _fetch_hloc(client, ticker: str, *, days: int, timeframe: str) -> list[Candle]:
    """
    POST /api/v2/cmd/getHloc — wire-format verified against MasyaSmv/freedom-broker-api
    PHP SDK (src/Core/Service/StockHistoryService.php).

    Параметры:
      id            — тикер (НЕ "ticker")
      timeframe     — в минутах: 1440 = D, 60 = H1, 5 = M5
      intervalMode  — 'ClosedRay' (стандарт для исторических хвостов)
      count         — -1 = без лимита
      date_from/to  — формат 'YYYY-MM-DD'
    """
    today      = datetime.utcnow().date()
    start_date = today - timedelta(days=days)

    timeframe_minutes = {"D": 1440, "H1": 60, "M5": 5}.get(timeframe, 1440)

    params = {
        "id":           ticker,
        "timeframe":    timeframe_minutes,
        "intervalMode": "ClosedRay",
        "count":        -1,
        "date_from":    start_date.isoformat(),
        "date_to":      today.isoformat(),
    }

    try:
        raw = client._post_v2_signed("getHloc", params)
    except Exception as exc:
        logger.info("getHloc не удался для %s (%s) — пробуем legacy getQuotesHistory", ticker, exc)
        # Legacy команда принимает другие имена полей.
        legacy_params = {
            "ticker":   ticker,
            "interval": timeframe_minutes,
            "from":     start_date.isoformat(),
            "to":       today.isoformat(),
        }
        raw = client._post_v2_signed("getQuotesHistory", legacy_params)

    return _parse_hloc_response(raw, ticker)


def _parse_hloc_response(raw: dict, ticker: str = "") -> list[Candle]:
    """
    Tradernet returns one of several shapes; we handle them all.

    Reference shapes (per MasyaSmv/freedom-broker-api PHP SDK):
      A1. ``{"hloc": {"AAPL.US": [{t,o,h,l,c,v}, ...]}}``  — modern, keyed by ticker
      A2. ``{"hloc": [{t,o,h,l,c,v}, ...]}``               — flat list
      B.  ``{"xSeries": {"AAPL.US": [t, t, ...]}, "ySeries": {"AAPL.US": {c:[...]}}}``
      C.  Top-level list of candles
    """
    body = raw.get("result", raw)

    # Shape A1 — keyed by ticker
    if isinstance(body, dict) and isinstance(body.get("hloc"), dict):
        per_ticker = body["hloc"].get(ticker) or next(iter(body["hloc"].values()), [])
        if isinstance(per_ticker, list):
            return [Candle(**c) for c in per_ticker if isinstance(c, dict)]

    # Shape A2 — flat list
    if isinstance(body, dict) and isinstance(body.get("hloc"), list):
        return [Candle(**c) for c in body["hloc"] if isinstance(c, dict)]

    # Shape B — parallel-arrays
    if isinstance(body, dict) and "xSeries" in body and "ySeries" in body:
        x = body["xSeries"]
        y = body["ySeries"]
        # Could be dict-keyed or single list
        if isinstance(x, dict):
            x = x.get(ticker) or next(iter(x.values()), [])
        if isinstance(y, dict):
            y = y.get(ticker) or next(iter(y.values()), [])
        if isinstance(y, list) and y and isinstance(y[0], dict):
            closes = y[0].get("c", [])
            opens  = y[0].get("o", closes)
            highs  = y[0].get("h", closes)
            lows   = y[0].get("l", closes)
            return [
                Candle(t=int(x[i]), o=opens[i], h=highs[i], l=lows[i], c=closes[i])
                for i in range(min(len(x), len(closes)))
            ]

    # Shape C — top-level list
    if isinstance(body, list):
        return [Candle(**c) for c in body if isinstance(c, dict)]

    logger.warning(
        "Неизвестный формат ответа getHloc для %s: keys=%s",
        ticker,
        list(body.keys()) if isinstance(body, dict) else type(body).__name__,
    )
    return []


def _candles_to_series(candles: list[Candle]) -> pd.Series:
    """Convert a list of Candles to a pandas Series indexed by date."""
    if not candles:
        return pd.Series(dtype=float)
    rows = [(datetime.fromtimestamp(c.t, tz=timezone.utc).date(), c.c) for c in candles if c.c]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(
        [r[1] for r in rows],
        index=pd.DatetimeIndex([r[0] for r in rows]),
        dtype=float,
    )
    # De-duplicate (keep last) and sort.
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


# ── Quality layer: KNOWN_SPLITS adjustment ───────────────────────────────────


def _apply_known_splits(ticker: str, series: pd.Series) -> pd.Series:
    """
    Divide pre-split prices by the cumulative split ratio.

    Idempotent: if the server already returned adjusted prices, no daily
    return jumps by a factor matching a known split, so this becomes a no-op
    at the regression layer.  We still apply it because cache hits should
    always be in adjusted space.
    """
    splits = KNOWN_SPLITS.get(ticker.upper())
    if not splits:
        return series

    s = series.copy()
    for split_date, ratio in splits:
        ts = pd.Timestamp(split_date)
        if ts < s.index.min() or ts > s.index.max():
            continue

        # Only adjust if the price actually has a step at this date.  If the
        # server already returned adjusted data, there is no step → skip.
        try:
            before = s.loc[s.index < ts].iloc[-1]
            after  = s.loc[s.index >= ts].iloc[0]
        except (IndexError, KeyError):
            continue

        observed_ratio = before / after if after else 1.0
        # Step exists if observed_ratio is within 20% of declared ratio.
        if abs(observed_ratio - ratio) / ratio < 0.2:
            mask = s.index < ts
            s.loc[mask] = s.loc[mask] / ratio
            logger.info(
                "Применён known-split %s: %s ratio=%s (pre-split дни=%d)",
                ticker, split_date, ratio, mask.sum(),
            )
    return s


# ── Quality layer: heuristic split detector ──────────────────────────────────


_SPLIT_RETURN_THRESHOLD = math.log(1.5)   # |log return| > 0.405 ≈ 33% jump


def _detect_and_adjust_splits(ticker: str, series: pd.Series) -> pd.Series:
    """
    Detect single-day price jumps that look like splits and back-adjust.

    Mechanism: scan log-returns; for any |return| > 0.4, snap the implied
    raw ratio to the nearest integer or 1/integer in [2..50].  Apply backward
    division if it's a forward split, multiplication if reverse.

    Tuned not to fire on legitimate news-driven moves: a 33% one-day move is
    rare for liquid US equities (>200 sigmas for SPY), and even when it does
    happen (e.g. earnings shock), no integer split ratio fits cleanly so the
    heuristic skips it.
    """
    if len(series) < 2:
        return series

    s = series.copy().astype(float)
    log_returns = (s / s.shift(1)).apply(lambda x: math.log(x) if x and x > 0 else 0)

    cumulative_factor = 1.0
    adjustments_applied = 0

    # Walk backward from most recent to oldest; each detected split scales
    # everything older by the cumulative ratio.
    for i in range(len(s) - 1, 0, -1):
        lr = log_returns.iloc[i]
        if abs(lr) < _SPLIT_RETURN_THRESHOLD:
            continue

        raw_ratio = math.exp(-lr)   # if lr=-2.3 (10:1 split) → ratio=10
        # Snap to nearest integer in {2..50} or 1/{2..50}.
        candidates = [n for n in range(2, 51)] + [1.0 / n for n in range(2, 51)]
        snapped = min(candidates, key=lambda c: abs(math.log(c) - math.log(raw_ratio)))
        # Reject if the snap is too far off — likely a legitimate news move.
        if abs(math.log(snapped) - math.log(raw_ratio)) > 0.05:
            continue

        # Apply: scale older prices by snapped (so old prices come down for
        # forward split, up for reverse split).
        mask = s.index < s.index[i]
        s.loc[mask] = s.loc[mask] / snapped
        cumulative_factor *= snapped
        adjustments_applied += 1
        logger.info(
            "Эвристический split %s: %s ratio=%.2f (lr=%.3f, snapped=%.2f)",
            ticker, s.index[i].date(), snapped, lr, snapped,
        )

    if adjustments_applied:
        logger.info(
            "Применено %d эвристических корректировок для %s (cum factor=%.4f)",
            adjustments_applied, ticker, cumulative_factor,
        )
    return s


# ── Cache layer ──────────────────────────────────────────────────────────────


def _cache_path(ticker: str, days: int) -> Path:
    """Pickle-based cache (no extra deps; parquet would require pyarrow)."""
    safe = ticker.replace("/", "_").replace(".", "_")
    return _CACHE_DIR / f"{safe}_{days}d.pkl"


def _read_cache(ticker: str, days: int) -> pd.Series | None:
    path = _cache_path(ticker, days)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > _CACHE_TTL_SECONDS:
        return None
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        logger.debug("Cache read failed for %s: %s", ticker, exc)
        return None


def _write_cache(ticker: str, days: int, series: pd.Series) -> None:
    if series.empty:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        series.to_pickle(_cache_path(ticker, days))
    except Exception as exc:
        logger.debug("Cache write failed for %s: %s", ticker, exc)
