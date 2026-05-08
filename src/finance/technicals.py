"""
Technical-analysis features used by pillar C (Technicals) of the scoring model.

All indicators are computed from CLOSE prices that the engine already fetches —
no extra network calls.  When a feature requires more history than is
available, it returns None and the scoring layer treats it as neutral.

Implemented indicators
----------------------
  Momentum_12m1m   : 12-month return minus the most recent month (cross-
                     sectional momentum, Asness 2013 standard).
  SMA200           : Last close vs 200-day simple moving average (trend).
  SMA_Cross        : Golden cross signal — SMA50 above SMA200.
  RSI14            : Wilder's Relative Strength Index (overbought / oversold).
  MACD             : MACD(12,26,9) — line, signal, histogram, regime.
  Bollinger_Z      : (Close - SMA20) / (2·sigma20) — mean-reversion gauge.
  HighProximity52w : Close / max(close, last 252) — strength of trend.
  VolumeConfirm    : Volume_20d / Volume_60d (only if volume is provided).

Each per-ticker call returns a dict with the raw values AND a -2..+2
"technical score" contribution that the Scoring orchestrator simply sums.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Default windows
WIN_RSI       = 14
WIN_MACD_FAST = 12
WIN_MACD_SLOW = 26
WIN_MACD_SIG  = 9
WIN_BOLL      = 20
WIN_SMA50     = 50
WIN_SMA100    = 100
WIN_SMA200    = 200
WIN_52W       = 252
WIN_MOM_LONG  = 252
WIN_MOM_SKIP  = 21
WIN_VOL_SHORT = 20
WIN_VOL_LONG  = 60


# ── Individual indicators ────────────────────────────────────────────────────

def _rsi_wilder(close: pd.Series, period: int = WIN_RSI) -> Optional[float]:
    """Wilder's RSI on closing prices.  Returns the latest value (0..100)."""
    if len(close) < period + 1:
        return None
    diff = close.diff().dropna()
    gain = diff.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else None


def _macd(close: pd.Series) -> Optional[dict]:
    """MACD(12,26,9).  Returns latest line / signal / histogram."""
    if len(close) < WIN_MACD_SLOW + WIN_MACD_SIG:
        return None
    ema_fast = close.ewm(span=WIN_MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=WIN_MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=WIN_MACD_SIG, adjust=False).mean()
    hist = macd_line - signal
    return {
        "line":   float(macd_line.iloc[-1]),
        "signal": float(signal.iloc[-1]),
        "hist":   float(hist.iloc[-1]),
    }


def _bollinger_z(close: pd.Series, win: int = WIN_BOLL) -> Optional[float]:
    """
    Position of the latest close inside Bollinger bands, expressed as
    z = (close - sma) / (2*std).  Range typically -1..+1; outside that
    range the price is beyond the standard 2-sigma band.
    """
    if len(close) < win + 1:
        return None
    sma = close.rolling(window=win).mean().iloc[-1]
    std = close.rolling(window=win).std().iloc[-1]
    if not pd.notna(std) or std == 0:
        return None
    return float((close.iloc[-1] - sma) / (2.0 * std))


def _sma_state(close: pd.Series) -> dict:
    """Trend state vs SMA50, SMA100, SMA200 plus golden-cross flag."""
    out: dict = {}
    last = float(close.iloc[-1])
    if len(close) >= WIN_SMA50:
        out["sma50"]  = float(close.rolling(WIN_SMA50).mean().iloc[-1])
    if len(close) >= WIN_SMA100:
        out["sma100"] = float(close.rolling(WIN_SMA100).mean().iloc[-1])
    if len(close) >= WIN_SMA200:
        out["sma200"] = float(close.rolling(WIN_SMA200).mean().iloc[-1])
    out["close"] = last
    if "sma50" in out and "sma200" in out:
        out["golden_cross"] = bool(out["sma50"] > out["sma200"])
    return out


def _momentum_12m1m(close: pd.Series) -> Optional[float]:
    """Asness-Moskowitz 12m-minus-1m momentum: r(t-252,t-21)."""
    if len(close) < WIN_MOM_LONG + 1:
        return None
    p_now = close.iloc[-1 - WIN_MOM_SKIP]      # one month ago
    p_old = close.iloc[-1 - WIN_MOM_LONG]      # ~12 months ago
    if not (pd.notna(p_now) and pd.notna(p_old) and p_old > 0):
        return None
    return float(p_now / p_old - 1.0)


def _high_proximity_52w(close: pd.Series) -> Optional[float]:
    """close / max(close, last 252) — 1.0 means at the 52w high."""
    if len(close) < WIN_52W:
        return None
    hi = close.iloc[-WIN_52W:].max()
    if not pd.notna(hi) or hi <= 0:
        return None
    return float(close.iloc[-1] / hi)


def _volume_confirm(volume: pd.Series) -> Optional[float]:
    """20d avg volume / 60d avg volume — >1.3 implies institutional pickup."""
    if volume is None or len(volume) < WIN_VOL_LONG:
        return None
    v20 = volume.rolling(WIN_VOL_SHORT).mean().iloc[-1]
    v60 = volume.rolling(WIN_VOL_LONG).mean().iloc[-1]
    if not (pd.notna(v20) and pd.notna(v60) and v60 > 0):
        return None
    return float(v20 / v60)


# ── Scoring (-2..+2 cap) ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class TechnicalReading:
    ticker: str
    score: float           # -2 .. +2
    components: dict       # individual contributions
    raw: dict              # raw indicator values for display


def _score_components(close: pd.Series,
                      volume: Optional[pd.Series] = None,
                      sector_momentum: Optional[float] = None) -> dict:
    """
    Convert raw indicators into a -2..+2 contribution per signal.

    The components are clipped at the end; individually they sum without cap.
    `sector_momentum` (cross-section reference) is optional — when absent we
    use a fixed ±15% threshold on the absolute Momentum_12m1m.
    """
    contrib: dict[str, float] = {}
    raw: dict[str, object] = {}

    # 1. Momentum 12m-1m
    mom = _momentum_12m1m(close)
    raw["momentum_12m1m"] = mom
    if mom is not None:
        if sector_momentum is not None:
            spread = mom - sector_momentum
            if   spread >  0.15: contrib["momentum"] = +1.0
            elif spread < -0.15: contrib["momentum"] = -1.0
            else:                contrib["momentum"] = 0.0
        else:
            if   mom >  0.15: contrib["momentum"] = +1.0
            elif mom < -0.15: contrib["momentum"] = -1.0
            else:             contrib["momentum"] = 0.0

    # 2. SMA200 trend + golden cross
    sma = _sma_state(close)
    raw["sma_state"] = sma
    if "sma200" in sma:
        if sma["close"] > sma["sma200"]:
            contrib["sma200"] = +1.0
        else:
            contrib["sma200"] = -1.0
        if sma.get("golden_cross") is True and sma["close"] > sma["sma200"]:
            contrib["sma200"] += 0.0  # already captured; no double count
        elif sma.get("golden_cross") is False and sma["close"] < sma["sma200"]:
            contrib["sma200"] += 0.0

    # 3. RSI14
    rsi = _rsi_wilder(close)
    raw["rsi14"] = rsi
    if rsi is not None:
        if   rsi > 70: contrib["rsi"] = -0.5
        elif rsi < 30: contrib["rsi"] = +0.5
        else:          contrib["rsi"] = 0.0

    # 4. MACD
    macd = _macd(close)
    raw["macd"] = macd
    if macd is not None:
        if macd["line"] > macd["signal"] and macd["line"] > 0:
            contrib["macd"] = +0.5
        elif macd["line"] < macd["signal"] and macd["line"] < 0:
            contrib["macd"] = -0.5
        else:
            contrib["macd"] = 0.0

    # 5. Bollinger position (mean-reversion)
    bz = _bollinger_z(close)
    raw["bollinger_z"] = bz
    if bz is not None:
        if   bz < -0.8: contrib["bollinger"] = +0.5
        elif bz >  0.8: contrib["bollinger"] = -0.5
        else:           contrib["bollinger"] = 0.0

    # 6. 52-week high proximity
    hp = _high_proximity_52w(close)
    raw["high_52w_prox"] = hp
    if hp is not None:
        if   hp > 0.95: contrib["hi52"] = +0.5
        elif hp < 0.70: contrib["hi52"] = -0.5
        else:           contrib["hi52"] = 0.0

    # 7. Volume confirmation (optional)
    vc = _volume_confirm(volume)
    raw["volume_ratio"] = vc
    if vc is not None:
        # Only positive contribution — high volume on its own isn't bad.
        contrib["volume"] = +0.25 if vc > 1.3 else 0.0

    raw_score = float(sum(contrib.values()))
    score = float(np.clip(raw_score, -2.0, 2.0))

    return {"score": score, "components": contrib, "raw": raw}


# ── Public API ───────────────────────────────────────────────────────────────

def compute_technicals(close_prices: pd.DataFrame,
                       tickers: list[str],
                       volume_frame: Optional[pd.DataFrame] = None,
                       sector_map: Optional[dict[str, str]] = None,
                       ) -> dict[str, TechnicalReading]:
    """
    Compute technical readings for a list of tickers.

    Args:
        close_prices : DataFrame indexed by date, columns = ticker symbols (in
                       the resolved Tradernet form, e.g. 'AAPL.US').
        tickers      : Original (unresolved) ticker names — used as the keys
                       of the returned dict for downstream alignment.
        volume_frame : Optional DataFrame in the same shape as close_prices
                       holding daily volumes.
        sector_map   : Optional {ticker: sector} for sector-relative momentum;
                       when missing a fixed ±15% threshold is used.

    Returns:
        {original_ticker: TechnicalReading}
    """
    out: dict[str, TechnicalReading] = {}

    # Cross-sectional sector momentum reference (median 12m-1m per sector).
    sector_momentum_map: dict[str, float] = {}
    if sector_map:
        per_sector: dict[str, list[float]] = {}
        for col in close_prices.columns:
            mom = _momentum_12m1m(close_prices[col].dropna())
            base = col.split(".")[0].upper() if "." in col else col.upper()
            sector = sector_map.get(base) or sector_map.get(col)
            if sector and mom is not None:
                per_sector.setdefault(sector, []).append(mom)
        sector_momentum_map = {s: float(np.median(v)) for s, v in per_sector.items() if v}

    for orig in tickers:
        # Find the matching column — accept exact match or ".XXX" suffix variant.
        col = None
        if orig in close_prices.columns:
            col = orig
        else:
            base = orig.split(".")[0].upper() if "." in orig else orig.upper()
            for c in close_prices.columns:
                if c == orig or c.split(".")[0].upper() == base:
                    col = c
                    break
        if col is None:
            continue

        close = close_prices[col].dropna()
        if len(close) < WIN_SMA50:
            continue
        volume = volume_frame[col].dropna() if (volume_frame is not None and col in volume_frame.columns) else None

        sec_ref = None
        if sector_map:
            base = orig.split(".")[0].upper() if "." in orig else orig.upper()
            sec_ref = sector_momentum_map.get(sector_map.get(base) or sector_map.get(orig))

        result = _score_components(close, volume=volume, sector_momentum=sec_ref)
        out[orig] = TechnicalReading(
            ticker     = orig,
            score      = result["score"],
            components = result["components"],
            raw        = result["raw"],
        )

    return out


__all__ = ["compute_technicals", "TechnicalReading"]
