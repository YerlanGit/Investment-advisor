"""
Action Plan engine — converts Total Score + Black-Litterman delta-weights
into concrete, executable rows with Buy zone / Sell target / Stop loss.

Levels are derived ONLY from the signals already produced by the engine:
    • ATR (Wilder RMA, from MAC3 calculate_atr)
    • SMA50 / SMA100 / SMA200 + 52-week high (from technicals.py)
    • RSI(14) and MACD (from technicals.py)

No external 'target price' source is consulted — every level is anchored
to a technical structure on the asset's own price series so it is fully
auditable and reproducible.

Outputs are AssetActionRow objects ready to render in the PDF Action Plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── Tunable risk-management constants (intentionally conservative) ───────────
ATR_BUY_LO_MULT      = 1.0     # Buy zone lower bound  = SMA50 - 1·ATR
ATR_BUY_HI_OFFSET_RSI_HOT = 0.5  # If RSI > 75 → shift buy zone -0.5 ATR
ATR_TAKE_MULT_BUY    = 3.0     # Take-profit          = price + 3·ATR (or 1.05·SMA200)
ATR_STOP_MULT_BUY    = 2.0     # Stop                 = price - 2·ATR (or SMA200)
ATR_SELL_HI_MULT     = 1.0     # Sell zone upper      = SMA50 + 1·ATR
ATR_STOP_MULT_SELL   = 2.0     # Sell stop            = price - 2·ATR
ATR_HOLD_STOP_MULT   = 2.5     # Hold stop            = price - 2.5·ATR (or SMA100)

MAX_TRADE_BLOCK_PORTFOLIO_PCT = 0.25   # Δw rebalance cap per single report


# ── Public dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AssetActionRow:
    ticker:        str
    action:        str                     # 'Strong Buy' / 'Buy' / 'Hold' / 'Trim' / 'Sell'
    delta_w_pp:    float                   # target - current, percentage points
    qty_delta:     Optional[int]           # whole units (for display)
    price:         float
    buy_zone:      Optional[tuple[float, float]]
    sell_zone:     Optional[tuple[float, float]]
    take_target:   Optional[float]
    stop_loss:     Optional[float]
    reason:        str                     # short justification

    def as_dict(self) -> dict:
        return {
            "ticker":      self.ticker,
            "action":      self.action,
            "delta_w_pp":  round(self.delta_w_pp, 2),
            "qty_delta":   self.qty_delta,
            "price":       round(self.price, 4),
            "buy_zone":    None if self.buy_zone   is None else
                           [round(self.buy_zone[0],   4), round(self.buy_zone[1],   4)],
            "sell_zone":   None if self.sell_zone  is None else
                           [round(self.sell_zone[0],  4), round(self.sell_zone[1],  4)],
            "take_target": None if self.take_target is None else round(self.take_target, 4),
            "stop_loss":   None if self.stop_loss   is None else round(self.stop_loss,   4),
            "reason":      self.reason,
        }


# ── Level computation ────────────────────────────────────────────────────────

def compute_levels(*,
                   action: str,
                   price: float,
                   atr_abs: Optional[float],
                   sma50:   Optional[float],
                   sma100:  Optional[float],
                   sma200:  Optional[float],
                   high_52w: Optional[float],
                   rsi:     Optional[float],
                   macd_below_zero: Optional[bool] = None) -> dict:
    """
    Compute Buy zone / Sell zone / Take target / Stop loss for one asset.

    Falls back gracefully when an indicator is missing — never returns
    a fabricated price level.
    """
    # If no ATR we can't anchor risk; produce only price-based stop.
    atr = float(atr_abs) if atr_abs is not None and atr_abs > 0 else None

    # RSI-based buy-zone shift: hot tape → require deeper pullback.
    rsi_hot = rsi is not None and rsi > 75
    rsi_oversold = rsi is not None and rsi < 30
    macd_hot_negative = bool(macd_below_zero)

    out: dict = {"buy_zone": None, "sell_zone": None,
                 "take_target": None, "stop_loss": None}

    if action in ("Strong Buy", "Buy"):
        if sma50 is not None and atr is not None:
            buy_lo = sma50 - ATR_BUY_LO_MULT * atr
            buy_hi = sma50
            if rsi_hot:
                buy_lo -= ATR_BUY_HI_OFFSET_RSI_HOT * atr
                buy_hi -= ATR_BUY_HI_OFFSET_RSI_HOT * atr
            if rsi_oversold:
                # Allow entries up to current price on truly oversold tape.
                buy_hi = max(buy_hi, price)
            if macd_hot_negative:
                # Less aggressive zone if trend isn't confirming.
                buy_lo = sma50 - 0.7 * atr
                buy_hi = sma50 - 0.2 * atr
            out["buy_zone"] = (float(min(buy_lo, buy_hi)), float(max(buy_lo, buy_hi)))

        # Take-profit target: 5% above SMA200 OR price + 3·ATR, whichever is higher.
        candidates = []
        if sma200 is not None:
            candidates.append(sma200 * 1.05)
        if atr is not None:
            candidates.append(price + ATR_TAKE_MULT_BUY * atr)
        if high_52w is not None:
            candidates.append(high_52w * 1.02)   # don't sell INTO the high
        if candidates:
            out["take_target"] = float(max(candidates))

        # Stop: -2 ATR or SMA200 (whichever is higher → tighter).
        stops = []
        if atr is not None:
            stops.append(price - ATR_STOP_MULT_BUY * atr)
        if sma200 is not None:
            stops.append(sma200)
        if stops:
            out["stop_loss"] = float(max(stops))
        return out

    if action in ("Trim", "Sell"):
        if sma50 is not None and atr is not None:
            sell_lo = sma50
            sell_hi = sma50 + ATR_SELL_HI_MULT * atr
            if high_52w is not None and high_52w > sell_hi:
                # Don't lift offer past the 52w high in one swoop.
                sell_hi = max(sell_hi, min(price * 1.02, high_52w))
            out["sell_zone"] = (float(sell_lo), float(sell_hi))
        if atr is not None:
            out["stop_loss"] = float(price - ATR_STOP_MULT_SELL * atr)
        return out

    # Hold — only protective stop.
    stops = []
    if atr is not None:
        stops.append(price - ATR_HOLD_STOP_MULT * atr)
    if sma100 is not None:
        stops.append(sma100)
    if stops:
        out["stop_loss"] = float(max(stops))
    return out


# ── Plan builder ─────────────────────────────────────────────────────────────

def build_action_plan(*,
                      perf_table,                   # pandas DataFrame
                      asset_scores: dict,           # {ticker: AssetScore-like}
                      technicals_map: dict,         # {ticker: TechnicalReading}
                      bl_records: Optional[list[dict]] = None,
                      portfolio_value: float = 0.0,
                      ) -> list[AssetActionRow]:
    """
    Stitch per-asset action with quantitative levels.

    Args:
        perf_table     : 'performance_table' from analyze_all() — must include
                         'Ticker', 'Current_Price', 'Quantity'; ATR_Absolute
                         is preferred for risk units.
        asset_scores   : Either AssetScore dataclass per ticker or already a
                         dict shape (e.g. results['asset_scores']).
        technicals_map : {ticker: TechnicalReading} from compute_technicals.
        bl_records     : Optional list of records from BLResult.as_records();
                         provides delta_w_pp + posterior_mu.
        portfolio_value: Used to convert delta_w_pp to a quantity recommendation.

    Returns a list of AssetActionRow ready for the PDF Action Plan table.
    The list is sorted: Trim / Sell first (highest priority), then Strong
    Buy / Buy, then Hold.  Cumulative |delta_w| is capped at 25% of NAV.
    """
    rows: list[AssetActionRow] = []
    bl_by_ticker: dict[str, dict] = {}
    if bl_records:
        bl_by_ticker = {r["ticker"]: r for r in bl_records}

    if perf_table is None or perf_table.empty:
        return rows

    if "Ticker" in perf_table.columns:
        iter_df = perf_table
    else:
        iter_df = perf_table.reset_index()

    for _, row in iter_df.iterrows():
        ticker = str(row.get("Ticker") or "?")
        sc = asset_scores.get(ticker)
        action = (sc.get("action") if isinstance(sc, dict)
                  else getattr(sc, "action", None) if sc else None) or "Hold"

        price  = float(row.get("Current_Price") or row.get("Price") or 0.0) or 0.0
        if price <= 0:
            continue

        atr_abs = row.get("ATR_Absolute")
        atr_abs = None if atr_abs is None or (isinstance(atr_abs, float) and atr_abs != atr_abs) \
                       else float(atr_abs)

        # Pull SMA / RSI / MACD from technicals reading (when present).
        tr = technicals_map.get(ticker)
        sma_state = (tr.raw.get("sma_state") if tr else {}) or {}
        sma50  = sma_state.get("sma50")
        sma100 = sma_state.get("sma100")
        sma200 = sma_state.get("sma200")
        rsi    = (tr.raw.get("rsi14") if tr else None)
        hi52   = None
        if tr and tr.raw.get("high_52w_prox") and price:
            # high_52w_prox = price / 52w_high → 52w_high = price / prox
            prox = tr.raw["high_52w_prox"]
            if prox and prox > 0:
                hi52 = float(price / prox)
        macd = tr.raw.get("macd") if tr else None
        macd_neg = bool(macd and macd["line"] < 0 and macd["signal"] < 0)

        levels = compute_levels(
            action=action, price=price, atr_abs=atr_abs,
            sma50=sma50, sma100=sma100, sma200=sma200,
            high_52w=hi52, rsi=rsi, macd_below_zero=macd_neg,
        )

        bl = bl_by_ticker.get(ticker)
        delta_w_pp = float(bl["delta_w_pp"]) if bl else 0.0
        qty_delta = None
        if bl and portfolio_value > 0 and price > 0:
            dollars = (delta_w_pp / 100.0) * portfolio_value
            qty_delta = int(round(dollars / price))

        # Reason: short, evidence-based.
        reason_bits = []
        if sc:
            tot = sc.get("total") if isinstance(sc, dict) else getattr(sc, "total", None)
            hot = sc.get("hotspot") if isinstance(sc, dict) else getattr(sc, "hotspot", None)
            if tot is not None:
                reason_bits.append(f"Score {tot:+.1f}")
            if hot:
                reason_bits.append("🔥 Hotspot TRC>20%")
        if rsi is not None:
            if rsi > 75: reason_bits.append("RSI hot")
            elif rsi < 30: reason_bits.append("RSI oversold")
        reason = " · ".join(reason_bits) if reason_bits else action

        rows.append(AssetActionRow(
            ticker=ticker, action=action, delta_w_pp=delta_w_pp,
            qty_delta=qty_delta, price=price,
            buy_zone=levels["buy_zone"], sell_zone=levels["sell_zone"],
            take_target=levels["take_target"], stop_loss=levels["stop_loss"],
            reason=reason,
        ))

    # Order: priority sells first, then buys, then holds.
    priority = {"Sell": 0, "Trim": 1, "Strong Buy": 2, "Buy": 3, "Hold": 4}
    rows.sort(key=lambda r: priority.get(r.action, 5))

    # Apply cumulative |Δw| cap.
    cumulative = 0.0
    capped: list[AssetActionRow] = []
    for r in rows:
        if abs(r.delta_w_pp) <= 0.0:
            capped.append(r)
            continue
        cumulative += abs(r.delta_w_pp) / 100.0
        if cumulative <= MAX_TRADE_BLOCK_PORTFOLIO_PCT:
            capped.append(r)
        else:
            # Demote to Hold w/o quantity once budget is exhausted.
            capped.append(AssetActionRow(
                ticker=r.ticker, action="Hold",
                delta_w_pp=0.0, qty_delta=None, price=r.price,
                buy_zone=None, sell_zone=None,
                take_target=None, stop_loss=r.stop_loss,
                reason=r.reason + " · deferred (turnover cap)",
            ))
    return capped


__all__ = ["AssetActionRow", "compute_levels", "build_action_plan",
           "MAX_TRADE_BLOCK_PORTFOLIO_PCT"]
