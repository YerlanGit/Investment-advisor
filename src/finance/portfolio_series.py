"""
Portfolio time-series math for report visuals (Separation of Concerns).

Sprint-1 #3: the cap-weighted return series behind the KPI sparklines and the
equity curve used to be computed INSIDE the Telegram/UI layer (tg_bot.py),
mixing finance math into the delivery module.  This module is the math core:
it consumes the engine's `results` dict and returns plain numeric series.  The
bot/renderer layer only turns those numbers into SVG — no log-returns,
rolling windows or drawdowns in the UI layer any more.

Pure numpy/pandas; no aiogram / no rendering imports.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Concrete benchmark display-name → resolved Tradernet ticker, used to pick the
# equity-curve benchmark line.  Kept here (next to the math) rather than in the
# bot so the whole computation lives in one place.
_BM_CONCRETE_MAP = {
    "S&P 500":      "SPY.US",
    "Nasdaq 100":   "QQQ.US",
    "Russell 2000": "IWM.US",
    "MSCI EM":      "EEM.US",
    "EM Bonds":     "EMB.US",
}


def _benchmark_display_name(ticker: str) -> str:
    """Human-readable name for a benchmark ticker (e.g. QQQ.US → 'Nasdaq 100').

    Uses the canonical BENCHMARK_LIST (profile_manager) so the equity-curve
    label matches the rest of the report; falls back to the bare ticker.
    """
    try:
        from profile_manager import BENCHMARK_LIST
        return BENCHMARK_LIST.get(ticker, ticker)
    except Exception:                                 # pragma: no cover - defensive
        return ticker


def compute_equity_curve_series(
    results: dict,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Portfolio vs benchmark daily LOG-return arrays for the equity-curve chart.

    Returns (port_log, bm_log, bm_name):
      • port_log : the engine's cap-weighted daily log-return series
                   (results["port_log_returns"]), finite-filtered.  None when
                   unavailable.
      • bm_log   : daily log returns of the chosen benchmark ETF, or None.
      • bm_name  : display name of the chosen benchmark, or None.

    The caller renders these (pdf_charts.equity_curve_svg); this function does
    not import any rendering code.
    """
    try:
        history = results.get("history_result")
        perf    = results.get("performance_table")
        if perf is None or getattr(perf, "empty", True) or history is None:
            return None, None, None

        prices = getattr(history, "data", None)
        if prices is None or getattr(prices, "empty", True):
            return None, None, None

        port_series = results.get("port_log_returns")
        if port_series is None or len(port_series) == 0:
            return None, None, None
        port_log = np.asarray(getattr(port_series, "values", port_series), dtype=float)
        port_log = port_log[np.isfinite(port_log)]
        if port_log.size == 0:
            return None, None, None

        bm_data = results.get("benchmark_comparison") or {}
        bm_log: Optional[np.ndarray] = None
        chosen_bm_name: Optional[str] = None

        def _bm_line(ticker: str) -> np.ndarray:
            series = prices[ticker].dropna()
            return np.log(series / series.shift(1)).dropna().values

        # 1) Honour the user's PROFILE benchmark — the actual ticker chosen in
        #    onboarding (e.g. QQQ.US for Nasdaq 100).  The previous code iterated
        #    the fixed _BM_CONCRETE_MAP in dict order and picked the FIRST
        #    available ticker (S&P 500 / SPY.US, always present as a factor), so
        #    the equity-curve line ignored the user's choice entirely
        #    (bug 2026-07-16: benchmark switched to Nasdaq → curve stayed S&P 500).
        profile_ticker = results.get("profile_benchmark_ticker")
        if (profile_ticker and "Профильный бенчмарк" in bm_data
                and profile_ticker in prices.columns):
            chosen_bm_name = _benchmark_display_name(profile_ticker)
            bm_log = _bm_line(profile_ticker)
        else:
            # 2) Fallback: the first concrete benchmark that actually loaded
            #    (used when no profile benchmark was chosen, or its prices are
            #    missing).  Iterates the comparison set, skipping the generic
            #    "Профильный бенчмарк" key (not in _BM_CONCRETE_MAP).
            for name in list(bm_data.keys()):
                ticker = _BM_CONCRETE_MAP.get(name)
                if ticker and ticker in prices.columns:
                    chosen_bm_name = name
                    bm_log = _bm_line(ticker)
                    break
        if chosen_bm_name:
            logger.info("Equity curve benchmark = %s", chosen_bm_name)
        return port_log, bm_log, chosen_bm_name
    except Exception as exc:                          # pragma: no cover - defensive
        logger.warning("compute_equity_curve_series failed: %s", exc)
        return None, None, None


def compute_kpi_trend_series(results: dict) -> Optional[dict]:
    """
    12-month CVaR / Sharpe / MaxDD trend series for the KPI sparklines.

    Window: last 252 trading days, sampled at ~12 evenly-spaced snapshots; each
    snapshot computes the metric over a trailing 60-day window of cap-weighted
    portfolio daily LOG returns.

    Returns {"cvar_pts", "sharpe_pts", "mdd_pts"} as RAW decimal series (CVaR /
    MaxDD are decimals, Sharpe a bare ratio), or None when there is not enough
    history (≥ 90 daily price obs and ≥ 3 valid snapshots).  The caller scales /
    renders; no rendering here.

    F-21 (2026-07-11): the series source is the engine's masked composite
    (results["port_log_returns"] — per-day renormalised weights, full panel;
    same series as the equity curve and the headline KPIs).  The previous
    in-module reconstruction row-dropna'd ALL held names — one listing with
    < 60 days of quotes (e.g. SPCX) collapsed the joint window below the
    minimum and every sparkline silently rendered «нет истории».  The legacy
    reconstruction is kept only as a fallback for callers that pass a results
    dict without the precomputed series.
    """
    try:
        port_lr = None
        plr = results.get("port_log_returns")
        if plr is not None and getattr(plr, "__len__", None) and len(plr) >= 90:
            port_lr = pd.Series(
                np.asarray(getattr(plr, "values", plr), dtype=float))
            port_lr = port_lr[np.isfinite(port_lr.values)]
            if len(port_lr) < 90:
                port_lr = None

        if port_lr is None:
            # Legacy fallback: rebuild from prices (pre-F-21 path).
            history = results.get("history_result")
            perf    = results.get("performance_table")
            if perf is None or getattr(perf, "empty", True) or history is None:
                return None
            prices = getattr(history, "data", None)
            if prices is None or getattr(prices, "empty", True) or len(prices) < 90:
                return None

            total_val = float(results.get("total_value") or 1.0)
            cols: list[str] = []
            weights: list[float] = []
            for _, row in perf.iterrows():
                t  = str(row.get("Ticker", "")).strip()
                cv = float(row.get("Current_Value", 0) or 0)
                if not t or cv <= 0:
                    continue
                # perf carries the broker's ORIGINAL tickers; the price frame is
                # keyed by RESOLVED tickers (e.g. AAPL → AAPL.US).  Match exact
                # first, then fall back to the base symbol before the dot.
                if t in prices.columns:
                    col = t
                else:
                    base = t.split(".")[0]
                    col  = next((c for c in prices.columns
                                 if c.split(".")[0] == base), None)
                if col is not None:
                    cols.append(col)
                    weights.append(cv / total_val)
            if not cols:
                return None

            w        = np.array(weights)
            # F-21: per-column min-history filter mirrors the engine's
            # sparse-guard so one thin listing cannot null the joint window.
            keep = [c for c in cols
                    if int(prices[c].notna().sum()) >= 60]
            if not keep:
                return None
            w = np.array([wi for c, wi in zip(cols, weights) if c in keep])
            daily_lr = np.log(prices[keep] / prices[keep].shift(1)).dropna()
            if len(daily_lr) < 60:
                return None
            port_lr = (daily_lr * w).sum(axis=1)

        if len(port_lr) > 252:
            port_lr = port_lr.iloc[-252:]

        n    = len(port_lr)
        step = max(1, n // 12)
        if step < 5:
            return None
        snap_indices = [step * (i + 1) - 1 for i in range(12) if step * (i + 1) - 1 < n]

        cvar_pts, sharpe_pts, mdd_pts = [], [], []
        for end in snap_indices:
            win = port_lr.iloc[max(0, end - 60):end + 1]
            if len(win) < 30:
                continue
            cutoff = max(1, int(len(win) * 0.05))
            cvar_pts.append(float(win.sort_values().iloc[:cutoff].mean()))
            std = float(win.std())
            sharpe_pts.append(float(win.mean()) / std * (252 ** 0.5) if std > 0 else 0.0)
            # `win` holds LOG returns — reconstruct equity with exp(cumsum).
            eq = np.exp(win.cumsum())
            dd = (eq / eq.cummax() - 1).min()
            mdd_pts.append(float(dd))

        if len(cvar_pts) < 3:
            return None
        return {"cvar_pts": cvar_pts, "sharpe_pts": sharpe_pts, "mdd_pts": mdd_pts}
    except Exception as exc:                          # pragma: no cover - defensive
        logger.warning("compute_kpi_trend_series failed: %s", exc)
        return None


__all__ = ["compute_equity_curve_series", "compute_kpi_trend_series"]
