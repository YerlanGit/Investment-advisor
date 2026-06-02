"""
Multi-period accumulated return helpers (1м / 3м / 6м / 12м / YTD).

Kept in a separate module from `investment_logic` because the math is pure
numpy/pandas — it does not need scikit-learn, Ridge regression or any of the
heavier engine dependencies.  This lets us unit-test the period computation
in environments where sklearn isn't installed (CI / minimal dev shells).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Trading-day windows for the standard "1м / 3м / 6м / 12м" panel.
# Counts are in TRADING days (≈ 21·n_months) because our return series is
# sampled per trading day — using calendar days would silently include
# weekends in the window length.
PERIOD_WINDOWS_TDAYS: list[tuple[str, int]] = [
    ("1m",  21),
    ("3m",  63),
    ("6m",  126),
    ("12m", 252),
]


def _cum_simple_from_log(window_log) -> float | None:
    """
    Convert a window of *daily log-returns* to a *simple* cumulative return.

    log-return aggregation is additive: r_simple = exp(Σ r_log) − 1.
    The caller is responsible for slicing the right window — this function
    makes no assumption about anchoring or length.

    NaN values inside the window are dropped so a single missing day does
    not poison the entire period — but if every entry is NaN we return None
    rather than a misleading 0.
    """
    if window_log is None:
        return None
    arr = np.asarray(window_log, dtype=float)
    if arr.size == 0:
        return None
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return float(np.exp(arr.sum()) - 1.0)


def compute_period_returns_table(port_log: pd.Series,
                                  bm_logs: dict[str, pd.Series]) -> dict:
    """
    Build a {benchmark_name: {periods: [...], window_*}} dict of multi-period
    accumulated returns (1m / 3m / 6m / 12m / YTD).

    Each benchmark is *inner-joined* against the portfolio series before the
    periodic windows are sliced — this is the same alignment policy used by
    the Tracking-Error / IR computation in `analyze_all`, so all returns in
    the same row share an identical date set and cannot drift due to
    different cleaning policies.

    Args:
        port_log : pd.Series of daily portfolio log-returns indexed by date.
        bm_logs  : dict of {bm_name: pd.Series of daily benchmark log-returns}.

    Returns:
        dict keyed by benchmark display name.  Each value contains:
          - periods: list[{period, n_days, port_pct, bm_pct, excess_pp}]
                     port_pct / bm_pct are SIMPLE returns (decimal).
                     excess_pp is in absolute percentage *points* (decimal).
          - window_start / window_end: ISO dates of the aligned window.
          - n_days_total: number of trading days in the inner-joined window.
        Returns {} when port_log is empty.
    """
    if port_log is None or port_log.empty:
        return {}

    out: dict[str, dict] = {}
    for bm_name, bm_log in bm_logs.items():
        if bm_log is None or bm_log.empty:
            continue
        aligned = pd.concat([port_log, bm_log], axis=1, join="inner").dropna()
        if aligned.empty:
            continue
        aligned.columns = ["port", "bm"]

        rows: list[dict] = []
        port_arr = aligned["port"].values
        bm_arr   = aligned["bm"].values
        for label, ndays in PERIOD_WINDOWS_TDAYS:
            if len(port_arr) < ndays:
                p_ret = b_ret = None
            else:
                p_ret = _cum_simple_from_log(port_arr[-ndays:])
                b_ret = _cum_simple_from_log(bm_arr[-ndays:])
            excess = (p_ret - b_ret) if (p_ret is not None and b_ret is not None) else None
            rows.append({
                "period":    label,
                "n_days":    ndays,
                "port_pct":  p_ret,    # decimal (e.g. 0.052 == +5.2%)
                "bm_pct":    b_ret,
                "excess_pp": excess,   # decimal-points (port − bm)
            })

        # YTD — calendar-anchored on the most recent observation's year.
        year      = aligned.index[-1].year
        ytd_start = pd.Timestamp(year=year, month=1, day=1)
        ytd_win   = aligned.loc[aligned.index >= ytd_start]
        if ytd_win.empty:
            p_ytd = b_ytd = None
        else:
            p_ytd = _cum_simple_from_log(ytd_win["port"].values)
            b_ytd = _cum_simple_from_log(ytd_win["bm"].values)
        rows.append({
            "period":    "YTD",
            "n_days":    int(len(ytd_win)) if not ytd_win.empty else 0,
            "port_pct":  p_ytd,
            "bm_pct":    b_ytd,
            "excess_pp": (p_ytd - b_ytd) if (p_ytd is not None and b_ytd is not None) else None,
        })

        out[bm_name] = {
            "periods":      rows,
            "window_start": aligned.index[0].date().isoformat(),
            "window_end":   aligned.index[-1].date().isoformat(),
            "n_days_total": int(len(aligned)),
        }
    return out


# ── Sparse-history-robust portfolio / benchmark return series ───────────────
#
# Root problem this section solves:
#   Building a *weighted portfolio* return series needs every constituent to
#   have a price on the same day.  The naive `prices.dropna()` drops a row if
#   ANY column is NaN — so a single thinly-traded instrument with short or
#   gappy history (e.g. a local exchange listing like `FFSPC6.1028.AIX`)
#   collapses the whole overlap window and silently nulls Tracking-Error /
#   Information-Ratio / multi-period returns for the ENTIRE portfolio.
#
# Policy — degrade per-asset, never globally:
#   • an instrument needs >= MIN_OVERLAP_TDAYS price points to enter the
#     cross-asset return panel; sparse names are dropped and reported, not
#     allowed to shrink everyone else's window;
#   • the surviving weights are renormalised so the series still represents
#     a fully-invested book;
#   • every benchmark is aligned by a PER-PAIR inner-join, so one benchmark's
#     own short history can't null the others.

MIN_OVERLAP_TDAYS = 60   # ≈ 3 trading months — floor for a usable return series


def build_portfolio_log_returns(price_df: "pd.DataFrame | None",
                                 weights: dict,
                                 *,
                                 min_obs: int = MIN_OVERLAP_TDAYS) -> tuple:
    """
    Weighted portfolio daily log-return series, robust to sparse constituents.

    Args:
        price_df : close-price DataFrame, columns = resolved tickers.
        weights  : {column_name: weight_decimal}.
        min_obs  : minimum non-NaN price points for a column to be kept.

    Returns:
        (port_log_series, info) where info = {
            "dropped":        [tickers excluded for short history],
            "covered_weight": Σ weight of the kept names BEFORE renormalising
                              (1.0 == the series represents the whole book),
            "n_days":         length of the resulting return series,
            "kept":           [tickers used],
        }.  port_log_series is None when nothing usable remains.
    """
    info = {"dropped": [], "covered_weight": 0.0, "n_days": 0, "kept": []}
    if price_df is None or getattr(price_df, "empty", True):
        return None, info

    keep_cols: list[str] = []
    for col in price_df.columns:
        if int(price_df[col].notna().sum()) >= min_obs:
            keep_cols.append(col)
        else:
            info["dropped"].append(str(col))
    if not keep_cols:
        return None, info

    raw_w   = {c: float(weights.get(c, 0.0)) for c in keep_cols}
    total_w = sum(raw_w.values())
    if total_w <= 1e-9:
        return None, info
    norm_w = np.array([raw_w[c] / total_w for c in keep_cols], dtype=float)

    panel   = price_df[keep_cols]
    log_ret = np.log(panel / panel.shift(1)).dropna()
    if len(log_ret) < 2:
        info["covered_weight"] = round(total_w, 4)
        return None, info

    port_log = pd.Series(log_ret.values @ norm_w, index=log_ret.index,
                         name="port_log")
    info.update(kept=keep_cols, covered_weight=round(total_w, 4),
                n_days=int(len(port_log)))
    return port_log, info


def compute_benchmark_stats(port_log: "pd.Series | None",
                            bm_log: "pd.Series | None",
                            *,
                            trading_days: int = 252) -> "dict | None":
    """
    Tracking Error / Information Ratio / annualised excess from a PER-PAIR
    inner-join of the portfolio and benchmark log-return series.

    Aligning per benchmark (rather than against one global pre-cleaned
    matrix) means a benchmark's own short history only shortens ITS row.
    Numerator (excess) and denominator (TE) are computed on the SAME aligned
    window, so the Information Ratio is scale-consistent.

    Returns a stats dict, or None when the overlap is < 2 trading days.
    """
    if port_log is None or getattr(port_log, "empty", True) or \
       bm_log is None or getattr(bm_log, "empty", True):
        return None
    aligned = pd.concat([port_log, bm_log], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return None
    p = aligned.iloc[:, 0].values
    b = aligned.iloc[:, 1].values
    # Self-consistent units: numerator and denominator BOTH on annualised
    # *log* space.  Tracking Error is std(p-b)·√T (already log-additive),
    # and the IR numerator is also taken in log space (mean(p-b)·T) so the
    # ratio is dimensionless and free of geometric/arithmetic skew.  The
    # geometric annualised display returns are kept for the caller's
    # display panels but are NOT used in the IR computation.
    diff_log         = p - b
    excess_ann_log   = float(np.mean(diff_log) * trading_days)
    te               = float(np.std(diff_log, ddof=1) * np.sqrt(trading_days))
    port_ann         = float(np.exp(float(np.mean(p)) * trading_days) - 1.0)
    bm_ann           = float(np.exp(float(np.mean(b)) * trading_days) - 1.0)
    excess_ann       = port_ann - bm_ann
    return {
        "tracking_error":    te,
        "information_ratio": (excess_ann_log / te) if te > 1e-12 else 0.0,
        "excess_ann":        excess_ann,
        "port_ann_return":   port_ann,
        "bm_ann_return":     bm_ann,
        "n_days":            int(len(aligned)),
    }


__all__ = [
    "PERIOD_WINDOWS_TDAYS",
    "MIN_OVERLAP_TDAYS",
    "_cum_simple_from_log",
    "compute_period_returns_table",
    "build_portfolio_log_returns",
    "compute_benchmark_stats",
]
