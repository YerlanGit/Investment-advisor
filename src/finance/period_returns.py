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


__all__ = [
    "PERIOD_WINDOWS_TDAYS",
    "_cum_simple_from_log",
    "compute_period_returns_table",
]
