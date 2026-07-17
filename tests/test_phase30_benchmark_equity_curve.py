"""
Phase 30 — equity-curve benchmark honours the user's chosen benchmark.

Regression for the 2026-07-16 report bug: the DEEP report's equity-curve line
stayed on S&P 500 even after the user switched their benchmark to Nasdaq.

Root cause: `compute_equity_curve_series`, when a profile benchmark was present,
iterated the fixed `_BM_CONCRETE_MAP` in dict order and picked the FIRST
available ticker — S&P 500 / SPY.US, always present as the market factor — so
the user's actual choice (e.g. QQQ.US) was never used.  The benchmark-comparison
TABLE was already correct (it keys on the profile ticker); only the equity-curve
line ignored it.

Fix: analyze_all now exposes `results['profile_benchmark_ticker']`, and the
equity-curve builder draws THAT ticker's line when a profile benchmark is set.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from finance.portfolio_series import compute_equity_curve_series  # noqa: E402


def _make_results(profile_ticker, bm_names, *, include_profile_slot=True):
    idx = pd.date_range("2025-01-01", periods=300, freq="B")
    # Distinct, monotonic trajectories so each benchmark's log-returns differ.
    prices = pd.DataFrame(
        {
            "SPY.US": np.linspace(100.0, 130.0, 300),   # +30%
            "QQQ.US": np.linspace(100.0, 160.0, 300),   # +60%
            "IWM.US": np.linspace(100.0, 108.0, 300),   # +8%
        },
        index=idx,
    )
    history = SimpleNamespace(data=prices)
    port = pd.Series(
        np.random.default_rng(0).normal(0.0003, 0.01, 300), index=idx)
    bm_data = {name: {} for name in bm_names}
    if include_profile_slot:
        bm_data = {"Профильный бенчмарк": {}, **bm_data}
    return {
        "history_result": history,
        "performance_table": pd.DataFrame({"x": [1]}),
        "port_log_returns": port,
        "benchmark_comparison": bm_data,
        "profile_benchmark_ticker": profile_ticker,
    }, prices


def _expected_log(prices, ticker):
    s = prices[ticker].dropna()
    return np.log(s / s.shift(1)).dropna().values


class EquityCurveBenchmarkTest(unittest.TestCase):

    def test_profile_nasdaq_draws_nasdaq_not_sp500(self):
        """THE bug: profile=Nasdaq must draw the Nasdaq line, not S&P 500."""
        results, prices = _make_results(
            "QQQ.US", ["S&P 500", "Nasdaq 100", "Russell 2000"])
        port_log, bm_log, bm_name = compute_equity_curve_series(results)
        self.assertEqual(bm_name, "Nasdaq 100")
        # The line must be QQQ's returns, not SPY's.
        np.testing.assert_allclose(bm_log, _expected_log(prices, "QQQ.US"))
        self.assertFalse(
            np.allclose(bm_log, _expected_log(prices, "SPY.US")),
            "equity-curve line still tracks S&P 500 — the bug is not fixed")

    def test_profile_sp500_draws_sp500(self):
        results, prices = _make_results("SPY.US", ["S&P 500", "Nasdaq 100"])
        _, bm_log, bm_name = compute_equity_curve_series(results)
        self.assertEqual(bm_name, "S&P 500")
        np.testing.assert_allclose(bm_log, _expected_log(prices, "SPY.US"))

    def test_no_profile_benchmark_falls_back_to_first_available(self):
        """No profile ticker → first concrete benchmark that loaded (unchanged)."""
        results, prices = _make_results(
            None, ["S&P 500", "Nasdaq 100"], include_profile_slot=False)
        _, bm_log, bm_name = compute_equity_curve_series(results)
        self.assertEqual(bm_name, "S&P 500")
        np.testing.assert_allclose(bm_log, _expected_log(prices, "SPY.US"))

    def test_profile_ticker_missing_prices_falls_back(self):
        """Profile ticker set but its prices absent → graceful fallback, no crash."""
        results, prices = _make_results("EEM.US", ["S&P 500"])  # EEM not in prices
        _, bm_log, bm_name = compute_equity_curve_series(results)
        self.assertEqual(bm_name, "S&P 500")
        np.testing.assert_allclose(bm_log, _expected_log(prices, "SPY.US"))


if __name__ == "__main__":
    unittest.main()
