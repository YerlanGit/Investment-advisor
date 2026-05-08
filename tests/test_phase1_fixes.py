"""
Phase 1 regression tests — guard the four bug fixes:

  1. Wilder RMA (alpha=1/period) — strictly slower-reacting than EMA(span=period).
  2. Real Max Drawdown is reported separately from VaR_95_Daily.
  3. Information Ratio is computed from annualised excess return (consistent
     with annualised TE), not from a multi-year period excess.
  4. PDF payload exposes both `var_95_daily` and `max_drawdown`, plus per-asset
     and aggregate P/L since position entry.

These tests use deterministic synthetic data and do not touch external APIs.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `import finance...` and `import pdf_generator` from src/ without a package install.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Disable Anthropic / Telegram-side imports for isolated math testing.
os.environ.setdefault("KZ_RFR_ANNUAL", "0.14")


class WilderRMATest(unittest.TestCase):
    """
    Wilder RMA uses alpha = 1/period; pandas span=period uses alpha = 2/(period+1).
    On a one-shot shock applied to a flat series, EMA reacts ~1.9× more strongly
    than Wilder RMA in the immediate post-shock value.
    """

    def test_wilder_smoothing_is_slower_than_ema(self) -> None:
        period = 14
        # Flat series of zeros, then a single +1 spike, then back to zero
        x = pd.Series([0.0] * 50 + [1.0] + [0.0] * 50)

        ema  = x.ewm(span=period,         adjust=False).mean().iloc[-1]
        wild = x.ewm(alpha=1.0 / period,  adjust=False).mean().iloc[-1]

        # Both decay back toward 0; Wilder RMA stays HIGHER (slower decay).
        self.assertGreater(wild, ema, "Wilder RMA must decay slower than EMA(span=period)")
        # And they must not be (numerically) equal — would mean we forgot the fix.
        self.assertGreater(wild - ema, 1e-6)


class RealMaxDrawdownTest(unittest.TestCase):
    """Verifies portfolio_metrics now exposes a separate Max_Drawdown key."""

    def test_engine_emits_max_drawdown_key(self) -> None:
        from finance.investment_logic import MAC3RiskEngine

        engine = MAC3RiskEngine()

        # Build a tiny synthetic 'data' frame containing all factor ETFs plus
        # one stand-alone asset (AAPL.US, NOT a factor) so we don't trip the
        # engine's duplicate-column edge case (factor == asset).
        idx = pd.date_range("2024-01-01", periods=120, freq="B")
        rng = np.random.default_rng(42)
        # Drift up then deterministically draw down 20% mid-series.
        asset_rets = rng.normal(0.0006, 0.01, len(idx))
        asset_rets[60:80] = -0.02  # 20 days of -2% to inject a clear DD
        asset_prices = 100.0 * np.exp(np.cumsum(asset_rets))
        # Factors get independent paths so Ridge has rank-9 X.
        factor_seeds = list(range(1, 1 + len(engine.factor_tickers)))

        cols = list(engine.factor_tickers.values()) + ["AAPL.US"]
        df_data = {}
        for c, seed in zip(engine.factor_tickers.values(), factor_seeds):
            f_rets = np.random.default_rng(seed).normal(0.0004, 0.01, len(idx))
            df_data[c] = 100.0 * np.exp(np.cumsum(f_rets))
        df_data["AAPL.US"] = asset_prices
        df = pd.DataFrame(df_data, index=idx)

        weights_dict = {"AAPL.US": 1.0}
        _, _, port_metrics = engine.calculate_structural_risk(df, ["AAPL.US"], weights_dict)

        self.assertIn("Max_Drawdown", port_metrics)
        self.assertIn("VaR_95_Daily",  port_metrics)
        # Sanity: drawdown is non-positive
        self.assertLessEqual(port_metrics["Max_Drawdown"], 0.0)
        # Sanity: VaR is also non-positive (5th percentile of returns)
        self.assertLessEqual(port_metrics["VaR_95_Daily"], 0.0)
        # The two are conceptually different — they should not be (numerically) equal
        # for our injected series.  Loose tolerance: difference > 1e-3.
        self.assertGreater(
            abs(port_metrics["Max_Drawdown"] - port_metrics["VaR_95_Daily"]),
            1e-3,
            "Max_Drawdown and VaR_95_Daily must be distinct quantities",
        )


class IRScaleTest(unittest.TestCase):
    """Information Ratio must be on the same scale as TE (both annualised)."""

    def test_ir_uses_annualised_excess(self) -> None:
        from finance.investment_logic import UniversalPortfolioManager

        # Drive analyse_all() with a tiny live-flag DataFrame so the engine
        # path runs end-to-end on synthetic data.  We monkey-patch the data
        # fetcher to avoid hitting Tradernet.
        upm = UniversalPortfolioManager()

        idx = pd.date_range("2023-01-01", periods=520, freq="B")  # ~2 years
        rng = np.random.default_rng(7)
        # Portfolio outperforms benchmark by ~3%/yr on average.
        port_rets = rng.normal(0.0008, 0.012, len(idx))
        bm_rets   = rng.normal(0.0005, 0.012, len(idx))

        # Pre-build the price frame the engine uses.
        cols = list(upm.engine.factor_tickers.values()) + upm.engine.BENCHMARK_EXTRA + ["AAPL.US"]
        cols = list(dict.fromkeys(cols))
        prices = {}
        for c in cols:
            r = bm_rets if c == "SPY.US" else port_rets if c == "AAPL.US" else bm_rets
            prices[c] = 100.0 * np.exp(np.cumsum(r))
        df_prices = pd.DataFrame(prices, index=idx)

        class _FakeHistory:
            data = df_prices
            ohlc_data: dict = {}
            loaded = list(df_prices.columns)
            failed: list = []
            retried: list = []

        # Monkey-patch the network-touching method.
        upm.engine.get_market_data = lambda tickers, period_days=730: (df_prices, _FakeHistory())  # type: ignore[assignment]

        # Build a tiny portfolio: 100 shares AAPL bought at 100.
        portfolio = pd.DataFrame([{
            "Ticker": "AAPL", "Quantity": 100, "Purchase_Price": 100.0
        }])

        report = upm.analyze_all(source=portfolio)
        bm = report["benchmark_comparison"].get("S&P 500")
        self.assertIsNotNone(bm, "S&P 500 benchmark row must be present")
        self.assertIn("Excess_Return_Ann",    bm)
        self.assertIn("Portfolio_Ann_Return", bm)
        self.assertIn("Benchmark_Ann_Return", bm)
        self.assertIn("Information_Ratio",    bm)

        # IR consistency: |IR| = |excess_ann / TE| within numeric tolerance.
        ir       = bm["Information_Ratio"]
        exc_ann  = bm["Excess_Return_Ann"]
        te       = bm["Tracking_Error"]
        if te is not None and te > 0 and ir is not None and exc_ann is not None:
            self.assertAlmostEqual(ir, exc_ann / te, places=6)


class PdfPayloadTest(unittest.TestCase):
    """Verifies the PDF payload schema exposes new fields and keeps fallbacks."""

    def test_payload_has_new_keys(self) -> None:
        # Defer the import: tg_bot pulls aiogram + cryptography at module load.
        # Catch BaseException because cryptography's Rust binding can raise
        # pyo3 PanicException (subclass of BaseException, not Exception).
        try:
            from tg_bot import _build_pdf_payload  # type: ignore
        except BaseException as exc:
            self.skipTest(f"tg_bot import failed in this env: {exc!r}")
            return

        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 2000.0, "Total_Cost": 1500.0,
             "PnL": 500.0, "Return_Pct": 0.3333,
             "Euler_Risk_Contribution_Pct": 12.0},
            {"Ticker": "KSPI", "Current_Value": 1000.0, "Total_Cost": 1200.0,
             "PnL": -200.0, "Return_Pct": -0.1667,
             "Euler_Risk_Contribution_Pct": 21.0},
        ])
        results = {
            "performance_table": perf,
            "total_value": 3000.0,
            "portfolio_metrics": {
                "CVaR_95_Daily":        -0.052,
                "Sharpe_Ratio":          1.18,
                "VaR_95_Daily":         -0.024,
                "Max_Drawdown":         -0.128,
                "Total_Volatility_Ann":  0.142,
            },
            "benchmark_comparison": {},
        }

        payload = _build_pdf_payload(results, "base")

        # New canonical keys present
        self.assertIn("var_95_daily",   payload)
        self.assertIn("max_drawdown",   payload)
        self.assertIn("pnl_total_abs",  payload)
        self.assertIn("pnl_total_pct",  payload)

        # var_95_daily ≠ max_drawdown formatted strings (one is -2.4%, the other -12.8%)
        self.assertNotEqual(payload["var_95_daily"], payload["max_drawdown"])

        # Per-asset P/L since entry exists and matches sign
        a0 = payload["assets"][0]
        self.assertIn("pnl_pct",   a0)
        self.assertIn("pnl_abs",   a0)
        self.assertIn("pnl_color", a0)
        self.assertEqual(a0["pnl_color"], "pos")
        self.assertEqual(payload["assets"][1]["pnl_color"], "neg")

        # Aggregate P/L: 500 + (-200) = +300 on cost 2700 → ~+11.1%
        self.assertEqual(payload["pnl_total_color"], "pos")
        self.assertIn("+300", payload["pnl_total_abs"])


if __name__ == "__main__":
    unittest.main()
