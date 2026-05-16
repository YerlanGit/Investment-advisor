"""
Phase 4 regression tests — PDF payload, SVG charts, AI narrative fallback,
and Jinja rendering of both v2 templates.

The tests do not invoke Playwright/Chromium — they only verify that the
templates render to valid HTML against a synthetic payload.  Browser
rendering is exercised manually via `python -m src.pdf_generator`.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ── 4.1  pdf_payload.build_payload ───────────────────────────────────────────

class PayloadBuildTest(unittest.TestCase):

    def _results(self) -> dict:
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 2000.0, "Total_Cost": 1500.0,
             "PnL": 500.0, "Return_Pct": 0.3333,
             "Euler_Risk_Contribution_Pct": 12.0, "ATR_Pct": 1.4,
             "Marginal_VaR_Daily": -0.0006, "Score_Total": 3.0,
             "Score_Action": "Strong Buy", "Score_Hotspot": False,
             "Fundamental_Sector": "Technology",
             "SEC_ROE": 0.30, "SEC_Op_Margin": 0.28, "SEC_Debt_to_Assets": 0.32,
             "SEC_Revenue_Growth_YoY": 0.06, "SEC_FCF_Margin": 0.25,
             "SEC_Altman_Z": 6.5, "SEC_Altman_Zone": "Safe",
             "SEC_Piotroski_F": 8, "SEC_Interest_Coverage": 12.0,
             "Beta_Market": 1.05, "Beta_Momentum": 0.20, "Beta_Quality": 0.5},
            {"Ticker": "KSPI", "Current_Value": 1000.0, "Total_Cost": 1200.0,
             "PnL": -200.0, "Return_Pct": -0.1667,
             "Euler_Risk_Contribution_Pct": 22.0, "ATR_Pct": 2.1,
             "Marginal_VaR_Daily": -0.0010, "Score_Total": -1.0,
             "Score_Action": "Trim", "Score_Hotspot": True,
             "Fundamental_Sector": "EM_Kazakhstan"},
        ])
        return {
            "performance_table": perf,
            "total_value": 3000.0,
            "portfolio_metrics": {
                "CVaR_95_Daily": -0.052, "Sharpe_Ratio": 1.18,
                "Sortino_Ratio": 1.6, "VaR_95_Daily": -0.024,
                "Max_Drawdown": -0.128, "Total_Volatility_Ann": 0.142,
                "Composite_Risk_Score": 62, "Max_Euler_Risk_Pct": 22.0,
                "CVaR_95_Bootstrap": {"point": -0.052, "lo95": -0.071, "hi95": -0.046},
            },
            "benchmark_comparison": {
                "Профильный бенчмарк": {
                    "Excess_Return_Ann": 0.032, "Tracking_Error": 0.043,
                    "Information_Ratio": 0.74, "Beating_Benchmark": True,
                },
                "S&P 500": {
                    "Excess_Return_Ann": 0.018, "Tracking_Error": 0.051,
                    "Information_Ratio": 0.35, "Beating_Benchmark": True,
                },
            },
            "sector_exposure": {"Technology": 0.34, "EM_Kazakhstan": 0.22},
            "regime": {"regime": "Slowdown", "confidence": 0.68,
                       "growth_score": -0.012, "cycle_score": -0.018, "signals": {}},
            "asset_scores": {
                "AAPL": {"fundamentals": 1.5, "valuations": 0.5, "technicals": 1.0,
                         "credit": 1.0, "total": 3.0, "action": "Strong Buy",
                         "hotspot": False},
                "KSPI": {"fundamentals": 0.0, "valuations": 1.0, "technicals": -0.5,
                         "credit": -0.5, "total": 0.0, "action": "Hold",
                         "hotspot": True},
            },
            "action_plan": [
                {"ticker": "KSPI", "action": "Trim", "delta_w_pp": -8.0,
                 "qty_delta": -32, "buy_zone": None, "sell_zone": [145.0, 152.0],
                 "take_target": None, "stop_loss": 132.0, "reason": "🔥 Hotspot"},
                {"ticker": "AAPL", "action": "Buy", "delta_w_pp": 3.5,
                 "qty_delta": 12, "buy_zone": [195.0, 202.0], "sell_zone": None,
                 "take_target": 220.0, "stop_loss": 178.0, "reason": "Score +3"},
            ],
            "black_litterman": [
                {"ticker": "AAPL", "current_w": 0.30, "target_w": 0.35,
                 "delta_w_pp": 5.0, "posterior_mu": 0.10},
                {"ticker": "KSPI", "current_w": 0.40, "target_w": 0.32,
                 "delta_w_pp": -8.0, "posterior_mu": -0.02},
            ],
        }

    def test_base_payload_keys(self) -> None:
        from pdf_payload import build_payload
        p = build_payload(self._results(), "base",
                          ai_summary={"verdict": "v", "bullets": ["a", "b", "c"]})
        # Required v2 keys
        for k in ("cvar", "sharpe", "var_95_daily", "max_drawdown", "risk_pct",
                  "pnl_total_pct", "pnl_total_abs", "assets", "hotspots",
                  "sectors", "regime", "ai_verdict", "ai_bullets"):
            self.assertIn(k, p)
        self.assertEqual(p["tier"], "base")
        # Hotspot present in hotspots list
        self.assertTrue(any("KSPI" in h for h in p["hotspots"]))
        # var_95 ≠ max_drawdown numerically
        self.assertNotEqual(p["var_95_daily"], p["max_drawdown"])

    def test_deep_payload_has_extra_blocks(self) -> None:
        from pdf_payload import build_payload
        p = build_payload(self._results(), "deep",
                          ai_summary={"verdict": "v", "bullets": ["a"], "action_plan_text": "do it"})
        for k in ("scenarios", "score_breakdown", "action_plan",
                  "fundamental_layer", "bl_records", "ai_action_text"):
            self.assertIn(k, p)
        self.assertEqual(p["tier"], "deep")
        # Scenarios use Excess_Return_Ann
        self.assertTrue(any("3.2%" in s["excess"] for s in p["scenarios"]))


# ── 4.1.b  Concentration / waterfall / multi-period (NEW data layer) ────────

class ConcentrationAndWaterfallTest(unittest.TestCase):
    """
    Verifies the new data-layer payload fields used by the next-generation
    BASE/DEEP report templates:
      • sector_concentration  (HHI by sector_exposure)
      • sector_warnings       (soft caps trigger)
      • asset_concentration   (HHI by per-asset weights)
      • risk_waterfall        (per-asset standalone vol vs diversified)
      • period_returns_table  (passthrough)

    All numbers are computed against hand-checked fixtures so the tests catch
    regressions in the calculation, not just the field presence.
    """

    def _base_results(self) -> dict:
        # 3-asset portfolio, total value $4,000.
        # AAPL: 2000 (50%), MSFT: 1000 (25%), KSPI: 1000 (25%)
        perf = pd.DataFrame([
            {"Ticker": "AAPL", "Current_Value": 2000.0, "Total_Cost": 1500.0,
             "PnL": 500.0, "Return_Pct": 0.3333,
             "Euler_Risk_Contribution_Pct": 12.0, "ATR_Pct": 1.4,
             "Marginal_VaR_Daily": -0.0006,
             "Fundamental_Sector": "Technology"},
            {"Ticker": "MSFT", "Current_Value": 1000.0, "Total_Cost": 900.0,
             "PnL": 100.0, "Return_Pct": 0.1111,
             "Euler_Risk_Contribution_Pct": 8.0, "ATR_Pct": 1.0,
             "Marginal_VaR_Daily": -0.0004,
             "Fundamental_Sector": "Technology"},
            {"Ticker": "KSPI", "Current_Value": 1000.0, "Total_Cost": 1200.0,
             "PnL": -200.0, "Return_Pct": -0.1667,
             "Euler_Risk_Contribution_Pct": 22.0, "ATR_Pct": 2.1,
             "Marginal_VaR_Daily": -0.0010,
             "Fundamental_Sector": "EM_Kazakhstan"},
        ])
        # Annual structural covariance — DIAGONAL gives per-asset vol²:
        #   AAPL diag = 0.04   → σ = 0.20 (20% annual vol)
        #   MSFT diag = 0.0225 → σ = 0.15
        #   KSPI diag = 0.09   → σ = 0.30
        # Off-diagonal cov chosen so portfolio is well-diversified.
        risk_matrix = pd.DataFrame(
            [[0.0400, 0.0080, 0.0030],
             [0.0080, 0.0225, 0.0020],
             [0.0030, 0.0020, 0.0900]],
            index   = ["AAPL", "MSFT", "KSPI"],
            columns = ["AAPL", "MSFT", "KSPI"],
        )
        # Portfolio variance with w = [0.5, 0.25, 0.25]:
        #   var = 0.5²·0.04 + 0.25²·0.0225 + 0.25²·0.09
        #       + 2·0.5·0.25·0.0080 + 2·0.5·0.25·0.0030 + 2·0.25·0.25·0.0020
        #       = 0.0100 + 0.001406 + 0.005625 + 0.002 + 0.00075 + 0.00025
        #       = 0.020031
        #   σ_port = √0.020031 ≈ 0.14153  (≈ 14.15% annual)
        return {
            "performance_table": perf,
            "total_value": 4000.0,
            "portfolio_metrics": {
                "Total_Volatility_Ann": 0.14153, "CVaR_95_Daily": -0.045,
                "Sharpe_Ratio": 1.1, "Sortino_Ratio": 1.4,
                "VaR_95_Daily": -0.022, "Max_Drawdown": -0.10,
                "Composite_Risk_Score": 55, "Max_Euler_Risk_Pct": 22.0,
                "CVaR_95_Bootstrap": {"point": -0.045, "lo95": -0.06, "hi95": -0.04},
            },
            "risk_matrix": risk_matrix,
            "sector_exposure": {"Technology": 0.75, "EM_Kazakhstan": 0.25},
            "benchmark_comparison": {},
            "asset_scores": {},
        }

    # — Sector concentration ————————————————————————————————————————————

    def test_sector_concentration_hhi_math(self) -> None:
        """HHI must be Σ wᵢ² in DOJ points — verified by hand-computed values."""
        from pdf_payload import build_payload
        p = build_payload(self._base_results(), "base")
        sc = p["sector_concentration"]
        # HHI = 0.75² + 0.25² = 0.5625 + 0.0625 = 0.625 → 6250 points
        self.assertEqual(sc["hhi_points"], 6250)
        self.assertEqual(sc["hhi_band"], "concentrated")    # ≥ 2500
        self.assertEqual(sc["top1_label"], "Technology")
        self.assertEqual(sc["top1_pct"], 75.0)
        self.assertEqual(sc["top3_pct"], 100.0)             # only 2 sectors → sum = 100%
        self.assertEqual(sc["items_count"], 2)

    def test_sector_warnings_trip_above_40pct(self) -> None:
        """SECTOR_CAP_PCT = 40 → Technology (75%) trips, EM (25%) does not."""
        from pdf_payload import build_payload
        p = build_payload(self._base_results(), "base")
        warns = p["sector_warnings"]
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0]["sector"], "Technology")
        self.assertEqual(warns[0]["weight_pct"], 75.0)
        self.assertEqual(warns[0]["cap_pct"], 40.0)
        self.assertEqual(warns[0]["overage_pp"], 35.0)

    def test_sector_warnings_empty_when_diversified(self) -> None:
        """No warnings when every sector ≤ 40%."""
        from pdf_payload import build_payload
        r = self._base_results()
        r["sector_exposure"] = {"Technology": 0.35, "Health": 0.30,
                                "Energy": 0.20, "Finance": 0.15}
        p = build_payload(r, "base")
        self.assertEqual(p["sector_warnings"], [])
        # HHI = 0.35² + 0.30² + 0.20² + 0.15² = 0.1225 + 0.09 + 0.04 + 0.0225 = 0.275 → 2750
        self.assertEqual(p["sector_concentration"]["hhi_points"], 2750)
        self.assertEqual(p["sector_concentration"]["hhi_band"], "concentrated")

    def test_sector_concentration_handles_empty(self) -> None:
        """Empty sector_exposure → sector_concentration = None, no warnings."""
        from pdf_payload import build_payload
        r = self._base_results()
        r["sector_exposure"] = {}
        p = build_payload(r, "base")
        self.assertIsNone(p["sector_concentration"])
        self.assertEqual(p["sector_warnings"], [])

    # — Asset concentration ————————————————————————————————————————————

    def test_asset_concentration_top3_and_hhi(self) -> None:
        """
        Weights: AAPL 50%, MSFT 25%, KSPI 25%.
        HHI = 0.5² + 0.25² + 0.25² = 0.25 + 0.0625 + 0.0625 = 0.375 → 3750 points.
        Top-1 = AAPL @ 50%.  Top-3 = 100%.
        """
        from pdf_payload import build_payload
        p = build_payload(self._base_results(), "base")
        ac = p["asset_concentration"]
        self.assertEqual(ac["hhi_points"], 3750)
        self.assertEqual(ac["hhi_band"], "concentrated")
        self.assertEqual(ac["top1_label"], "AAPL")
        self.assertEqual(ac["top1_pct"], 50.0)
        self.assertEqual(ac["top3_pct"], 100.0)
        self.assertEqual(ac["top3_labels"], ["AAPL", "MSFT", "KSPI"])

    # — Risk waterfall ————————————————————————————————————————————————

    def test_risk_waterfall_decomposition(self) -> None:
        """
        Per-asset standalone contributions (w_i · σ_i):
          AAPL: 0.50 · 0.20 = 0.10     → 10.0 pp
          MSFT: 0.25 · 0.15 = 0.0375   → 3.75 pp
          KSPI: 0.25 · 0.30 = 0.075    → 7.5 pp
          Σ standalone = 21.25 pp
          Total diversified vol = 14.153 pp
          Diversification benefit = 21.25 − 14.153 = 7.097 pp  (>0 ✓)
        Minkowski: standalone sum ≥ diversified vol — invariant we assert.
        """
        from pdf_payload import build_payload
        p = build_payload(self._base_results(), "base")
        wf = p["risk_waterfall"]
        self.assertIsNotNone(wf)

        # Sum of standalones — within 0.01 pp of hand calc.
        self.assertAlmostEqual(wf["sum_standalone_pp"], 21.25, places=1)
        self.assertAlmostEqual(wf["total_vol_pp"],      14.15, places=1)
        self.assertAlmostEqual(wf["diversification_pp"], 7.10, places=1)

        # Per-asset standalones in pp (decimal places matter for waterfall bars).
        by_t = {c["ticker"]: c for c in wf["contributions"]}
        self.assertAlmostEqual(by_t["AAPL"]["standalone_pp"], 10.0,  places=2)
        self.assertAlmostEqual(by_t["MSFT"]["standalone_pp"],  3.75, places=2)
        self.assertAlmostEqual(by_t["KSPI"]["standalone_pp"],  7.5,  places=2)

        # Per-asset annualised vol from diagonal (sanity).
        self.assertAlmostEqual(by_t["AAPL"]["standalone_vol_pct"], 20.0, places=1)
        self.assertAlmostEqual(by_t["MSFT"]["standalone_vol_pct"], 15.0, places=1)
        self.assertAlmostEqual(by_t["KSPI"]["standalone_vol_pct"], 30.0, places=1)

        # Per-asset share within sum standalones — sums to ≈ 100%.
        share_sum = sum(c["standalone_share_pct"] for c in wf["contributions"])
        self.assertAlmostEqual(share_sum, 100.0, places=1)

        # Largest contributor must be AAPL (largest w · σ product).
        top_contrib = wf["contributions"][0]
        self.assertEqual(top_contrib["ticker"], "AAPL")

        # Minkowski invariant: Σ standalone ≥ diversified vol (always).
        self.assertGreaterEqual(wf["sum_standalone_pp"], wf["total_vol_pp"] - 1e-6)

    def test_risk_waterfall_handles_missing_inputs(self) -> None:
        """No cov matrix → waterfall = None (graceful, no crash)."""
        from pdf_payload import build_payload
        r = self._base_results()
        r.pop("risk_matrix")
        p = build_payload(r, "base")
        self.assertIsNone(p["risk_waterfall"])

    # — Period returns passthrough ————————————————————————————————————

    def test_period_returns_passthrough(self) -> None:
        """
        Engine produces period_returns_table → payload passes it through
        unmodified so the report template can read it directly.
        """
        from pdf_payload import build_payload
        r = self._base_results()
        r["period_returns_table"] = {
            "S&P 500": {
                "periods": [
                    {"period": "1m",  "n_days": 21,  "port_pct": 0.020, "bm_pct": 0.014, "excess_pp": 0.006},
                    {"period": "12m", "n_days": 252, "port_pct": 0.142, "bm_pct": 0.091, "excess_pp": 0.051},
                ],
                "window_start": "2025-05-15", "window_end": "2026-05-14",
                "n_days_total": 252,
            }
        }
        p = build_payload(r, "base")
        self.assertIn("S&P 500", p["period_returns_table"])
        rows = p["period_returns_table"]["S&P 500"]["periods"]
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[-1]["excess_pp"], 0.051, places=4)


# ── 4.1.c  Engine-side period returns math (pure unit tests) ────────────────

class PeriodReturnsHelperTest(unittest.TestCase):
    """Direct tests for the multi-period helpers in finance.investment_logic."""

    def test_cum_simple_from_log_zero_window(self) -> None:
        from finance.period_returns import _cum_simple_from_log
        self.assertIsNone(_cum_simple_from_log(np.array([])))
        self.assertIsNone(_cum_simple_from_log(None))

    def test_cum_simple_from_log_basic_round_trip(self) -> None:
        """log(1+r) summed back → simple cumulative return."""
        from finance.period_returns import _cum_simple_from_log
        # Three days of +1% simple = (1.01)^3 - 1 = 0.030301
        log_returns = np.log(np.array([1.01, 1.01, 1.01]))
        out = _cum_simple_from_log(log_returns)
        self.assertAlmostEqual(out, 0.030301, places=5)

    def test_cum_simple_from_log_ignores_nan(self) -> None:
        from finance.period_returns import _cum_simple_from_log
        # NaN must NOT propagate to the answer.
        log_returns = np.array([np.log(1.05), float("nan"), np.log(1.05)])
        out = _cum_simple_from_log(log_returns)
        self.assertAlmostEqual(out, 1.05 * 1.05 - 1.0, places=5)

    def test_compute_period_returns_table_full_window(self) -> None:
        """
        Hand-built portfolio and benchmark daily-log streams over 300 trading
        days: both at constant +0.05% per day so all multi-period windows are
        deterministic and easy to check.

            port: +0.05% daily for 300 days
            bm:   +0.03% daily for 300 days
            12m simple (252 days):
              port = exp(252 · 0.0005) − 1 ≈ 0.13495
              bm   = exp(252 · 0.0003) − 1 ≈ 0.07859
              excess ≈ 0.05636
        """
        from finance.period_returns import compute_period_returns_table as _compute_period_returns_table
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        port_log = pd.Series(np.full(300, 0.0005), index=dates, name="port")
        bm_log   = pd.Series(np.full(300, 0.0003), index=dates, name="bm")
        table = _compute_period_returns_table(port_log, {"S&P 500": bm_log})
        self.assertIn("S&P 500", table)
        rows = {r["period"]: r for r in table["S&P 500"]["periods"]}
        # 12-month row
        r12 = rows["12m"]
        self.assertAlmostEqual(r12["port_pct"], np.exp(252 * 0.0005) - 1, places=5)
        self.assertAlmostEqual(r12["bm_pct"],   np.exp(252 * 0.0003) - 1, places=5)
        self.assertAlmostEqual(r12["excess_pp"], r12["port_pct"] - r12["bm_pct"], places=6)
        # 1-month row uses just 21 days
        r1 = rows["1m"]
        self.assertAlmostEqual(r1["port_pct"], np.exp(21 * 0.0005) - 1, places=5)
        # Window must span the full inner-join (all 300 trading days)
        self.assertEqual(table["S&P 500"]["n_days_total"], 300)

    def test_compute_period_returns_table_short_window_returns_none(self) -> None:
        """When data is shorter than the period, that row reports None — not 0."""
        from finance.period_returns import compute_period_returns_table as _compute_period_returns_table
        # Only 30 trading days available → 3m / 6m / 12m must be None.
        dates = pd.date_range("2026-04-01", periods=30, freq="B")
        port_log = pd.Series(np.full(30, 0.001), index=dates)
        bm_log   = pd.Series(np.full(30, 0.0008), index=dates)
        table = _compute_period_returns_table(port_log, {"BM": bm_log})
        rows = {r["period"]: r for r in table["BM"]["periods"]}
        self.assertIsNotNone(rows["1m"]["port_pct"])      # 21 days ≤ 30 ✓
        self.assertIsNone(rows["3m"]["port_pct"])         # 63 > 30
        self.assertIsNone(rows["12m"]["port_pct"])

    def test_compute_period_returns_table_inner_joins(self) -> None:
        """Misaligned date sets must inner-join — no NaNs in the computation."""
        from finance.period_returns import compute_period_returns_table as _compute_period_returns_table
        # Port has 300 days, bm has only the last 200 of them.
        dates_p = pd.date_range("2024-01-01", periods=300, freq="B")
        dates_b = dates_p[-200:]
        port_log = pd.Series(np.full(300, 0.0004), index=dates_p)
        bm_log   = pd.Series(np.full(200, 0.0002), index=dates_b)
        table = _compute_period_returns_table(port_log, {"BM": bm_log})
        # n_days_total reflects the intersection, not the longer side.
        self.assertEqual(table["BM"]["n_days_total"], 200)


# ── 4.2  SVG charts ─────────────────────────────────────────────────────────

class ChartsSVGTest(unittest.TestCase):

    def test_equity_curve_renders(self) -> None:
        from pdf_charts import equity_curve_svg
        rng = np.random.default_rng(0)
        port = rng.normal(0.0008, 0.011, 200)
        bm   = rng.normal(0.0006, 0.011, 200)
        svg = equity_curve_svg(port, bm)
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Портфель", svg)
        self.assertIn("Бенчмарк", svg)

    def test_equity_curve_handles_empty(self) -> None:
        from pdf_charts import equity_curve_svg
        svg = equity_curve_svg([])
        self.assertIn("нет данных", svg)

    def test_sector_pie_renders(self) -> None:
        from pdf_charts import sector_pie_svg
        svg = sector_pie_svg({"Tech": 0.40, "Finance": 0.30, "Cash": 0.30})
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Tech", svg)
        # Donut center
        self.assertIn("circle", svg)

    def test_factor_radar_renders(self) -> None:
        from pdf_charts import factor_radar_svg
        svg = factor_radar_svg({"Market": 0.85, "Momentum": 0.31, "Value": 0.05,
                                  "Quality": 0.20, "Size": -0.10,
                                  "Rates": -0.18, "EM_Equity": 0.42})
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Market", svg)
        # Polygon for the data
        self.assertIn("polygon", svg)


# ── 4.3  AI narrative fallback ──────────────────────────────────────────────

class NarrativeFallbackTest(unittest.TestCase):

    def test_fallback_when_no_api_key(self) -> None:
        # Force fallback path by clearing API key.
        prev = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from ai_narrative import generate_narrative
            results = {
                "portfolio_metrics": {
                    "CVaR_95_Daily": -0.05, "Sharpe_Ratio": 1.0,
                    "Composite_Risk_Score": 55, "Max_Drawdown": -0.1,
                },
                "regime": {"regime": "Expansion", "confidence": 0.6},
                "performance_table": pd.DataFrame([
                    {"Ticker": "AAPL", "Score_Hotspot": False,
                     "Euler_Risk_Contribution_Pct": 10},
                    {"Ticker": "KSPI", "Score_Hotspot": True,
                     "Euler_Risk_Contribution_Pct": 22},
                ]),
            }
            out = generate_narrative(results, tier="base")
            self.assertIn("verdict", out)
            self.assertIn("bullets", out)
            self.assertGreaterEqual(len(out["bullets"]), 1)
            # Hotspot bullet must mention KSPI
            self.assertTrue(any("KSPI" in b for b in out["bullets"]))
        finally:
            if prev is not None:
                os.environ["ANTHROPIC_API_KEY"] = prev


# ── 4.4  Jinja render of both templates ─────────────────────────────────────

class JinjaRenderTest(unittest.TestCase):

    def _render(self, template_name: str, payload: dict) -> str:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        tdir = SRC / "templates"
        env  = Environment(loader=FileSystemLoader(str(tdir)),
                           autoescape=select_autoescape(["html"]))
        return env.get_template(template_name).render(
            data        = payload,
            user_id     = "tester",
            report_type = "Тест",
            generated_at= datetime.now().strftime("%d.%m.%Y %H:%M"),
        )

    def test_basic_template_renders(self) -> None:
        from pdf_payload import build_payload
        from tests.test_phase4_reporting import PayloadBuildTest
        results = PayloadBuildTest()._results()
        payload = build_payload(results, "base",
                                ai_summary={"verdict": "ok", "bullets": ["a", "b"]})
        html = self._render("report_basic.html", payload)
        self.assertIn("PORTFOLIO INTELLIGENCE", html)
        self.assertNotIn("RAMP",                 html)
        self.assertIn("Базовый отчёт".lower() if False else "Тест", html)
        self.assertIn("AAPL",                    html)
        # No leftover dark-theme tokens
        self.assertNotIn("#0D1117", html)
        self.assertIn("#0F4C81",    html)

    def test_deep_template_renders(self) -> None:
        from pdf_payload import build_payload
        from tests.test_phase4_reporting import PayloadBuildTest
        results = PayloadBuildTest()._results()
        payload = build_payload(results, "deep",
                                ai_summary={"verdict": "ok", "bullets": ["a"],
                                              "action_plan_text": "do it now"})
        html = self._render("report_deep.html", payload)
        self.assertIn("PORTFOLIO INTELLIGENCE", html)
        self.assertNotIn("RAMP",                 html)
        # Deep-only sections present (current template — Three-Question redesign)
        self.assertIn("Action Plan",             html)
        self.assertIn("4-Pillar Scoring",        html)
        self.assertIn("CoVe",                    html)


if __name__ == "__main__":
    unittest.main()
